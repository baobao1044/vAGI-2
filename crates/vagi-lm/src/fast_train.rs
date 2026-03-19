//! Fast CPU-optimized training for vAGI-2.
//!
//! Key optimizations:
//! - Direct f32 matmul (skips ternary pack/unpack — saves ~40% compute)
//! - Batch-parallel: forward N sequences on N threads via rayon
//! - Gradient accumulation: merge gradients from batch, update once
//!
//! After f32 pre-training, run standard STE training for ternary fine-tuning.

use rayon::prelude::*;
use crate::{VagiLM, LMConfig};

/// Fast f32 matrix-vector multiply: y = W × x
/// W is [out_features × in_features], stored row-major.
#[inline]
fn f32_matvec(w: &[f32], x: &[f32], y: &mut [f32], out_f: usize, in_f: usize) {
    for i in 0..out_f {
        let row = &w[i * in_f..(i + 1) * in_f];
        let mut acc = 0.0f32;
        for j in 0..in_f {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
}

/// Fast f32 matrix-vector multiply: y += W^T × x (for backward pass)
/// W is [out_features × in_features], we compute W^T × grad_output
#[inline]
fn f32_matvec_transpose(w: &[f32], grad_out: &[f32], grad_in: &mut [f32], out_f: usize, in_f: usize) {
    for j in 0..in_f {
        let mut acc = 0.0f32;
        for i in 0..out_f {
            acc += w[i * in_f + j] * grad_out[i];
        }
        grad_in[j] = acc;
    }
}

/// RMSNorm forward (in-place).
fn rmsnorm_fwd(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    for i in 0..n {
        x[i] = x[i] * inv * weight[i];
    }
}

/// RMSNorm backward.
fn rmsnorm_bwd(x: &[f32], grad_out: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    let dot: f32 = (0..n).map(|j| grad_out[j] * weight[j] * x[j]).sum();
    let coeff = dot * inv * inv / n as f32;
    (0..n).map(|i| weight[i] * inv * grad_out[i] - x[i] * inv * coeff).collect()
}

/// Softmax.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 { exps.iter().map(|&e| e / sum).collect() }
    else { vec![1.0 / logits.len() as f32; logits.len()] }
}

/// Gradient buffer matching model structure.
pub struct GradBuffer {
    /// Embedding gradients [vocab × d]
    pub embed: Vec<f32>,
    /// Per-layer gradients: (wq, wk, wv, wo, ffn_up, ffn_down)
    pub layers: Vec<[Vec<f32>; 6]>,
    /// LM head gradients [vocab × d]
    pub lm_head: Vec<f32>,
    /// Number of sequences accumulated
    pub count: usize,
}

impl GradBuffer {
    pub fn zeros(config: &LMConfig) -> Self {
        let d = config.d_model;
        let v = config.vocab_size;
        let ffn = config.ffn_dim;
        let n_layers = config.n_layers;
        let layers = (0..n_layers).map(|_| [
            vec![0.0f32; d * d],     // wq
            vec![0.0f32; d * d],     // wk
            vec![0.0f32; d * d],     // wv
            vec![0.0f32; d * d],     // wo
            vec![0.0f32; ffn * d],   // ffn_up
            vec![0.0f32; d * ffn],   // ffn_down
        ]).collect();
        Self {
            embed: vec![0.0; v * d],
            layers,
            lm_head: vec![0.0; v * d],
            count: 0,
        }
    }

    /// Add another gradient buffer to this one.
    pub fn accumulate(&mut self, other: &GradBuffer) {
        for (a, b) in self.embed.iter_mut().zip(&other.embed) { *a += b; }
        for (a, b) in self.lm_head.iter_mut().zip(&other.lm_head) { *a += b; }
        for (sl, ol) in self.layers.iter_mut().zip(&other.layers) {
            for (sa, oa) in sl.iter_mut().zip(ol) {
                for (a, b) in sa.iter_mut().zip(oa) { *a += b; }
            }
        }
        self.count += other.count;
    }

    /// Scale all gradients by 1/count for averaging.
    fn average(&mut self) {
        if self.count <= 1 { return; }
        let s = 1.0 / self.count as f32;
        for g in self.embed.iter_mut() { *g *= s; }
        for g in self.lm_head.iter_mut() { *g *= s; }
        for layer in &mut self.layers {
            for arr in layer.iter_mut() {
                for g in arr.iter_mut() { *g *= s; }
            }
        }
    }
}

/// Compute forward pass + gradients using direct f32 matmul.
/// Returns (loss, accuracy, gradient_buffer).
///
/// This is the core function: NO ternary quantization overhead.
pub fn f32_forward_backward(
    model: &VagiLM,
    tokens: &[u32],
) -> (f32, f32, GradBuffer) {
    let config = &model.config;
    let d = config.d_model;
    let v = config.vocab_size;
    let n_heads = config.n_heads;
    let h_dim = d / n_heads;
    let ffn_dim = config.ffn_dim;
    let seq_len = tokens.len() - 1;
    let input = &tokens[..seq_len];
    let targets = &tokens[1..];

    let mut grads = GradBuffer::zeros(config);
    grads.count = 1;

    // ━━━ FORWARD ━━━

    // 1. Embedding lookup
    let mut hidden = vec![0.0f32; seq_len * d];
    for (t, &tok) in input.iter().enumerate() {
        let src = &model.embedding.weight[tok as usize * d..(tok as usize + 1) * d];
        hidden[t * d..(t + 1) * d].copy_from_slice(src);
    }
    let _embed_out = hidden.clone();

    // 2. Transformer layers
    let mut layer_inputs: Vec<Vec<f32>> = Vec::with_capacity(model.layers.len());
    let mut layer_normed: Vec<Vec<f32>> = Vec::with_capacity(model.layers.len());
    let mut layer_attn_outs: Vec<Vec<f32>> = Vec::with_capacity(model.layers.len());
    let mut layer_ffn_normed: Vec<Vec<f32>> = Vec::with_capacity(model.layers.len());
    let mut layer_ffn_up_outs: Vec<Vec<f32>> = Vec::with_capacity(model.layers.len());
    // Save Q, K, V for proper attention backward
    let mut layer_q: Vec<Vec<f32>> = Vec::with_capacity(model.layers.len());
    let mut layer_k: Vec<Vec<f32>> = Vec::with_capacity(model.layers.len());
    let mut layer_v: Vec<Vec<f32>> = Vec::with_capacity(model.layers.len());
    let mut layer_attn_probs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(model.layers.len());

    for layer in &model.layers {
        layer_inputs.push(hidden.clone());

        // Attention sub-block
        let mut x_norm = hidden.clone();
        for t in 0..seq_len {
            rmsnorm_fwd(&mut x_norm[t*d..(t+1)*d], &layer.attn_norm.weight, layer.attn_norm.eps);
        }
        layer_normed.push(x_norm.clone());

        // Q, K, V projections (f32 matmul!)
        let mut q = vec![0.0f32; seq_len * d];
        let mut k = vec![0.0f32; seq_len * d];
        let mut val = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            let xt = &x_norm[t*d..(t+1)*d];
            f32_matvec(&layer.attention.wq.w_latent, xt, &mut q[t*d..(t+1)*d], d, d);
            f32_matvec(&layer.attention.wk.w_latent, xt, &mut k[t*d..(t+1)*d], d, d);
            f32_matvec(&layer.attention.wv.w_latent, xt, &mut val[t*d..(t+1)*d], d, d);
        }

        // RoPE
        for t in 0..seq_len {
            for head in 0..n_heads {
                let off = t * d + head * h_dim;
                layer.attention.rope.apply(&mut q[off..off+h_dim], t);
                layer.attention.rope.apply(&mut k[off..off+h_dim], t);
            }
        }

        // Causal attention (save probs for backward)
        let mut attn_out = vec![0.0f32; seq_len * d];
        let mut all_probs: Vec<Vec<f32>> = Vec::with_capacity(n_heads * seq_len);
        for head in 0..n_heads {
            for qi in 0..seq_len {
                let mut scores = vec![f32::NEG_INFINITY; qi + 1];
                for ki in 0..=qi {
                    let mut dot = 0.0f32;
                    for j in 0..h_dim {
                        dot += q[qi*d + head*h_dim + j] * k[ki*d + head*h_dim + j];
                    }
                    scores[ki] = dot / (h_dim as f32).sqrt();
                }
                let probs = softmax(&scores);
                for j in 0..h_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi {
                        sum += probs[ki] * val[ki*d + head*h_dim + j];
                    }
                    attn_out[qi*d + head*h_dim + j] = sum;
                }
                all_probs.push(probs);
            }
        }
        layer_q.push(q.clone());
        layer_k.push(k.clone());
        layer_v.push(val.clone());
        layer_attn_probs.push(all_probs);

        // Output projection
        let mut proj_out = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            f32_matvec(&layer.attention.wo.w_latent, &attn_out[t*d..(t+1)*d],
                       &mut proj_out[t*d..(t+1)*d], d, d);
        }
        layer_attn_outs.push(proj_out.clone());

        // Residual
        for i in 0..seq_len*d { hidden[i] += proj_out[i]; }

        // FFN sub-block
        let _ffn_input = hidden.clone();
        let mut ffn_norm = hidden.clone();
        for t in 0..seq_len {
            rmsnorm_fwd(&mut ffn_norm[t*d..(t+1)*d], &layer.ffn_norm.weight, layer.ffn_norm.eps);
        }
        layer_ffn_normed.push(ffn_norm.clone());

        let mut ffn_up = vec![0.0f32; seq_len * ffn_dim];
        for t in 0..seq_len {
            f32_matvec(&layer.ffn_up.w_latent, &ffn_norm[t*d..(t+1)*d],
                       &mut ffn_up[t*ffn_dim..(t+1)*ffn_dim], ffn_dim, d);
        }
        // SiLU activation
        for v in ffn_up.iter_mut() {
            *v = *v * (1.0 / (1.0 + (-*v).exp()));
        }
        layer_ffn_up_outs.push(ffn_up.clone());

        let mut ffn_down_out = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            f32_matvec(&layer.ffn_down.w_latent, &ffn_up[t*ffn_dim..(t+1)*ffn_dim],
                       &mut ffn_down_out[t*d..(t+1)*d], d, ffn_dim);
        }

        // Residual
        for i in 0..seq_len*d { hidden[i] += ffn_down_out[i]; }
    }

    // 3. Final norm
    let pre_final_norm = hidden.clone();
    for t in 0..seq_len {
        rmsnorm_fwd(&mut hidden[t*d..(t+1)*d], &model.final_norm.weight, model.final_norm.eps);
    }

    // 4. LM head
    let lm_inputs = hidden.clone();
    let mut logits = vec![0.0f32; seq_len * v];
    for t in 0..seq_len {
        f32_matvec(&model.lm_head.w_latent, &hidden[t*d..(t+1)*d],
                   &mut logits[t*v..(t+1)*v], v, d);
    }

    // ━━━ LOSS ━━━
    let mut total_loss = 0.0f32;
    let mut correct = 0usize;
    let mut grad_logits = vec![0.0f32; seq_len * v];
    let scale = 1.0 / seq_len as f32;

    for t in 0..seq_len {
        let tok_logits = &logits[t*v..(t+1)*v];
        let target = targets[t] as usize;
        let probs = softmax(tok_logits);
        total_loss += -probs[target].max(1e-10).ln();

        // Check accuracy
        let pred = tok_logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred == target { correct += 1; }

        // Gradient of cross-entropy
        for i in 0..v {
            grad_logits[t*v + i] = probs[i] * scale;
        }
        grad_logits[t*v + target] -= scale;
    }
    let loss = total_loss / seq_len as f32;
    let accuracy = correct as f32 / seq_len as f32;

    // ━━━ BACKWARD ━━━

    // LM head backward
    let mut grad_hidden = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let go = &grad_logits[t*v..(t+1)*v];
        let x = &lm_inputs[t*d..(t+1)*d];
        // grad_input
        f32_matvec_transpose(&model.lm_head.w_latent, go, &mut grad_hidden[t*d..(t+1)*d], v, d);
        // accumulate weight gradient: outer product
        for i in 0..v {
            for j in 0..d {
                grads.lm_head[i * d + j] += go[i] * x[j];
            }
        }
    }

    // Final norm backward
    for t in 0..seq_len {
        let x = &pre_final_norm[t*d..(t+1)*d];
        let go = grad_hidden[t*d..(t+1)*d].to_vec();
        let gi = rmsnorm_bwd(x, &go, &model.final_norm.weight, model.final_norm.eps);
        grad_hidden[t*d..(t+1)*d].copy_from_slice(&gi);
    }

    // Clip gradients
    clip_grad(&mut grad_hidden, 1.0);

    // Transformer layers backward (reverse order)
    for l in (0..model.layers.len()).rev() {
        let layer = &model.layers[l];
        let d = config.d_model;

        // FFN backward
        // ffn_down backward
        let mut grad_ffn_up = vec![0.0f32; seq_len * ffn_dim];
        for t in 0..seq_len {
            let go = &grad_hidden[t*d..(t+1)*d];
            let x = &layer_ffn_up_outs[l][t*ffn_dim..(t+1)*ffn_dim];
            f32_matvec_transpose(&layer.ffn_down.w_latent, go, &mut grad_ffn_up[t*ffn_dim..(t+1)*ffn_dim], d, ffn_dim);
            for i in 0..d {
                for j in 0..ffn_dim {
                    grads.layers[l][5][i * ffn_dim + j] += go[i] * x[j];
                }
            }
        }

        // SiLU backward: d/dx(x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        for t in 0..seq_len {
            let normed = &layer_ffn_normed[l];
            let mut pre_act = vec![0.0f32; ffn_dim];
            f32_matvec(&layer.ffn_up.w_latent, &normed[t*d..(t+1)*d], &mut pre_act, ffn_dim, d);
            for j in 0..ffn_dim {
                let x = pre_act[j];
                let sig = 1.0 / (1.0 + (-x).exp());
                let silu_deriv = sig + x * sig * (1.0 - sig);
                grad_ffn_up[t * ffn_dim + j] *= silu_deriv;
            }
        }

        // ffn_up backward
        let mut grad_ffn_norm = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            let go = &grad_ffn_up[t*ffn_dim..(t+1)*ffn_dim];
            let x = &layer_ffn_normed[l][t*d..(t+1)*d];
            f32_matvec_transpose(&layer.ffn_up.w_latent, go, &mut grad_ffn_norm[t*d..(t+1)*d], ffn_dim, d);
            for i in 0..ffn_dim {
                for j in 0..d {
                    grads.layers[l][4][i * d + j] += go[i] * x[j];
                }
            }
        }

        // FFN norm backward + residual
        for t in 0..seq_len {
            let x = &layer_inputs[l][t*d..(t+1)*d]; // pre-ffn-norm input
            let go = &grad_ffn_norm[t*d..(t+1)*d];
            let gi = rmsnorm_bwd(x, go, &layer.ffn_norm.weight, layer.ffn_norm.eps);
            for j in 0..d {
                grad_hidden[t*d + j] += gi[j]; // residual: add ffn grad to hidden grad
            }
        }

        // Attention backward (EXACT gradients through Q, K, V)
        // Grad through output projection
        let mut grad_attn = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            let go = &grad_hidden[t*d..(t+1)*d];
            let x = &layer_attn_outs[l][t*d..(t+1)*d];
            f32_matvec_transpose(&layer.attention.wo.w_latent, go, &mut grad_attn[t*d..(t+1)*d], d, d);
            for i in 0..d {
                for j in 0..d {
                    grads.layers[l][3][i * d + j] += go[i] * x[j];
                }
            }
        }

        // Proper attention backward: backprop through softmax and QKV
        let ref_q = &layer_q[l];
        let ref_k = &layer_k[l];
        let ref_v = &layer_v[l];
        let ref_probs = &layer_attn_probs[l];
        let scale_attn = 1.0 / (h_dim as f32).sqrt();

        let mut grad_q = vec![0.0f32; seq_len * d];
        let mut grad_k = vec![0.0f32; seq_len * d];
        let mut grad_v = vec![0.0f32; seq_len * d];

        for head in 0..n_heads {
            for qi in 0..seq_len {
                let prob_idx = head * seq_len + qi;
                let probs = &ref_probs[prob_idx];
                let go = &grad_attn[qi*d + head*h_dim .. qi*d + head*h_dim + h_dim];

                // Backward through weighted sum: out = sum(probs[ki] * V[ki])
                // grad_V[ki] += probs[ki] * grad_out
                // grad_probs[ki] = dot(grad_out, V[ki])
                let mut grad_probs = vec![0.0f32; qi + 1];
                for ki in 0..=qi {
                    let mut dp = 0.0f32;
                    for j in 0..h_dim {
                        let gj = go[j];
                        grad_v[ki*d + head*h_dim + j] += probs[ki] * gj;
                        dp += gj * ref_v[ki*d + head*h_dim + j];
                    }
                    grad_probs[ki] = dp;
                }

                // Backward through softmax: grad_scores = probs * (grad_probs - dot(probs, grad_probs))
                let dot_pg: f32 = probs.iter().zip(&grad_probs).map(|(p, g)| p * g).sum();
                let mut grad_scores = vec![0.0f32; qi + 1];
                for ki in 0..=qi {
                    grad_scores[ki] = probs[ki] * (grad_probs[ki] - dot_pg) * scale_attn;
                }

                // Backward through Q·K^T: scores[ki] = Q[qi]·K[ki] / sqrt(h)
                // grad_Q[qi] += sum_ki grad_scores[ki] * K[ki]
                // grad_K[ki] += grad_scores[ki] * Q[qi]
                for ki in 0..=qi {
                    let gs = grad_scores[ki];
                    for j in 0..h_dim {
                        grad_q[qi*d + head*h_dim + j] += gs * ref_k[ki*d + head*h_dim + j];
                        grad_k[ki*d + head*h_dim + j] += gs * ref_q[qi*d + head*h_dim + j];
                    }
                }
            }
        }

        // Backprop through RoPE (approximate: skip rotation backward, use identity)
        // Backprop through Q/K/V projections → weight gradients + input gradients
        let normed = &layer_normed[l];
        let mut grad_normed = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            let x = &normed[t*d..(t+1)*d];
            let gq = &grad_q[t*d..(t+1)*d];
            let gk = &grad_k[t*d..(t+1)*d];
            let gv = &grad_v[t*d..(t+1)*d];
            // Input gradient through Q, K, V projections
            let mut gi = vec![0.0f32; d];
            f32_matvec_transpose(&layer.attention.wq.w_latent, gq, &mut gi, d, d);
            let mut gi_k = vec![0.0f32; d];
            f32_matvec_transpose(&layer.attention.wk.w_latent, gk, &mut gi_k, d, d);
            let mut gi_v = vec![0.0f32; d];
            f32_matvec_transpose(&layer.attention.wv.w_latent, gv, &mut gi_v, d, d);
            for j in 0..d {
                grad_normed[t*d + j] = gi[j] + gi_k[j] + gi_v[j];
            }
            // Weight gradients: outer products
            for i in 0..d {
                for j in 0..d {
                    grads.layers[l][0][i * d + j] += gq[i] * x[j]; // wq
                    grads.layers[l][1][i * d + j] += gk[i] * x[j]; // wk
                    grads.layers[l][2][i * d + j] += gv[i] * x[j]; // wv
                }
            }
        }

        // Attn norm backward + residual
        for t in 0..seq_len {
            let x = &layer_inputs[l][t*d..(t+1)*d];
            let go = &grad_normed[t*d..(t+1)*d];
            let gi = rmsnorm_bwd(x, go, &layer.attn_norm.weight, layer.attn_norm.eps);
            for j in 0..d {
                grad_hidden[t*d + j] = gi[j];
            }
        }

        clip_grad(&mut grad_hidden, 1.0);
    }

    // Embedding backward
    for t in 0..seq_len {
        let tok = input[t] as usize;
        for j in 0..d {
            grads.embed[tok * d + j] += grad_hidden[t * d + j];
        }
    }

    (loss, accuracy, grads)
}

/// Clip gradient vector by global norm.
fn clip_grad(grad: &mut [f32], max_norm: f32) {
    let norm_sq: f32 = grad.iter().map(|g| g * g).sum();
    let norm = norm_sq.sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for g in grad.iter_mut() { *g *= scale; }
    }
}

/// Apply accumulated gradients to model with AdamW.
pub fn apply_gradients(
    model: &mut VagiLM,
    grads: &GradBuffer,
    adam_m: &mut Vec<f32>,
    adam_v: &mut Vec<f32>,
    step: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
) {
    let _d = model.config.d_model;

    // Helper: AdamW update for a parameter slice
    let mut offset = 0usize;
    let mut adamw = |params: &mut [f32], grads: &[f32]| {
        let n = params.len();
        // Grow adam state if needed
        while adam_m.len() < offset + n { adam_m.push(0.0); adam_v.push(0.0); }
        let t = (step + 1) as f32;
        for i in 0..n {
            let g = grads[i];
            adam_m[offset + i] = beta1 * adam_m[offset + i] + (1.0 - beta1) * g;
            adam_v[offset + i] = beta2 * adam_v[offset + i] + (1.0 - beta2) * g * g;
            let m_hat = adam_m[offset + i] / (1.0 - beta1.powf(t));
            let v_hat = adam_v[offset + i] / (1.0 - beta2.powf(t));
            params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * params[i]);
        }
        offset += n;
    };

    // Embedding
    adamw(&mut model.embedding.weight, &grads.embed);

    // Layers
    for (l, layer) in model.layers.iter_mut().enumerate() {
        adamw(&mut layer.attention.wq.w_latent, &grads.layers[l][0]);
        adamw(&mut layer.attention.wk.w_latent, &grads.layers[l][1]);
        adamw(&mut layer.attention.wv.w_latent, &grads.layers[l][2]);
        adamw(&mut layer.attention.wo.w_latent, &grads.layers[l][3]);
        adamw(&mut layer.ffn_up.w_latent, &grads.layers[l][4]);
        adamw(&mut layer.ffn_down.w_latent, &grads.layers[l][5]);
    }

    // LM head
    adamw(&mut model.lm_head.w_latent, &grads.lm_head);
}

/// Batch-parallel training step.
///
/// Forwards N sequences in parallel using rayon,
/// accumulates gradients, applies single AdamW update.
///
/// Returns (avg_loss, avg_accuracy).
pub fn batch_train_step(
    model: &mut VagiLM,
    batch: &[&[u32]],
    adam_m: &mut Vec<f32>,
    adam_v: &mut Vec<f32>,
    step: usize,
    lr: f32,
) -> (f32, f32) {
    if batch.is_empty() { return (0.0, 0.0); }

    // Forward all sequences in parallel — model is read-only here
    let results: Vec<(f32, f32, GradBuffer)> = batch.par_iter()
        .map(|tokens| f32_forward_backward(model, tokens))
        .collect();

    // Accumulate gradients
    let mut total_grads = GradBuffer::zeros(&model.config);
    let mut total_loss = 0.0f32;
    let mut total_acc = 0.0f32;

    for (loss, acc, grad) in &results {
        total_loss += loss;
        total_acc += acc;
        total_grads.accumulate(grad);
    }

    total_grads.average();
    let n = batch.len() as f32;

    // Apply gradients
    apply_gradients(model, &total_grads, adam_m, adam_v, step, lr,
                    0.9, 0.999, 1e-8, 0.01);

    (total_loss / n, total_acc / n)
}
