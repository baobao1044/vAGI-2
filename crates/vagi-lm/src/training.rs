//! Language model training — cross-entropy loss + backpropagation.
//!
//! Implements teacher-forcing training for `VagiLM`:
//! - Forward pass with intermediate caching
//! - Cross-entropy loss over next-token prediction
//! - Backpropagation through all layers (LM head, transformer, embedding)
//! - Weight updates via STE (straight-through estimator)
//!
//! Key design decisions:
//! - Uses latent f32 weights (not quantized ternary) for gradient propagation
//!   to avoid gradient vanishing through sparse ternary matrices
//! - Proper RMSNorm backward with Jacobian computation
//! - Gradient clipping to prevent exploding gradients in deep networks

use crate::model::VagiLM;

// ── Cache structures ─────────────────────────────────────────────

/// Cached intermediates from one transformer layer's forward pass.
struct LayerCache {
    /// Input to this layer [seq_len * d]
    x_normed: Vec<f32>,
    /// Q projections after RoPE [seq_len * d]
    q: Vec<f32>,
    /// K projections after RoPE [seq_len * d]
    k: Vec<f32>,
    /// V projections [seq_len * d]
    v: Vec<f32>,
    /// Attention weights [n_heads * seq_len * seq_len] (causal)
    attn_weights: Vec<f32>,
    /// Attention output before O projection [seq_len * d]
    attn_pre_out: Vec<f32>,
    /// After attention residual h = input + attn_out [seq_len * d]
    h: Vec<f32>,
    /// h after ffn_norm [seq_len * d]
    h_normed: Vec<f32>,
    /// Pre-norm h (before ffn_norm) for RMSNorm backward
    h_pre_norm: Vec<f32>,
    /// Pre-norm x (before attn_norm) for RMSNorm backward
    x_pre_norm: Vec<f32>,
    /// FFN up output before activation [seq_len][ffn_dim]
    ffn_up_out: Vec<Vec<f32>>,
    /// FFN activated output [seq_len][ffn_dim]
    ffn_activated: Vec<Vec<f32>>,
}

/// Training hyperparameters.
pub struct TrainConfig {
    /// Learning rate.
    pub lr: f32,
    /// Gradient clipping max norm.
    pub grad_clip: f32,
    /// Weight decay (L2 regularization).
    pub weight_decay: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            grad_clip: 1.0,
            weight_decay: 0.0,
        }
    }
}

impl TrainConfig {
    /// Tuned config for tiny model.
    pub fn for_tiny() -> Self {
        Self {
            lr: 0.01,
            grad_clip: 1.0,
            weight_decay: 0.0,
        }
    }
}

// ── Public API ───────────────────────────────────────────────────

/// Compute cross-entropy loss without updating weights.
pub fn compute_loss(model: &VagiLM, tokens: &[u32]) -> f32 {
    if tokens.len() < 2 { return 0.0; }
    let seq_len = tokens.len() - 1;
    let v = model.config.vocab_size;

    let logits = model.forward(&tokens[..seq_len]);
    let targets = &tokens[1..];

    let mut total_loss = 0.0f32;
    for t in 0..seq_len {
        let tok_logits = &logits[t * v..(t + 1) * v];
        let target = targets[t] as usize;
        let probs = softmax(tok_logits);
        total_loss += -probs[target].max(1e-10).ln();
    }
    total_loss / seq_len as f32
}

/// One training step: forward → loss → backward → update weights.
///
/// Returns the cross-entropy loss for this step.
pub fn train_step(model: &mut VagiLM, tokens: &[u32], lr: f32) -> f32 {
    let cfg = TrainConfig { lr, ..TrainConfig::default() };
    train_step_cfg(model, tokens, &cfg)
}

/// Training step with full configuration.
pub fn train_step_cfg(model: &mut VagiLM, tokens: &[u32], cfg: &TrainConfig) -> f32 {
    if tokens.len() < 2 { return 0.0; }
    let seq_len = tokens.len() - 1;
    let d = model.config.d_model;
    let v = model.config.vocab_size;
    let input_tokens = &tokens[..seq_len];
    let targets = &tokens[1..];

    // ═══════════════════════════════════════════
    //  FORWARD PASS WITH CACHING
    // ═══════════════════════════════════════════

    // 1. Embedding
    let hidden_0 = model.embedding.forward(input_tokens);

    // 2. Transformer layers
    let mut layer_caches: Vec<LayerCache> = Vec::with_capacity(model.layers.len());
    let mut hidden = hidden_0.clone();

    for layer in model.layers.iter() {
        let (output, cache) = forward_layer_cached(layer, &hidden, seq_len);
        layer_caches.push(cache);
        hidden = output;
    }

    // 3. Final RMSNorm (per token)
    let pre_norm = hidden.clone();
    for t in 0..seq_len {
        model.final_norm.forward(&mut hidden[t * d..(t + 1) * d]);
    }

    // 4. LM head
    let lm_inputs: Vec<Vec<f32>> = (0..seq_len)
        .map(|t| hidden[t * d..(t + 1) * d].to_vec())
        .collect();
    let mut logits = vec![0.0f32; seq_len * v];
    for t in 0..seq_len {
        model.lm_head.forward(&lm_inputs[t], &mut logits[t * v..(t + 1) * v]);
    }

    // ═══════════════════════════════════════════
    //  COMPUTE LOSS + GRADIENT W.R.T. LOGITS
    // ═══════════════════════════════════════════

    let mut total_loss = 0.0f32;
    let mut grad_logits = vec![0.0f32; seq_len * v];

    for t in 0..seq_len {
        let tok_logits = &logits[t * v..(t + 1) * v];
        let target = targets[t] as usize;
        let probs = softmax(tok_logits);
        total_loss += -probs[target].max(1e-10).ln();

        // ∂CE/∂logits = (probs - one_hot(target)) / seq_len
        let scale = 1.0 / seq_len as f32;
        for i in 0..v {
            grad_logits[t * v + i] = probs[i] * scale;
        }
        grad_logits[t * v + target] -= scale;
    }
    total_loss /= seq_len as f32;

    // ═══════════════════════════════════════════
    //  BACKWARD PASS
    // ═══════════════════════════════════════════

    // 1. LM head backward
    let mut grad_hidden = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let grad_out = &grad_logits[t * v..(t + 1) * v];
        let grad_in = ste_backward_with_grad(
            &mut model.lm_head, grad_out, &lm_inputs[t], cfg.lr,
        );
        grad_hidden[t * d..(t + 1) * d].copy_from_slice(&grad_in);
    }

    // 2. Final RMSNorm backward (proper)
    for t in 0..seq_len {
        let x = &pre_norm[t * d..(t + 1) * d];
        let grad_out = grad_hidden[t * d..(t + 1) * d].to_vec();
        let grad_in = rmsnorm_backward(
            x, &grad_out, &model.final_norm.weight, model.final_norm.eps,
        );
        grad_hidden[t * d..(t + 1) * d].copy_from_slice(&grad_in);
    }

    // Gradient clipping
    clip_grad(&mut grad_hidden, cfg.grad_clip);

    // 3. Transformer layers backward (reverse)
    for l in (0..model.layers.len()).rev() {
        grad_hidden = backward_layer(
            &mut model.layers[l],
            &grad_hidden,
            &layer_caches[l],
            seq_len,
            cfg,
        );
        clip_grad(&mut grad_hidden, cfg.grad_clip);
    }

    // 4. Embedding backward: update embedding vectors
    for t in 0..seq_len {
        let token = input_tokens[t] as usize;
        let start = token * d;
        for j in 0..d {
            model.embedding.weight[start + j] -= cfg.lr * grad_hidden[t * d + j];
        }
    }

    total_loss
}

/// Train over token sequences for one epoch. Returns average loss.
pub fn train_epoch(model: &mut VagiLM, sequences: &[Vec<u32>], lr: f32) -> f32 {
    let cfg = TrainConfig { lr, ..TrainConfig::default() };
    train_epoch_cfg(model, sequences, &cfg)
}

/// Train with full configuration. Returns average loss.
pub fn train_epoch_cfg(
    model: &mut VagiLM,
    sequences: &[Vec<u32>],
    cfg: &TrainConfig,
) -> f32 {
    if sequences.is_empty() { return 0.0; }
    let mut total_loss = 0.0f32;
    let mut count = 0;
    for seq in sequences {
        if seq.len() >= 2 {
            total_loss += train_step_cfg(model, seq, cfg);
            count += 1;
        }
    }
    if count > 0 { total_loss / count as f32 } else { 0.0 }
}

// ── Internal helpers ─────────────────────────────────────────────

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

/// Clip gradient vector by global norm.
fn clip_grad(grad: &mut [f32], max_norm: f32) {
    let norm_sq: f32 = grad.iter().map(|g| g * g).sum();
    let norm = norm_sq.sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}

/// RMSNorm backward.
///
/// Given y = x * scale / rms, computes ∂L/∂x from ∂L/∂y.
/// Formula: ∂L/∂x_i = scale_i/rms * (∂L/∂y_i - x_i/(n*rms²) * Σ_j ∂L/∂y_j * scale_j * x_j)
fn rmsnorm_backward(
    x: &[f32],
    grad_output: &[f32],
    scale: &[f32],
    eps: f32,
) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_sq = inv_rms * inv_rms;

    // Compute dot = Σ_j (∂L/∂y_j * scale_j * x_j)
    let dot: f32 = (0..n).map(|j| {
        grad_output[j] * scale[j] * x[j]
    }).sum();

    let mut grad_input = vec![0.0f32; n];
    let coeff = dot * inv_rms_sq / n as f32;
    for i in 0..n {
        grad_input[i] = scale[i] * inv_rms * grad_output[i] - x[i] * inv_rms * coeff;
    }
    grad_input
}

/// STELinear backward: update weights AND return gradient w.r.t. input.
///
/// Uses latent f32 weights (not quantized ternary) for gradient propagation
/// to avoid gradient vanishing through sparse ternary matrices.
fn ste_backward_with_grad(
    layer: &mut vagi_core::ste::STELinear,
    grad_output: &[f32],
    x: &[f32],
    lr: f32,
) -> Vec<f32> {
    // Compute grad_input using LATENT f32 weights (not quantized ternary).
    // This is critical: ternary weights have ~50% zeros which kills gradient flow.
    // The STE principle says we use quantized weights for forward, but latent for backward.
    let mut grad_input = vec![0.0f32; layer.in_features];
    for j in 0..layer.in_features {
        let mut sum = 0.0f32;
        for i in 0..layer.out_features {
            sum += layer.w_latent[i * layer.in_features + j] * grad_output[i];
        }
        grad_input[j] = sum;
    }

    // Update latent weights via STE
    layer.backward_update(grad_output, x, lr);

    grad_input
}

/// Forward one transformer layer with caching for backward pass.
fn forward_layer_cached(
    layer: &crate::transformer::TransformerLayer,
    x: &[f32],
    seq_len: usize,
) -> (Vec<f32>, LayerCache) {
    let d = layer.d_model;
    let ffn_dim = layer.ffn_dim;
    let attn = &layer.attention;
    let h_dim = attn.head_dim;
    let n_heads = attn.n_heads;

    // Save pre-norm input for RMSNorm backward
    let x_pre_norm = x.to_vec();

    // ── Attention block ──
    let mut x_normed = x.to_vec();
    for t in 0..seq_len {
        layer.attn_norm.forward(&mut x_normed[t * d..(t + 1) * d]);
    }

    // Q, K, V projections
    let mut q_all = vec![0.0f32; seq_len * d];
    let mut k_all = vec![0.0f32; seq_len * d];
    let mut v_all = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let x_t = &x_normed[t * d..(t + 1) * d];
        attn.wq.forward(x_t, &mut q_all[t * d..(t + 1) * d]);
        attn.wk.forward(x_t, &mut k_all[t * d..(t + 1) * d]);
        attn.wv.forward(x_t, &mut v_all[t * d..(t + 1) * d]);
    }

    // Apply RoPE
    for t in 0..seq_len {
        for head in 0..n_heads {
            let offset = t * d + head * h_dim;
            attn.rope.apply(&mut q_all[offset..offset + h_dim], t);
            attn.rope.apply(&mut k_all[offset..offset + h_dim], t);
        }
    }

    // Attention computation + cache weights
    let mut attn_pre_out = vec![0.0f32; seq_len * d];
    let mut attn_weights = vec![0.0f32; n_heads * seq_len * seq_len];

    for head in 0..n_heads {
        for qi in 0..seq_len {
            let q_offset = qi * d + head * h_dim;
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for ki in 0..=qi {
                let k_offset = ki * d + head * h_dim;
                let mut dot = 0.0f32;
                for j in 0..h_dim {
                    dot += q_all[q_offset + j] * k_all[k_offset + j];
                }
                scores[ki] = dot / (h_dim as f32).sqrt();
            }

            let max_s = scores[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0.0f32; qi + 1];
            let mut sum_exp = 0.0f32;
            for i in 0..=qi {
                exps[i] = (scores[i] - max_s).exp();
                sum_exp += exps[i];
            }
            if sum_exp > 0.0 {
                for e in exps.iter_mut() { *e /= sum_exp; }
            }

            let aw_base = head * seq_len * seq_len + qi * seq_len;
            for i in 0..=qi {
                attn_weights[aw_base + i] = exps[i];
            }

            let out_offset = qi * d + head * h_dim;
            for vi in 0..=qi {
                let v_offset = vi * d + head * h_dim;
                let w = exps[vi];
                for j in 0..h_dim {
                    attn_pre_out[out_offset + j] += w * v_all[v_offset + j];
                }
            }
        }
    }

    // O projection
    let mut attn_out = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        attn.wo.forward(
            &attn_pre_out[t * d..(t + 1) * d],
            &mut attn_out[t * d..(t + 1) * d],
        );
    }

    // Residual: h = x + attn_out
    let h: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

    // ── FFN block ──
    let h_pre_norm = h.clone();
    let mut h_normed = h.clone();
    for t in 0..seq_len {
        layer.ffn_norm.forward(&mut h_normed[t * d..(t + 1) * d]);
    }

    let mut ffn_up_out = Vec::with_capacity(seq_len);
    let mut ffn_activated = Vec::with_capacity(seq_len);
    let mut output = h.clone();

    for t in 0..seq_len {
        let tok = &h_normed[t * d..(t + 1) * d];
        let mut up = vec![0.0f32; ffn_dim];
        layer.ffn_up.forward(tok, &mut up);

        let pre_act = up.clone();
        layer.activation.forward(&mut up);
        let activated = up.clone();

        let mut down = vec![0.0f32; d];
        layer.ffn_down.forward(&activated, &mut down);

        for j in 0..d {
            output[t * d + j] += down[j];
        }

        ffn_up_out.push(pre_act);
        ffn_activated.push(activated);
    }

    let cache = LayerCache {
        x_normed,
        q: q_all,
        k: k_all,
        v: v_all,
        attn_weights,
        attn_pre_out,
        h,
        h_normed,
        h_pre_norm,
        x_pre_norm,
        ffn_up_out,
        ffn_activated,
    };

    (output, cache)
}

/// Backward through one transformer layer. Returns gradient w.r.t. layer input.
fn backward_layer(
    layer: &mut crate::transformer::TransformerLayer,
    grad_output: &[f32],
    cache: &LayerCache,
    seq_len: usize,
    cfg: &TrainConfig,
) -> Vec<f32> {
    let d = layer.d_model;
    let n_heads = layer.attention.n_heads;
    let h_dim = layer.attention.head_dim;
    let lr = cfg.lr;

    // ── FFN block backward ──
    // output = h + ffn_out → grad_h = grad_output (residual copies gradient)

    let mut grad_h = grad_output.to_vec();

    for t in 0..seq_len {
        let grad_down_out = &grad_output[t * d..(t + 1) * d];

        // FFN down backward
        let grad_activated = ste_backward_with_grad(
            &mut layer.ffn_down,
            grad_down_out,
            &cache.ffn_activated[t],
            lr,
        );

        // Activation backward
        let grad_up = layer.activation.backward_input(
            &cache.ffn_up_out[t],
            &grad_activated,
        );

        // FFN up backward
        let grad_h_normed_t = ste_backward_with_grad(
            &mut layer.ffn_up,
            &grad_up,
            &cache.h_normed[t * d..(t + 1) * d],
            lr,
        );

        // FFN norm backward (proper RMSNorm backward)
        let h_t = &cache.h_pre_norm[t * d..(t + 1) * d];
        let grad_h_t = rmsnorm_backward(
            h_t, &grad_h_normed_t,
            &layer.ffn_norm.weight, layer.ffn_norm.eps,
        );
        for j in 0..d {
            grad_h[t * d + j] += grad_h_t[j];
        }
    }

    // ── Attention block backward ──
    // h = input + attn_out → grad_input = grad_h (residual)

    let mut grad_input = grad_h.clone();

    // O projection backward (per token)
    let mut grad_attn_pre_out = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let grad_o = ste_backward_with_grad(
            &mut layer.attention.wo,
            &grad_h[t * d..(t + 1) * d],
            &cache.attn_pre_out[t * d..(t + 1) * d],
            lr,
        );
        grad_attn_pre_out[t * d..(t + 1) * d].copy_from_slice(&grad_o);
    }

    // Attention mechanism backward (per head)
    let mut grad_v = vec![0.0f32; seq_len * d];
    let mut grad_q = vec![0.0f32; seq_len * d];
    let mut grad_k = vec![0.0f32; seq_len * d];

    for head in 0..n_heads {
        for qi in 0..seq_len {
            let q_offset = qi * d + head * h_dim;
            let aw_base = head * seq_len * seq_len + qi * seq_len;

            // Grad for V
            for vi in 0..=qi {
                let v_offset = vi * d + head * h_dim;
                let w = cache.attn_weights[aw_base + vi];
                for j in 0..h_dim {
                    grad_v[v_offset + j] += w * grad_attn_pre_out[q_offset + j];
                }
            }

            // Grad for attention weights
            let mut grad_aw = vec![0.0f32; qi + 1];
            for vi in 0..=qi {
                let v_offset = vi * d + head * h_dim;
                let mut dot = 0.0f32;
                for j in 0..h_dim {
                    dot += grad_attn_pre_out[q_offset + j] * cache.v[v_offset + j];
                }
                grad_aw[vi] = dot;
            }

            // Softmax backward
            let dot_sum: f32 = (0..=qi).map(|i| {
                cache.attn_weights[aw_base + i] * grad_aw[i]
            }).sum();
            let mut grad_scores = vec![0.0f32; qi + 1];
            for i in 0..=qi {
                let p = cache.attn_weights[aw_base + i];
                grad_scores[i] = p * (grad_aw[i] - dot_sum);
            }

            let scale = 1.0 / (h_dim as f32).sqrt();

            // Grad for Q and K
            for ki in 0..=qi {
                let k_offset = ki * d + head * h_dim;
                for j in 0..h_dim {
                    grad_q[q_offset + j] += grad_scores[ki] * cache.k[k_offset + j] * scale;
                    grad_k[k_offset + j] += grad_scores[ki] * cache.q[q_offset + j] * scale;
                }
            }
        }
    }

    // RoPE backward: inverse rotation
    for t in 0..seq_len {
        for head in 0..n_heads {
            let offset = t * d + head * h_dim;
            rope_backward(&mut grad_q[offset..offset + h_dim], &layer.attention.rope, t);
            rope_backward(&mut grad_k[offset..offset + h_dim], &layer.attention.rope, t);
        }
    }

    // Q, K, V projection backward
    let mut grad_x_normed = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let x_t = &cache.x_normed[t * d..(t + 1) * d];

        let gq = ste_backward_with_grad(
            &mut layer.attention.wq,
            &grad_q[t * d..(t + 1) * d],
            x_t, lr,
        );
        let gk = ste_backward_with_grad(
            &mut layer.attention.wk,
            &grad_k[t * d..(t + 1) * d],
            x_t, lr,
        );
        let gv = ste_backward_with_grad(
            &mut layer.attention.wv,
            &grad_v[t * d..(t + 1) * d],
            x_t, lr,
        );

        for j in 0..d {
            grad_x_normed[t * d + j] = gq[j] + gk[j] + gv[j];
        }
    }

    // Attention norm backward (proper RMSNorm backward)
    for t in 0..seq_len {
        let x_t = &cache.x_pre_norm[t * d..(t + 1) * d];
        let grad_xn = &grad_x_normed[t * d..(t + 1) * d];
        let grad_x = rmsnorm_backward(
            x_t, grad_xn,
            &layer.attn_norm.weight, layer.attn_norm.eps,
        );
        for j in 0..d {
            grad_input[t * d + j] += grad_x[j];
        }
    }

    grad_input
}

/// RoPE backward: apply inverse rotation (transpose = cos, -sin).
fn rope_backward(
    grad: &mut [f32],
    rope: &crate::attention::RoPECache,
    pos: usize,
) {
    let half = grad.len() / 2;
    let base = pos * half;
    for i in 0..half {
        let g0 = grad[2 * i];
        let g1 = grad[2 * i + 1];
        let c = rope.cos[base + i];
        let s = rope.sin[base + i];
        grad[2 * i]     = g0 * c + g1 * s;
        grad[2 * i + 1] = -g0 * s + g1 * c;
    }
}

// ── LMTrainer: Advanced training with AdamW ─────────────────────

/// Advanced training configuration.
#[derive(Clone)]
pub struct AdvancedConfig {
    /// Peak learning rate.
    pub lr: f32,
    /// AdamW β1 (first moment decay).
    pub beta1: f32,
    /// AdamW β2 (second moment decay).
    pub beta2: f32,
    /// AdamW epsilon.
    pub eps: f32,
    /// Weight decay (L2 regularization).
    pub weight_decay: f32,
    /// Gradient clipping max norm.
    pub grad_clip: f32,
    /// Warmup steps (linear LR ramp from 0 to lr).
    pub warmup_steps: usize,
    /// Total training steps (for cosine decay).
    pub total_steps: usize,
    /// Label smoothing factor (0 = no smoothing, 0.1 typical).
    pub label_smoothing: f32,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            grad_clip: 1.0,
            warmup_steps: 10,
            total_steps: 200,
            label_smoothing: 0.1,
        }
    }
}

impl AdvancedConfig {
    /// Get effective learning rate at given step (warmup + cosine decay).
    pub fn lr_at_step(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.lr * (step + 1) as f32 / self.warmup_steps as f32
        } else if self.total_steps > self.warmup_steps {
            // Cosine decay to 10% of peak
            let decay_steps = self.total_steps - self.warmup_steps;
            let progress = (step - self.warmup_steps) as f32 / decay_steps as f32;
            let min_lr = self.lr * 0.1;
            min_lr + 0.5 * (self.lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        } else {
            self.lr
        }
    }
}

/// Training metrics for a single step.
#[derive(Clone, Debug)]
pub struct TrainMetrics {
    /// Cross-entropy loss.
    pub loss: f32,
    /// Perplexity (exp(loss)).
    pub perplexity: f32,
    /// Token prediction accuracy (top-1).
    pub accuracy: f32,
    /// Effective learning rate at this step.
    pub lr: f32,
}

/// Advanced language model trainer with AdamW optimizer.
///
/// Maintains per-parameter optimizer state (first/second moment estimates)
/// for all learnable weights in the model. Uses cosine LR scheduling with
/// warmup and label smoothing for better generalization.
pub struct LMTrainer {
    /// First moment estimates (m), flattened across all params.
    pub(crate) adam_m: Vec<f32>,
    /// Second moment estimates (v), flattened across all params.
    pub(crate) adam_v: Vec<f32>,
    /// Offsets into adam_m/adam_v for each parameter group.
    /// [embedding, layer0_wq, layer0_wk, ..., lm_head]
    param_offsets: Vec<usize>,
    /// Current step count.
    pub(crate) step: usize,
    /// Configuration.
    config: AdvancedConfig,
}

impl LMTrainer {
    /// Create a new trainer for the given model.
    pub fn new(model: &VagiLM, config: AdvancedConfig) -> Self {
        let mut offsets = Vec::new();
        let mut total_params = 0usize;

        // Embedding
        offsets.push(total_params);
        total_params += model.embedding.weight.len();

        // Each transformer layer: wq, wk, wv, wo, ffn_up, ffn_down
        for layer in &model.layers {
            offsets.push(total_params);
            total_params += layer.attention.wq.w_latent.len();
            offsets.push(total_params);
            total_params += layer.attention.wk.w_latent.len();
            offsets.push(total_params);
            total_params += layer.attention.wv.w_latent.len();
            offsets.push(total_params);
            total_params += layer.attention.wo.w_latent.len();
            offsets.push(total_params);
            total_params += layer.ffn_up.w_latent.len();
            offsets.push(total_params);
            total_params += layer.ffn_down.w_latent.len();
        }

        // LM head
        offsets.push(total_params);
        total_params += model.lm_head.w_latent.len();

        Self {
            adam_m: vec![0.0; total_params],
            adam_v: vec![0.0; total_params],
            param_offsets: offsets,
            step: 0,
            config,
        }
    }

    /// Get current step count.
    pub fn step_count(&self) -> usize { self.step }

    /// Get current effective learning rate.
    pub fn current_lr(&self) -> f32 { self.config.lr_at_step(self.step) }

    /// One training step with AdamW. Returns metrics.
    pub fn train_step(&mut self, model: &mut VagiLM, tokens: &[u32]) -> TrainMetrics {
        if tokens.len() < 2 {
            return TrainMetrics { loss: 0.0, perplexity: 1.0, accuracy: 0.0, lr: 0.0 };
        }

        let seq_len = tokens.len() - 1;
        let d = model.config.d_model;
        let v = model.config.vocab_size;
        let input_tokens = &tokens[..seq_len];
        let targets = &tokens[1..];
        let lr = self.config.lr_at_step(self.step);
        let label_smoothing = self.config.label_smoothing;

        // ═══ FORWARD PASS ═══
        let hidden_0 = model.embedding.forward(input_tokens);
        let mut layer_caches: Vec<LayerCache> = Vec::with_capacity(model.layers.len());
        let mut hidden = hidden_0.clone();

        for layer in model.layers.iter() {
            let (output, cache) = forward_layer_cached(layer, &hidden, seq_len);
            layer_caches.push(cache);
            hidden = output;
        }

        let pre_norm = hidden.clone();
        for t in 0..seq_len {
            model.final_norm.forward(&mut hidden[t * d..(t + 1) * d]);
        }

        let lm_inputs: Vec<Vec<f32>> = (0..seq_len)
            .map(|t| hidden[t * d..(t + 1) * d].to_vec())
            .collect();
        let mut logits = vec![0.0f32; seq_len * v];
        for t in 0..seq_len {
            model.lm_head.forward(&lm_inputs[t], &mut logits[t * v..(t + 1) * v]);
        }

        // ═══ LOSS WITH LABEL SMOOTHING + METRICS ═══
        let mut total_loss = 0.0f32;
        let mut grad_logits = vec![0.0f32; seq_len * v];
        let mut correct = 0usize;

        for t in 0..seq_len {
            let tok_logits = &logits[t * v..(t + 1) * v];
            let target = targets[t] as usize;
            let probs = softmax(tok_logits);

            // Accuracy: check if top-1 prediction matches target
            let predicted = probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            if predicted == target { correct += 1; }

            // Label-smoothed cross-entropy loss
            // smooth_target = (1-ε) * one_hot + ε/V
            let smooth = label_smoothing / v as f32;
            let on_value = 1.0 - label_smoothing + smooth;
            let mut token_loss = 0.0f32;
            for i in 0..v {
                let target_prob = if i == target { on_value } else { smooth };
                token_loss -= target_prob * probs[i].max(1e-10).ln();
            }
            total_loss += token_loss;

            // Gradient: ∂(smoothed CE)/∂logits = probs - smooth_target
            let scale = 1.0 / seq_len as f32;
            for i in 0..v {
                let target_prob = if i == target { on_value } else { smooth };
                grad_logits[t * v + i] = (probs[i] - target_prob) * scale;
            }
        }
        total_loss /= seq_len as f32;

        // ═══ BACKWARD PASS (compute grads, don't update yet) ═══
        let mut param_idx = self.param_offsets.len() - 1; // Start from lm_head (last)

        // LM head backward
        let mut grad_hidden = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            let grad_out = &grad_logits[t * v..(t + 1) * v];
            let grad_in = self.ste_backward_adamw(
                &mut model.lm_head, grad_out, &lm_inputs[t], lr, param_idx,
            );
            grad_hidden[t * d..(t + 1) * d].copy_from_slice(&grad_in);
        }

        // Final RMSNorm backward
        for t in 0..seq_len {
            let x = &pre_norm[t * d..(t + 1) * d];
            let grad_out = grad_hidden[t * d..(t + 1) * d].to_vec();
            let grad_in = rmsnorm_backward(
                x, &grad_out, &model.final_norm.weight, model.final_norm.eps,
            );
            grad_hidden[t * d..(t + 1) * d].copy_from_slice(&grad_in);
        }

        clip_grad(&mut grad_hidden, self.config.grad_clip);

        // Transformer layers backward
        for l in (0..model.layers.len()).rev() {
            // Each layer has 6 param groups: wq, wk, wv, wo, ffn_up, ffn_down
            // param_idx for this layer's ffn_down
            param_idx = 1 + l * 6; // skip embedding (index 0)
            grad_hidden = self.backward_layer_adamw(
                &mut model.layers[l],
                &grad_hidden,
                &layer_caches[l],
                seq_len,
                lr,
                param_idx,
            );
            clip_grad(&mut grad_hidden, self.config.grad_clip);
        }

        // Embedding backward with AdamW
        let emb_offset = self.param_offsets[0];
        for t in 0..seq_len {
            let token = input_tokens[t] as usize;
            let start = token * d;
            for j in 0..d {
                let idx = start + j;
                let grad = grad_hidden[t * d + j];
                self.adamw_update(
                    &mut model.embedding.weight[idx],
                    grad, emb_offset + idx, lr, 0.0, // no weight decay on embeddings
                );
            }
        }

        self.step += 1;

        TrainMetrics {
            loss: total_loss,
            perplexity: total_loss.exp(),
            accuracy: correct as f32 / seq_len as f32,
            lr,
        }
    }

    /// Train one epoch over all sequences. Returns average metrics.
    pub fn train_epoch(&mut self, model: &mut VagiLM, sequences: &[Vec<u32>]) -> TrainMetrics {
        let mut total = TrainMetrics { loss: 0.0, perplexity: 0.0, accuracy: 0.0, lr: 0.0 };
        let mut count = 0;
        for seq in sequences {
            if seq.len() >= 2 {
                let m = self.train_step(model, seq);
                total.loss += m.loss;
                total.perplexity += m.perplexity;
                total.accuracy += m.accuracy;
                total.lr = m.lr;
                count += 1;
            }
        }
        if count > 0 {
            total.loss /= count as f32;
            total.perplexity /= count as f32;
            total.accuracy /= count as f32;
        }
        total
    }

    /// Apply AdamW update to a single parameter.
    fn adamw_update(
        &mut self,
        param: &mut f32,
        grad: f32,
        state_idx: usize,
        lr: f32,
        weight_decay: f32,
    ) {
        let step = self.step + 1; // 1-indexed for bias correction
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;

        // Update moments
        self.adam_m[state_idx] = beta1 * self.adam_m[state_idx] + (1.0 - beta1) * grad;
        self.adam_v[state_idx] = beta2 * self.adam_v[state_idx] + (1.0 - beta2) * grad * grad;

        // Bias correction
        let m_hat = self.adam_m[state_idx] / (1.0 - beta1.powi(step as i32));
        let v_hat = self.adam_v[state_idx] / (1.0 - beta2.powi(step as i32));

        // AdamW update: decouple weight decay from gradient
        *param -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * *param);
    }

    /// STELinear backward with AdamW: compute grad_input + apply AdamW to weights.
    fn ste_backward_adamw(
        &mut self,
        layer: &mut vagi_core::ste::STELinear,
        grad_output: &[f32],
        x: &[f32],
        lr: f32,
        param_group_idx: usize,
    ) -> Vec<f32> {
        let in_f = layer.in_features;
        let out_f = layer.out_features;

        // Compute grad_input using latent f32 weights
        let mut grad_input = vec![0.0f32; in_f];
        for j in 0..in_f {
            let mut sum = 0.0f32;
            for i in 0..out_f {
                sum += layer.w_latent[i * in_f + j] * grad_output[i];
            }
            grad_input[j] = sum;
        }

        // Apply AdamW to each weight
        let offset = self.param_offsets[param_group_idx];
        let wd = self.config.weight_decay;
        for m in 0..out_f {
            for n in 0..in_f {
                let idx = m * in_f + n;
                // STE clip check
                if layer.w_latent[idx].abs() > layer.clip_range { continue; }
                let grad = grad_output[m] * x[n];
                self.adamw_update(&mut layer.w_latent[idx], grad, offset + idx, lr, wd);
            }
        }

        grad_input
    }

    /// Backward through transformer layer with AdamW updates.
    fn backward_layer_adamw(
        &mut self,
        layer: &mut crate::transformer::TransformerLayer,
        grad_output: &[f32],
        cache: &LayerCache,
        seq_len: usize,
        lr: f32,
        base_param_idx: usize, // index of wq in param_offsets
    ) -> Vec<f32> {
        let d = layer.d_model;
        let n_heads = layer.attention.n_heads;
        let h_dim = layer.attention.head_dim;

        // FFN backward
        let mut grad_h = grad_output.to_vec();

        for t in 0..seq_len {
            let grad_down_out = &grad_output[t * d..(t + 1) * d];
            let grad_activated = self.ste_backward_adamw(
                &mut layer.ffn_down, grad_down_out,
                &cache.ffn_activated[t], lr, base_param_idx + 5,
            );
            let grad_up = layer.activation.backward_input(
                &cache.ffn_up_out[t], &grad_activated,
            );
            let grad_h_normed_t = self.ste_backward_adamw(
                &mut layer.ffn_up, &grad_up,
                &cache.h_normed[t * d..(t + 1) * d], lr, base_param_idx + 4,
            );
            let h_t = &cache.h_pre_norm[t * d..(t + 1) * d];
            let grad_h_t = rmsnorm_backward(
                h_t, &grad_h_normed_t,
                &layer.ffn_norm.weight, layer.ffn_norm.eps,
            );
            for j in 0..d {
                grad_h[t * d + j] += grad_h_t[j];
            }
        }

        // Attention backward
        let mut grad_input = grad_h.clone();
        let mut grad_attn_pre_out = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            let grad_o = self.ste_backward_adamw(
                &mut layer.attention.wo,
                &grad_h[t * d..(t + 1) * d],
                &cache.attn_pre_out[t * d..(t + 1) * d],
                lr, base_param_idx + 3,
            );
            grad_attn_pre_out[t * d..(t + 1) * d].copy_from_slice(&grad_o);
        }

        let mut grad_v = vec![0.0f32; seq_len * d];
        let mut grad_q = vec![0.0f32; seq_len * d];
        let mut grad_k = vec![0.0f32; seq_len * d];

        for head in 0..n_heads {
            for qi in 0..seq_len {
                let q_offset = qi * d + head * h_dim;
                let aw_base = head * seq_len * seq_len + qi * seq_len;

                for vi in 0..=qi {
                    let v_offset = vi * d + head * h_dim;
                    let w = cache.attn_weights[aw_base + vi];
                    for j in 0..h_dim {
                        grad_v[v_offset + j] += w * grad_attn_pre_out[q_offset + j];
                    }
                }

                let mut grad_aw = vec![0.0f32; qi + 1];
                for vi in 0..=qi {
                    let v_offset = vi * d + head * h_dim;
                    let mut dot = 0.0f32;
                    for j in 0..h_dim {
                        dot += grad_attn_pre_out[q_offset + j] * cache.v[v_offset + j];
                    }
                    grad_aw[vi] = dot;
                }

                let dot_sum: f32 = (0..=qi).map(|i| {
                    cache.attn_weights[aw_base + i] * grad_aw[i]
                }).sum();
                let mut grad_scores = vec![0.0f32; qi + 1];
                for i in 0..=qi {
                    let p = cache.attn_weights[aw_base + i];
                    grad_scores[i] = p * (grad_aw[i] - dot_sum);
                }

                let scale = 1.0 / (h_dim as f32).sqrt();
                for ki in 0..=qi {
                    let k_offset = ki * d + head * h_dim;
                    for j in 0..h_dim {
                        grad_q[q_offset + j] += grad_scores[ki] * cache.k[k_offset + j] * scale;
                        grad_k[k_offset + j] += grad_scores[ki] * cache.q[q_offset + j] * scale;
                    }
                }
            }
        }

        for t in 0..seq_len {
            for head in 0..n_heads {
                let offset = t * d + head * h_dim;
                rope_backward(&mut grad_q[offset..offset + h_dim], &layer.attention.rope, t);
                rope_backward(&mut grad_k[offset..offset + h_dim], &layer.attention.rope, t);
            }
        }

        let mut grad_x_normed = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            let x_t = &cache.x_normed[t * d..(t + 1) * d];
            let gq = self.ste_backward_adamw(
                &mut layer.attention.wq,
                &grad_q[t * d..(t + 1) * d], x_t, lr, base_param_idx,
            );
            let gk = self.ste_backward_adamw(
                &mut layer.attention.wk,
                &grad_k[t * d..(t + 1) * d], x_t, lr, base_param_idx + 1,
            );
            let gv = self.ste_backward_adamw(
                &mut layer.attention.wv,
                &grad_v[t * d..(t + 1) * d], x_t, lr, base_param_idx + 2,
            );
            for j in 0..d {
                grad_x_normed[t * d + j] = gq[j] + gk[j] + gv[j];
            }
        }

        for t in 0..seq_len {
            let x_t = &cache.x_pre_norm[t * d..(t + 1) * d];
            let grad_xn = &grad_x_normed[t * d..(t + 1) * d];
            let grad_x = rmsnorm_backward(
                x_t, grad_xn,
                &layer.attn_norm.weight, layer.attn_norm.eps,
            );
            for j in 0..d {
                grad_input[t * d + j] += grad_x[j];
            }
        }

        grad_input
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LMConfig;
    use crate::tokenizer::ByteTokenizer;

    #[test]
    fn test_softmax_valid() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax should sum to 1, got {sum}");
    }

    #[test]
    fn test_rmsnorm_backward_gradient() {
        let x = vec![0.5, -0.3, 1.2, -0.8];
        let scale = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;
        let grad_output = vec![1.0, 0.5, -0.3, 0.8];
        let analytical = rmsnorm_backward(&x, &grad_output, &scale, eps);
        let h = 1e-4;
        for i in 0..x.len() {
            let mut x_p = x.clone();
            let mut x_m = x.clone();
            x_p[i] += h;
            x_m[i] -= h;
            let rms_p = (x_p.iter().map(|v| v*v).sum::<f32>() / x_p.len() as f32 + eps).sqrt();
            let rms_m = (x_m.iter().map(|v| v*v).sum::<f32>() / x_m.len() as f32 + eps).sqrt();
            let y_p: Vec<f32> = x_p.iter().zip(&scale).map(|(xi, si)| xi / rms_p * si).collect();
            let y_m: Vec<f32> = x_m.iter().zip(&scale).map(|(xi, si)| xi / rms_m * si).collect();
            let loss_p: f32 = y_p.iter().zip(&grad_output).map(|(y, g)| y * g).sum();
            let loss_m: f32 = y_m.iter().zip(&grad_output).map(|(y, g)| y * g).sum();
            let numerical = (loss_p - loss_m) / (2.0 * h);
            assert!(
                (analytical[i] - numerical).abs() < 0.01,
                "RMSNorm grad mismatch at {i}: analytical={}, numerical={numerical}",
                analytical[i]
            );
        }
    }

    #[test]
    fn test_gradient_clipping() {
        let mut grad = vec![3.0, 4.0];
        clip_grad(&mut grad, 1.0);
        let norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lr_schedule() {
        let cfg = AdvancedConfig {
            lr: 0.01,
            warmup_steps: 10,
            total_steps: 100,
            ..AdvancedConfig::default()
        };
        // During warmup: linearly increasing
        assert!(cfg.lr_at_step(0) < cfg.lr_at_step(5));
        assert!(cfg.lr_at_step(5) < cfg.lr_at_step(9));
        // At warmup end: peak
        let peak = cfg.lr_at_step(10);
        assert!((peak - 0.01).abs() < 0.002);
        // After warmup: cosine decay
        assert!(cfg.lr_at_step(50) < peak);
        assert!(cfg.lr_at_step(90) < cfg.lr_at_step(50));
        // Never goes below min
        assert!(cfg.lr_at_step(100) > 0.0);
    }

    #[test]
    fn test_compute_loss_finite() {
        let model = VagiLM::new(LMConfig::tiny());
        let tokens = ByteTokenizer::new().encode("Hello world");
        let loss = compute_loss(&model, &tokens);
        assert!(loss.is_finite() && loss > 0.0);
    }

    #[test]
    fn test_sgd_training_reduces_loss() {
        let mut model = VagiLM::new(LMConfig::tiny());
        let tokens = ByteTokenizer::new().encode("AAAA");
        let cfg = TrainConfig::for_tiny();
        let initial_loss = compute_loss(&model, &tokens);
        for _ in 0..30 {
            train_step_cfg(&mut model, &tokens, &cfg);
        }
        let final_loss = compute_loss(&model, &tokens);
        eprintln!("SGD: {initial_loss:.4} → {final_loss:.4}");
        assert!(final_loss < initial_loss);
    }

    #[test]
    fn test_adamw_trainer_basic() {
        let mut model = VagiLM::new(LMConfig::tiny());
        let cfg = AdvancedConfig::default();
        let mut trainer = LMTrainer::new(&model, cfg);
        let tokens = ByteTokenizer::new().encode("Hi");
        let m = trainer.train_step(&mut model, &tokens);
        assert!(m.loss.is_finite() && m.loss > 0.0);
        assert!(m.perplexity > 1.0);
    }

    /// The main benchmark: AdamW vs SGD convergence comparison.
    #[test]
    fn test_adamw_vs_sgd_benchmark() {
        let tok = ByteTokenizer::new();
        let pattern = tok.encode("ABCABC");
        let n_steps = 200;

        // ── SGD baseline ──
        let mut sgd_model = VagiLM::new(LMConfig::tiny());
        let sgd_initial = compute_loss(&sgd_model, &pattern);
        let sgd_cfg = TrainConfig::for_tiny();
        for _ in 0..n_steps {
            train_step_cfg(&mut sgd_model, &pattern, &sgd_cfg);
        }
        let sgd_final = compute_loss(&sgd_model, &pattern);
        let sgd_improvement = (sgd_initial - sgd_final) / sgd_initial * 100.0;

        // ── AdamW ──
        let mut adam_model = VagiLM::new(LMConfig::tiny());
        let adam_initial = compute_loss(&adam_model, &pattern);
        let adam_cfg = AdvancedConfig {
            lr: 0.01,
            warmup_steps: 5,
            total_steps: n_steps,
            label_smoothing: 0.1,
            weight_decay: 0.001, // low for tiny dataset
            ..AdvancedConfig::default()
        };
        let mut trainer = LMTrainer::new(&adam_model, adam_cfg);
        let mut last_metrics = TrainMetrics { loss: 0.0, perplexity: 1.0, accuracy: 0.0, lr: 0.0 };

        for step in 0..n_steps {
            last_metrics = trainer.train_step(&mut adam_model, &pattern);
            if step % 40 == 0 {
                let eval = compute_loss(&adam_model, &pattern);
                eprintln!(
                    "AdamW step {:3}: loss={:.4}, acc={:.1}%, ppl={:.2}, lr={:.5}, eval={:.4}",
                    step, last_metrics.loss, last_metrics.accuracy * 100.0,
                    last_metrics.perplexity, last_metrics.lr, eval,
                );
            }
        }
        let adam_final = compute_loss(&adam_model, &pattern);
        let adam_improvement = (adam_initial - adam_final) / adam_initial * 100.0;

        eprintln!("═══ SGD vs AdamW Benchmark ═══");
        eprintln!("SGD:   {sgd_initial:.4} → {sgd_final:.4} ({sgd_improvement:.1}%)");
        eprintln!("AdamW: {adam_initial:.4} → {adam_final:.4} ({adam_improvement:.1}%)");
        eprintln!("Final accuracy: {:.1}%", last_metrics.accuracy * 100.0);
        eprintln!("Final perplexity: {:.2}", adam_final.exp());
        eprintln!("══════════════════════════════");

        // Both should achieve significant improvement
        assert!(
            adam_improvement > 70.0,
            "AdamW should reduce loss by >70%, got {adam_improvement:.1}%"
        );
    }

    /// Test multi-pattern learning with AdamW.
    #[test]
    fn test_adamw_multi_pattern() {
        let tok = ByteTokenizer::new();
        let mut model = VagiLM::new(LMConfig::tiny());
        let patterns: Vec<Vec<u32>> = vec![
            tok.encode("AAAA"),
            tok.encode("BBBB"),
            tok.encode("ABAB"),
            tok.encode("AABB"),
        ];

        let initial_loss: f32 = patterns.iter()
            .map(|p| compute_loss(&model, p))
            .sum::<f32>() / patterns.len() as f32;

        let cfg = AdvancedConfig {
            lr: 0.01,
            warmup_steps: 5,
            total_steps: 200,
            label_smoothing: 0.1,
            ..AdvancedConfig::default()
        };
        let mut trainer = LMTrainer::new(&model, cfg);

        for _epoch in 0..30 {
            for pattern in &patterns {
                trainer.train_step(&mut model, pattern);
            }
        }

        let final_loss: f32 = patterns.iter()
            .map(|p| compute_loss(&model, p))
            .sum::<f32>() / patterns.len() as f32;
        let improvement = (initial_loss - final_loss) / initial_loss * 100.0;

        eprintln!("AdamW multi-pattern: {initial_loss:.4} → {final_loss:.4} ({improvement:.1}%)");
        assert!(
            improvement > 70.0,
            "Multi-pattern should improve >70%: {improvement:.1}%"
        );
    }

    /// Test accuracy tracking reaches high values.
    #[test]
    fn test_adamw_accuracy_convergence() {
        let tok = ByteTokenizer::new();
        let tokens = tok.encode("ABCABC");
        let mut model = VagiLM::new(LMConfig::tiny());
        let cfg = AdvancedConfig {
            lr: 0.01,
            warmup_steps: 5,
            total_steps: 200,
            ..AdvancedConfig::default()
        };
        let mut trainer = LMTrainer::new(&model, cfg);

        let mut best_acc = 0.0f32;
        for _ in 0..150 {
            let m = trainer.train_step(&mut model, &tokens);
            if m.accuracy > best_acc { best_acc = m.accuracy; }
        }

        eprintln!("Best accuracy reached: {:.1}%", best_acc * 100.0);
        assert!(
            best_acc > 0.5,
            "Should reach >50% accuracy, got {:.1}%", best_acc * 100.0
        );
    }

    #[test]
    fn test_train_epoch_sgd() {
        let mut model = VagiLM::new(LMConfig::tiny());
        let tok = ByteTokenizer::new();
        let seqs: Vec<Vec<u32>> = vec![tok.encode("AB"), tok.encode("CD")];
        let loss = train_epoch(&mut model, &seqs, 0.01);
        assert!(loss.is_finite() && loss > 0.0);
    }
}

