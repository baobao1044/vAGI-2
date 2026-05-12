//! Language model training - cross-entropy loss + backpropagation.
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

/// Cached intermediates from one transformer layer's forward pass.
struct LayerCache {
    x_normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_weights: Vec<f32>,
    attn_pre_out: Vec<f32>,
    h: Vec<f32>,
    h_normed: Vec<f32>,
    h_pre_norm: Vec<f32>,
    x_pre_norm: Vec<f32>,
    ffn_up_out: Vec<Vec<f32>>,
    ffn_activated: Vec<Vec<f32>>,
}

/// Training hyperparameters.
pub struct TrainConfig {
    pub lr: f32,
    pub grad_clip: f32,
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
    pub fn for_tiny() -> Self {
        Self::default()
    }
}

/// Compute cross-entropy loss without updating weights.
pub fn compute_loss(model: &VagiLM, tokens: &[u32]) -> f32 {
    if tokens.len() < 2 {
        return 0.0;
    }
    let seq_len = tokens.len() - 1;
    let vocab = model.config.vocab_size;
    let logits = model.forward(&tokens[..seq_len]);
    let targets = &tokens[1..];

    let mut total_loss = 0.0f32;
    for t in 0..seq_len {
        let tok_logits = &logits[t * vocab..(t + 1) * vocab];
        let target = targets[t] as usize;
        let probs = softmax(tok_logits);
        total_loss += -probs[target].max(1e-10).ln();
    }

    total_loss / seq_len as f32
}

/// One training step: forward -> loss -> backward -> update weights.
pub fn train_step(model: &mut VagiLM, tokens: &[u32], lr: f32) -> f32 {
    let cfg = TrainConfig {
        lr,
        ..TrainConfig::default()
    };
    train_step_cfg(model, tokens, &cfg)
}

/// Training step with full configuration.
pub fn train_step_cfg(model: &mut VagiLM, tokens: &[u32], cfg: &TrainConfig) -> f32 {
    if tokens.len() < 2 {
        return 0.0;
    }
    let seq_len = tokens.len() - 1;
    let d = model.config.d_model;
    let vocab = model.config.vocab_size;
    let input_tokens = &tokens[..seq_len];
    let targets = &tokens[1..];

    let hidden_0 = model.embedding.forward(input_tokens);

    let mut layer_caches: Vec<LayerCache> = Vec::with_capacity(model.layers.len());
    let mut hidden = hidden_0.clone();

    for layer in &model.layers {
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
    let mut logits = vec![0.0f32; seq_len * vocab];
    for t in 0..seq_len {
        model.lm_head
            .forward(&lm_inputs[t], &mut logits[t * vocab..(t + 1) * vocab]);
    }

    let mut total_loss = 0.0f32;
    let mut grad_logits = vec![0.0f32; seq_len * vocab];

    for t in 0..seq_len {
        let tok_logits = &logits[t * vocab..(t + 1) * vocab];
        let target = targets[t] as usize;
        let probs = softmax(tok_logits);
        total_loss += -probs[target].max(1e-10).ln();

        let scale = 1.0 / seq_len as f32;
        for i in 0..vocab {
            grad_logits[t * vocab + i] = probs[i] * scale;
        }
        grad_logits[t * vocab + target] -= scale;
    }
    total_loss /= seq_len as f32;

    let mut grad_hidden = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let grad_out = &grad_logits[t * vocab..(t + 1) * vocab];
        let grad_in =
            ste_backward_with_grad(&mut model.lm_head, grad_out, &lm_inputs[t], cfg.lr);
        grad_hidden[t * d..(t + 1) * d].copy_from_slice(&grad_in);
    }

    for t in 0..seq_len {
        let x = &pre_norm[t * d..(t + 1) * d];
        let grad_out = grad_hidden[t * d..(t + 1) * d].to_vec();
        let grad_in = rmsnorm_backward(
            x,
            &grad_out,
            &model.final_norm.weight,
            model.final_norm.eps,
        );
        grad_hidden[t * d..(t + 1) * d].copy_from_slice(&grad_in);
    }

    clip_grad(&mut grad_hidden, cfg.grad_clip);

    for layer_idx in (0..model.layers.len()).rev() {
        grad_hidden = backward_layer(
            &mut model.layers[layer_idx],
            &grad_hidden,
            &layer_caches[layer_idx],
            seq_len,
            cfg,
        );
        clip_grad(&mut grad_hidden, cfg.grad_clip);
    }

    for t in 0..seq_len {
        let token = input_tokens[t] as usize;
        let start = token * d;
        for j in 0..d {
            model.embedding.weight[start + j] -= cfg.lr * grad_hidden[t * d + j];
        }
    }

    total_loss
}

pub fn train_epoch(model: &mut VagiLM, sequences: &[Vec<u32>], lr: f32) -> f32 {
    let cfg = TrainConfig {
        lr,
        ..TrainConfig::default()
    };
    train_epoch_cfg(model, sequences, &cfg)
}

pub fn train_epoch_cfg(model: &mut VagiLM, sequences: &[Vec<u32>], cfg: &TrainConfig) -> f32 {
    if sequences.is_empty() {
        return 0.0;
    }

    let mut total_loss = 0.0f32;
    let mut count = 0usize;
    for seq in sequences {
        if seq.len() >= 2 {
            total_loss += train_step_cfg(model, seq, cfg);
            count += 1;
        }
    }

    if count > 0 {
        total_loss / count as f32
    } else {
        0.0
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

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

fn rmsnorm_backward(x: &[f32], grad_output: &[f32], scale: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_sq = inv_rms * inv_rms;

    let dot: f32 = (0..n).map(|j| grad_output[j] * scale[j] * x[j]).sum();
    let coeff = dot * inv_rms_sq / n as f32;

    let mut grad_input = vec![0.0f32; n];
    for i in 0..n {
        grad_input[i] = scale[i] * inv_rms * grad_output[i] - x[i] * inv_rms * coeff;
    }
    grad_input
}

fn ste_backward_with_grad(
    layer: &mut vagi_core::ste::STELinear,
    grad_output: &[f32],
    x: &[f32],
    lr: f32,
) -> Vec<f32> {
    let mut grad_input = vec![0.0f32; layer.in_features];
    for j in 0..layer.in_features {
        let mut sum = 0.0f32;
        for i in 0..layer.out_features {
            sum += layer.w_latent[i * layer.in_features + j] * grad_output[i];
        }
        grad_input[j] = sum;
    }

    layer.backward_update(grad_output, x, lr);
    grad_input
}

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

    let x_pre_norm = x.to_vec();

    let mut x_normed = x.to_vec();
    for t in 0..seq_len {
        layer.attn_norm.forward(&mut x_normed[t * d..(t + 1) * d]);
    }

    let mut q_all = vec![0.0f32; seq_len * d];
    let mut k_all = vec![0.0f32; seq_len * d];
    let mut v_all = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let x_t = &x_normed[t * d..(t + 1) * d];
        attn.wq.forward(x_t, &mut q_all[t * d..(t + 1) * d]);
        attn.wk.forward(x_t, &mut k_all[t * d..(t + 1) * d]);
        attn.wv.forward(x_t, &mut v_all[t * d..(t + 1) * d]);
    }

    for t in 0..seq_len {
        for head in 0..n_heads {
            let offset = t * d + head * h_dim;
            attn.rope.apply(&mut q_all[offset..offset + h_dim], t);
            attn.rope.apply(&mut k_all[offset..offset + h_dim], t);
        }
    }

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

            let max_s = scores[..=qi]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0.0f32; qi + 1];
            let mut sum_exp = 0.0f32;
            for i in 0..=qi {
                exps[i] = (scores[i] - max_s).exp();
                sum_exp += exps[i];
            }
            if sum_exp > 0.0 {
                for exp in &mut exps {
                    *exp /= sum_exp;
                }
            }

            let aw_base = head * seq_len * seq_len + qi * seq_len;
            attn_weights[aw_base..aw_base + qi + 1].copy_from_slice(&exps[..qi + 1]);

            let out_offset = qi * d + head * h_dim;
            for vi in 0..=qi {
                let v_offset = vi * d + head * h_dim;
                let weight = exps[vi];
                for j in 0..h_dim {
                    attn_pre_out[out_offset + j] += weight * v_all[v_offset + j];
                }
            }
        }
    }

    let mut attn_out = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        attn.wo.forward(
            &attn_pre_out[t * d..(t + 1) * d],
            &mut attn_out[t * d..(t + 1) * d],
        );
    }

    let h: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

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

    (
        output,
        LayerCache {
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
        },
    )
}

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

    let mut grad_h = grad_output.to_vec();
    for t in 0..seq_len {
        let grad_down_out = &grad_output[t * d..(t + 1) * d];
        let grad_activated =
            ste_backward_with_grad(&mut layer.ffn_down, grad_down_out, &cache.ffn_activated[t], lr);
        let grad_up = layer
            .activation
            .backward_input(&cache.ffn_up_out[t], &grad_activated);
        let grad_h_normed_t = ste_backward_with_grad(
            &mut layer.ffn_up,
            &grad_up,
            &cache.h_normed[t * d..(t + 1) * d],
            lr,
        );

        let h_t = &cache.h_pre_norm[t * d..(t + 1) * d];
        let grad_h_t =
            rmsnorm_backward(h_t, &grad_h_normed_t, &layer.ffn_norm.weight, layer.ffn_norm.eps);
        for j in 0..d {
            grad_h[t * d + j] += grad_h_t[j];
        }
    }

    let mut grad_input = grad_h.clone();
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

    let mut grad_v = vec![0.0f32; seq_len * d];
    let mut grad_q = vec![0.0f32; seq_len * d];
    let mut grad_k = vec![0.0f32; seq_len * d];

    for head in 0..n_heads {
        for qi in 0..seq_len {
            let q_offset = qi * d + head * h_dim;
            let aw_base = head * seq_len * seq_len + qi * seq_len;

            for vi in 0..=qi {
                let v_offset = vi * d + head * h_dim;
                let weight = cache.attn_weights[aw_base + vi];
                for j in 0..h_dim {
                    grad_v[v_offset + j] += weight * grad_attn_pre_out[q_offset + j];
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

            let dot_sum: f32 = (0..=qi)
                .map(|i| cache.attn_weights[aw_base + i] * grad_aw[i])
                .sum();
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
        let gq = ste_backward_with_grad(
            &mut layer.attention.wq,
            &grad_q[t * d..(t + 1) * d],
            x_t,
            lr,
        );
        let gk = ste_backward_with_grad(
            &mut layer.attention.wk,
            &grad_k[t * d..(t + 1) * d],
            x_t,
            lr,
        );
        let gv = ste_backward_with_grad(
            &mut layer.attention.wv,
            &grad_v[t * d..(t + 1) * d],
            x_t,
            lr,
        );
        for j in 0..d {
            grad_x_normed[t * d + j] = gq[j] + gk[j] + gv[j];
        }
    }

    for t in 0..seq_len {
        let x_t = &cache.x_pre_norm[t * d..(t + 1) * d];
        let grad_xn = &grad_x_normed[t * d..(t + 1) * d];
        let grad_x =
            rmsnorm_backward(x_t, grad_xn, &layer.attn_norm.weight, layer.attn_norm.eps);
        for j in 0..d {
            grad_input[t * d + j] += grad_x[j];
        }
    }

    grad_input
}

fn rope_backward(grad: &mut [f32], rope: &crate::attention::RoPECache, pos: usize) {
    let half = grad.len() / 2;
    let base = pos * half;
    for i in 0..half {
        let g0 = grad[2 * i];
        let g1 = grad[2 * i + 1];
        let c = rope.cos[base + i];
        let s = rope.sin[base + i];
        grad[2 * i] = g0 * c + g1 * s;
        grad[2 * i + 1] = -g0 * s + g1 * c;
    }
}

/// Advanced training configuration.
#[derive(Clone)]
pub struct AdvancedConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
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
        if self.warmup_steps > 0 && step < self.warmup_steps {
            self.lr * (step + 1) as f32 / self.warmup_steps as f32
        } else if self.total_steps > self.warmup_steps {
            let decay_steps = self.total_steps - self.warmup_steps;
            let step_in_decay = (step.saturating_sub(self.warmup_steps)).min(decay_steps);
            let progress = step_in_decay as f32 / decay_steps as f32;
            let min_lr = self.lr * 0.1;
            min_lr + 0.5 * (self.lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        } else {
            self.lr
        }
    }
}

/// Aggregate metrics returned by `LMTrainer`.
#[derive(Clone, Copy, Debug)]
pub struct TrainMetrics {
    pub loss: f32,
    pub perplexity: f32,
    pub accuracy: f32,
    pub lr: f32,
}

/// Lightweight trainer wrapper that keeps the public AdamW-facing API stable.
///
/// The optimizer state vectors are retained for checkpoint compatibility even
/// though the current implementation routes weight updates through the verified
/// SGD/STE path above.
pub struct LMTrainer {
    config: AdvancedConfig,
    pub(crate) adam_m: Vec<f32>,
    pub(crate) adam_v: Vec<f32>,
    pub(crate) step: usize,
}

impl LMTrainer {
    pub fn new(model: &VagiLM, config: AdvancedConfig) -> Self {
        let state_len = optimizer_state_len(model);
        Self {
            config,
            adam_m: vec![0.0; state_len],
            adam_v: vec![0.0; state_len],
            step: 0,
        }
    }

    pub fn step_count(&self) -> usize {
        self.step
    }

    pub fn current_lr(&self) -> f32 {
        self.config.lr_at_step(self.step)
    }

    pub fn train_step(&mut self, model: &mut VagiLM, tokens: &[u32]) -> TrainMetrics {
        if tokens.len() < 2 {
            return TrainMetrics {
                loss: 0.0,
                perplexity: 1.0,
                accuracy: 0.0,
                lr: 0.0,
            };
        }

        let lr = self.config.lr_at_step(self.step);
        let cfg = TrainConfig {
            lr,
            grad_clip: self.config.grad_clip,
            weight_decay: self.config.weight_decay,
        };
        let loss = train_step_cfg(model, tokens, &cfg);
        self.step += 1;

        let accuracy = compute_accuracy(model, tokens);
        TrainMetrics {
            loss,
            perplexity: loss.exp(),
            accuracy,
            lr,
        }
    }

    pub fn train_epoch(&mut self, model: &mut VagiLM, sequences: &[Vec<u32>]) -> TrainMetrics {
        let mut total = TrainMetrics {
            loss: 0.0,
            perplexity: 0.0,
            accuracy: 0.0,
            lr: 0.0,
        };
        let mut count = 0usize;

        for seq in sequences {
            if seq.len() >= 2 {
                let metrics = self.train_step(model, seq);
                total.loss += metrics.loss;
                total.perplexity += metrics.perplexity;
                total.accuracy += metrics.accuracy;
                total.lr = metrics.lr;
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
}

fn compute_accuracy(model: &VagiLM, tokens: &[u32]) -> f32 {
    if tokens.len() < 2 {
        return 0.0;
    }

    let seq_len = tokens.len() - 1;
    let vocab = model.config.vocab_size;
    let logits = model.forward(&tokens[..seq_len]);
    let targets = &tokens[1..];
    let mut correct = 0usize;

    for t in 0..seq_len {
        let tok_logits = &logits[t * vocab..(t + 1) * vocab];
        let probs = softmax(tok_logits);
        let predicted = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        if predicted == targets[t] as usize {
            correct += 1;
        }
    }

    correct as f32 / seq_len as f32
}

fn optimizer_state_len(model: &VagiLM) -> usize {
    let mut total = model.embedding.weight.len() + model.lm_head.w_latent.len();
    total += model.final_norm.weight.len();
    for layer in &model.layers {
        total += layer.attention.wq.w_latent.len();
        total += layer.attention.wk.w_latent.len();
        total += layer.attention.wv.w_latent.len();
        total += layer.attention.wo.w_latent.len();
        total += layer.ffn_up.w_latent.len();
        total += layer.ffn_down.w_latent.len();
        total += layer.attn_norm.weight.len();
        total += layer.ffn_norm.weight.len();
        total += layer.activation.weights_slice().len();
    }
    total
}
