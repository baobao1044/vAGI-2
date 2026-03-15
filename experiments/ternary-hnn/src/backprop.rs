//! Analytical backpropagation with STE for HNN models.
//!
//! Key concepts:
//! - Forward pass saves intermediate activations (cache)
//! - Backward pass computes dL/dW analytically via chain rule
//! - STE: forward uses Q(W_latent), backward treats Q as identity
//! - HNN dynamics: finite-diff of H w.r.t. state, then backprop through each

use crate::models::{HNNModel, HNNFP32, HNNTernary, HNNAdaptive, MLPFP32};

// ── Activation helpers ────────────────────────────────────────

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn silu_derivative(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig + x * sig * (1.0 - sig)
}

fn adaptive_forward(x: f32, weights: &[f32; 3]) -> f32 {
    weights[0] * x + weights[1] * x.sin() + weights[2] * x.tanh()
}

fn adaptive_derivatives(x: f32, weights: &[f32; 3]) -> (f32, [f32; 3]) {
    // d(activation)/dx and d(activation)/d(weights)
    let dx = weights[0] + weights[1] * x.cos() + weights[2] * (1.0 - x.tanh().powi(2));
    let dw = [x, x.sin(), x.tanh()]; // d(activation)/d(weight_i)
    (dx, dw)
}

// ── Layer Cache ───────────────────────────────────────────────

/// Cached intermediate values for backward pass.
#[derive(Clone)]
struct LayerCache {
    input: Vec<f32>,          // x (input to this layer)
    pre_activation: Vec<f32>, // W*x + b (before activation)
    in_features: usize,
    out_features: usize,
}

// ── FP32 Layer Forward/Backward ───────────────────────────────

/// Forward pass with cache for FP32 linear layer.
fn forward_fp32_cached(
    w: &[f32], b: &[f32],
    in_f: usize, out_f: usize,
    x: &[f32],
) -> (Vec<f32>, LayerCache) {
    let mut pre_act = b.to_vec();
    for m in 0..out_f {
        for n in 0..in_f {
            pre_act[m] += w[m * in_f + n] * x[n];
        }
    }
    let cache = LayerCache {
        input: x.to_vec(),
        pre_activation: pre_act.clone(),
        in_features: in_f,
        out_features: out_f,
    };
    (pre_act, cache)
}

/// Backward through FP32 linear layer (no activation — activation handled separately).
/// Returns: (grad_w, grad_b, grad_x)
fn backward_fp32(
    w: &[f32],
    cache: &LayerCache,
    grad_output: &[f32], // dL/d(pre_activation) = dL/d(output) * activation'
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let in_f = cache.in_features;
    let out_f = cache.out_features;

    // dL/dW[m,n] = dL/dh[m] * x[n]
    let mut grad_w = vec![0.0f32; out_f * in_f];
    for m in 0..out_f {
        for n in 0..in_f {
            grad_w[m * in_f + n] = grad_output[m] * cache.input[n];
        }
    }

    // dL/db[m] = dL/dh[m]
    let grad_b = grad_output.to_vec();

    // dL/dx[n] = sum_m dL/dh[m] * W[m,n]
    let mut grad_x = vec![0.0f32; in_f];
    for n in 0..in_f {
        for m in 0..out_f {
            grad_x[n] += grad_output[m] * w[m * in_f + n];
        }
    }

    (grad_w, grad_b, grad_x)
}

/// Backward through ternary linear layer with STE.
/// Forward used Q(W_latent), backward treats Q as identity → dL/dW_latent = dL/dh × x^T
/// But for grad_x, we use Q(W_latent)^T (quantized weights in forward).
fn backward_ternary_ste(
    w_latent: &[f32],
    gamma: f32,
    cache: &LayerCache,
    grad_output: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let in_f = cache.in_features;
    let out_f = cache.out_features;

    // STE: dL/dW_latent = dL/dh × x^T (same as FP32 — straight-through!)
    let mut grad_w = vec![0.0f32; out_f * in_f];
    for m in 0..out_f {
        for n in 0..in_f {
            grad_w[m * in_f + n] = grad_output[m] * cache.input[n];
        }
    }

    let grad_b = grad_output.to_vec();

    // For grad_x: use quantized weights (what was actually used in forward)
    let mut grad_x = vec![0.0f32; in_f];
    for n in 0..in_f {
        for m in 0..out_f {
            let row_start = m * in_f;
            let abs_mean: f32 = w_latent[row_start..row_start + in_f]
                .iter().map(|w| w.abs()).sum::<f32>() / in_f as f32;
            let threshold = gamma * abs_mean;
            let w_q = quantize_weight(w_latent[row_start + n], threshold);
            grad_x[n] += grad_output[m] * w_q;
        }
    }

    (grad_w, grad_b, grad_x)
}

fn quantize_weight(w: f32, threshold: f32) -> f32 {
    if w >= threshold { 1.0 }
    else if w <= -threshold { -1.0 }
    else { 0.0 }
}

// ── Full Network Gradient ─────────────────────────────────────

/// Gradient of all FP32 model parameters w.r.t. a scalar output H.
/// Returns flat Vec of all weight/bias gradients.
pub fn gradient_hnn_fp32(model: &HNNFP32, state: &[f32]) -> Vec<f32> {
    let layers = model.layers();
    let n_layers = layers.len();

    // Forward with cache
    let mut x = state.to_vec();
    let mut caches = Vec::with_capacity(n_layers);
    for (i, layer) in layers.iter().enumerate() {
        let (pre_act, cache) = forward_fp32_cached(&layer.w, &layer.b, layer.in_features, layer.out_features, &x);
        caches.push(cache);
        if i < n_layers - 1 {
            x = pre_act.iter().map(|&v| silu(v)).collect();
        } else {
            x = pre_act;
        }
    }
    // x[0] = H (scalar output)

    // Backward: dL/dH = 1.0
    let mut grad = vec![1.0f32]; // gradient w.r.t. output layer's pre_activation

    let mut all_grads = Vec::new();
    let mut layer_grads = Vec::new(); // (grad_w, grad_b) per layer, reversed

    for i in (0..n_layers).rev() {
        let (grad_w, grad_b, grad_x) = backward_fp32(&layers[i].w, &caches[i], &grad);
        layer_grads.push((grad_w, grad_b));

        if i > 0 {
            // Apply activation derivative (SiLU) to get grad w.r.t. previous layer's output
            grad = grad_x.iter().zip(caches[i-1].pre_activation.iter())
                .map(|(&gx, &pa)| gx * silu_derivative(pa))
                .collect();
        }
    }

    // Reverse to get in forward order, flatten
    layer_grads.reverse();
    for (gw, gb) in layer_grads {
        all_grads.extend(gw);
        all_grads.extend(gb);
    }
    all_grads
}

/// Gradient of all ternary model parameters w.r.t. H, using STE.
pub fn gradient_hnn_ternary(model: &HNNTernary, state: &[f32]) -> Vec<f32> {
    let layers = model.layers();
    let n_layers = layers.len();

    // Forward with cache
    let mut x = state.to_vec();
    let mut caches = Vec::with_capacity(n_layers);
    for (i, layer) in layers.iter().enumerate() {
        // Forward uses quantized weights
        let (pre_act, cache) = forward_ternary_cached(
            &layer.w_latent, &layer.b, layer.gamma,
            layer.in_features, layer.out_features, &x,
        );
        caches.push(cache);
        if i < n_layers - 1 {
            x = pre_act.iter().map(|&v| silu(v)).collect();
        } else {
            x = pre_act;
        }
    }

    // Backward
    let mut grad = vec![1.0f32];
    let mut layer_grads = Vec::new();

    for i in (0..n_layers).rev() {
        let (grad_w, grad_b, grad_x) = backward_ternary_ste(
            &layers[i].w_latent, layers[i].gamma, &caches[i], &grad,
        );
        layer_grads.push((grad_w, grad_b));

        if i > 0 {
            grad = grad_x.iter().zip(caches[i-1].pre_activation.iter())
                .map(|(&gx, &pa)| gx * silu_derivative(pa))
                .collect();
        }
    }

    layer_grads.reverse();
    let mut all_grads = Vec::new();
    for (gw, gb) in layer_grads {
        all_grads.extend(gw);
        all_grads.extend(gb);
    }
    all_grads
}

/// Gradient of all adaptive model parameters w.r.t. H, using STE + basis gradients.
pub fn gradient_hnn_adaptive(model: &HNNAdaptive, state: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let layers = model.layers();
    let activations = model.activations();
    let n_layers = layers.len();

    // Forward with cache
    let mut x = state.to_vec();
    let mut caches = Vec::with_capacity(n_layers);
    for (i, layer) in layers.iter().enumerate() {
        let (pre_act, cache) = forward_ternary_cached(
            &layer.w_latent, &layer.b, layer.gamma,
            layer.in_features, layer.out_features, &x,
        );
        caches.push(cache);
        if i < n_layers - 1 {
            x = pre_act.iter().map(|&v| adaptive_forward(v, &activations[i].weights)).collect();
        } else {
            x = pre_act;
        }
    }

    // Backward
    let mut grad = vec![1.0f32];
    let mut layer_grads = Vec::new();
    let mut basis_grads = Vec::new(); // [f32; 3] per activation layer

    for i in (0..n_layers).rev() {
        let (grad_w, grad_b, grad_x) = backward_ternary_ste(
            &layers[i].w_latent, layers[i].gamma, &caches[i], &grad,
        );
        layer_grads.push((grad_w, grad_b));

        if i > 0 {
            // Through AdaptiveBasis activation
            let act = &activations[i-1];
            let mut basis_grad = [0.0f32; 3];

            let mut new_grad = vec![0.0f32; grad_x.len()];
            for (j, (&gx, &pa)) in grad_x.iter().zip(caches[i-1].pre_activation.iter()).enumerate() {
                let (dx, dw) = adaptive_derivatives(pa, &act.weights);
                new_grad[j] = gx * dx;
                // Accumulate basis gradient
                for k in 0..3 {
                    basis_grad[k] += gx * dw[k];
                }
            }
            basis_grads.push(basis_grad);
            grad = new_grad;
        }
    }

    layer_grads.reverse();
    basis_grads.reverse();

    let mut weight_grads = Vec::new();
    for (gw, gb) in layer_grads {
        weight_grads.extend(gw);
        weight_grads.extend(gb);
    }

    let mut flat_basis_grads = Vec::new();
    for bg in basis_grads {
        flat_basis_grads.extend(bg);
    }

    (weight_grads, flat_basis_grads)
}

/// Forward through ternary layer (uses quantized weights) with cache.
fn forward_ternary_cached(
    w_latent: &[f32], b: &[f32], gamma: f32,
    in_f: usize, out_f: usize,
    x: &[f32],
) -> (Vec<f32>, LayerCache) {
    let mut pre_act = b.to_vec();
    for m in 0..out_f {
        let row_start = m * in_f;
        let abs_mean: f32 = w_latent[row_start..row_start + in_f]
            .iter().map(|w| w.abs()).sum::<f32>() / in_f as f32;
        let threshold = gamma * abs_mean;
        for n in 0..in_f {
            let w_q = quantize_weight(w_latent[row_start + n], threshold);
            pre_act[m] += w_q * x[n];
        }
    }
    let cache = LayerCache {
        input: x.to_vec(),
        pre_activation: pre_act.clone(),
        in_features: in_f,
        out_features: out_f,
    };
    (pre_act, cache)
}

// ── MLP Gradient ──────────────────────────────────────────────

/// Gradient of MLP parameters w.r.t. output.
pub fn gradient_mlp_fp32(model: &MLPFP32, state: &[f32]) -> Vec<Vec<f32>> {
    let layers = model.layers();
    let n_layers = layers.len();
    let out_dim = layers.last().unwrap().out_features;

    // Forward with cache
    let mut x = state.to_vec();
    let mut caches = Vec::with_capacity(n_layers);
    for (i, layer) in layers.iter().enumerate() {
        let (pre_act, cache) = forward_fp32_cached(&layer.w, &layer.b, layer.in_features, layer.out_features, &x);
        caches.push(cache);
        if i < n_layers - 1 {
            x = pre_act.iter().map(|&v| silu(v)).collect();
        } else {
            x = pre_act;
        }
    }
    // x = output vector

    // Compute gradient for each output dimension
    let mut all_output_grads = Vec::with_capacity(out_dim);

    for out_idx in 0..out_dim {
        let mut grad = vec![0.0f32; out_dim];
        grad[out_idx] = 1.0;

        let mut layer_grads = Vec::new();

        for i in (0..n_layers).rev() {
            let (grad_w, grad_b, grad_x) = backward_fp32(&layers[i].w, &caches[i], &grad);
            layer_grads.push((grad_w, grad_b));

            if i > 0 {
                grad = grad_x.iter().zip(caches[i-1].pre_activation.iter())
                    .map(|(&gx, &pa)| gx * silu_derivative(pa))
                    .collect();
            }
        }

        layer_grads.reverse();
        let mut flat = Vec::new();
        for (gw, gb) in layer_grads {
            flat.extend(gw);
            flat.extend(gb);
        }
        all_output_grads.push(flat);
    }

    all_output_grads
}

// ── HNN Loss Gradient ─────────────────────────────────────────

/// Compute full loss gradient for HNN-FP32 using analytical backprop.
/// loss = (1/N) Σ ||dynamics(state) - target||²
/// where dynamics = finite-diff of H w.r.t. state
pub fn loss_gradient_fp32(
    model: &HNNFP32,
    pairs: &[(Vec<f32>, Vec<f32>)],
) -> Vec<f32> {
    let n = pairs.len() as f32;
    if n == 0.0 { return vec![]; }

    let d = model.d_state();
    let eps = 1e-4f32;
    let n_params = count_fp32_params(model);
    let mut total_grad = vec![0.0f32; n_params];

    for (state, target_deriv) in pairs {
        // Compute dynamics via finite diff of H
        let grad_h0 = gradient_hnn_fp32(model, state);
        let h0 = model.hamiltonian(state);

        for j in 0..(2 * d) {
            let mut s_perturbed = state.clone();
            if j < d {
                s_perturbed[d + j] += eps; // perturb p_j for dq_j/dt
            } else {
                s_perturbed[j - d] += eps; // perturb q_j for dp_j/dt
            }
            let h_perturbed = model.hamiltonian(&s_perturbed);
            let grad_h_perturbed = gradient_hnn_fp32(model, &s_perturbed);

            let dynamics_j = if j < d {
                (h_perturbed - h0) / eps    // ∂H/∂p_j
            } else {
                -(h_perturbed - h0) / eps   // -∂H/∂q_j
            };

            let residual = dynamics_j - target_deriv[j];

            // ∂dynamics_j/∂W = (∂H_perturbed/∂W - ∂H_0/∂W) / eps * sign
            let sign = if j < d { 1.0 } else { -1.0 };

            for (k, g) in total_grad.iter_mut().enumerate() {
                let d_dynamics_dw = sign * (grad_h_perturbed[k] - grad_h0[k]) / eps;
                *g += 2.0 * residual * d_dynamics_dw / n;
            }
        }
    }

    total_grad
}

/// Compute full loss gradient for HNN-Ternary using STE.
pub fn loss_gradient_ternary(
    model: &HNNTernary,
    pairs: &[(Vec<f32>, Vec<f32>)],
) -> Vec<f32> {
    let n = pairs.len() as f32;
    if n == 0.0 { return vec![]; }

    let d = model.d_state();
    let eps = 1e-4f32;
    let n_params = count_ternary_params(model);
    let mut total_grad = vec![0.0f32; n_params];

    for (state, target_deriv) in pairs {
        let grad_h0 = gradient_hnn_ternary(model, state);
        let h0 = model.hamiltonian(state);

        for j in 0..(2 * d) {
            let mut s_perturbed = state.clone();
            if j < d {
                s_perturbed[d + j] += eps;
            } else {
                s_perturbed[j - d] += eps;
            }
            let h_perturbed = model.hamiltonian(&s_perturbed);
            let grad_h_perturbed = gradient_hnn_ternary(model, &s_perturbed);

            let dynamics_j = if j < d {
                (h_perturbed - h0) / eps
            } else {
                -(h_perturbed - h0) / eps
            };

            let residual = dynamics_j - target_deriv[j];
            let sign = if j < d { 1.0 } else { -1.0 };

            for (k, g) in total_grad.iter_mut().enumerate() {
                let d_dynamics_dw = sign * (grad_h_perturbed[k] - grad_h0[k]) / eps;
                *g += 2.0 * residual * d_dynamics_dw / n;
            }
        }
    }

    total_grad
}

/// Compute full loss gradient for HNN-Adaptive using STE + basis grad.
pub fn loss_gradient_adaptive(
    model: &HNNAdaptive,
    pairs: &[(Vec<f32>, Vec<f32>)],
) -> (Vec<f32>, Vec<f32>) {
    let n = pairs.len() as f32;
    if n == 0.0 { return (vec![], vec![]); }

    let d = model.d_state();
    let eps = 1e-4f32;

    let (sample_wg, sample_bg) = gradient_hnn_adaptive(model, &pairs[0].0);
    let n_weight_params = sample_wg.len();
    let n_basis_params = sample_bg.len();
    let mut total_wgrad = vec![0.0f32; n_weight_params];
    let mut total_bgrad = vec![0.0f32; n_basis_params];

    for (state, target_deriv) in pairs {
        let (grad_w0, grad_b0) = gradient_hnn_adaptive(model, state);
        let h0 = model.hamiltonian(state);

        for j in 0..(2 * d) {
            let mut s_perturbed = state.clone();
            if j < d {
                s_perturbed[d + j] += eps;
            } else {
                s_perturbed[j - d] += eps;
            }
            let h_perturbed = model.hamiltonian(&s_perturbed);
            let (grad_w_p, grad_b_p) = gradient_hnn_adaptive(model, &s_perturbed);

            let dynamics_j = if j < d {
                (h_perturbed - h0) / eps
            } else {
                -(h_perturbed - h0) / eps
            };

            let residual = dynamics_j - target_deriv[j];
            let sign = if j < d { 1.0 } else { -1.0 };

            for (k, g) in total_wgrad.iter_mut().enumerate() {
                let d_dynamics_dw = sign * (grad_w_p[k] - grad_w0[k]) / eps;
                *g += 2.0 * residual * d_dynamics_dw / n;
            }
            for (k, g) in total_bgrad.iter_mut().enumerate() {
                if k < grad_b_p.len() && k < grad_b0.len() {
                    let d_dynamics_db = sign * (grad_b_p[k] - grad_b0[k]) / eps;
                    *g += 2.0 * residual * d_dynamics_db / n;
                }
            }
        }
    }

    (total_wgrad, total_bgrad)
}

/// Compute MLP loss gradient analytically.
pub fn loss_gradient_mlp(
    model: &MLPFP32,
    pairs: &[(Vec<f32>, Vec<f32>)],
) -> Vec<f32> {
    let n = pairs.len() as f32;
    if n == 0.0 { return vec![]; }

    let n_params = count_mlp_params(model);
    let mut total_grad = vec![0.0f32; n_params];

    for (state, target_deriv) in pairs {
        let pred = model.predict_derivatives(state);
        let output_grads = gradient_mlp_fp32(model, state); // Vec<Vec<f32>>, one per output dim

        for (j, (p, t)) in pred.iter().zip(target_deriv.iter()).enumerate() {
            let residual = p - t;
            if j < output_grads.len() {
                for (k, g) in total_grad.iter_mut().enumerate() {
                    if k < output_grads[j].len() {
                        *g += 2.0 * residual * output_grads[j][k] / n;
                    }
                }
            }
        }
    }

    total_grad
}

// ── Parameter counting ────────────────────────────────────────

fn count_fp32_params(model: &HNNFP32) -> usize {
    model.layers().iter().map(|l| l.w.len() + l.b.len()).sum()
}

fn count_ternary_params(model: &HNNTernary) -> usize {
    model.layers().iter().map(|l| l.w_latent.len() + l.b.len()).sum()
}

fn count_mlp_params(model: &MLPFP32) -> usize {
    model.layers().iter().map(|l| l.w.len() + l.b.len()).sum()
}

// ── Weight Update Functions ───────────────────────────────────

/// Apply gradient to FP32 model parameters.
pub fn apply_gradient_fp32(model: &mut HNNFP32, grad: &[f32], lr: f32) {
    let mut idx = 0;
    for layer in model.layers_mut() {
        for w in layer.w.iter_mut() {
            *w -= lr * grad[idx];
            idx += 1;
        }
        for b in layer.b.iter_mut() {
            *b -= lr * grad[idx];
            idx += 1;
        }
    }
}

/// Apply gradient to ternary model latent weights (STE update).
pub fn apply_gradient_ternary(model: &mut HNNTernary, grad: &[f32], lr: f32) {
    let mut idx = 0;
    for layer in model.layers_mut() {
        for w in layer.w_latent.iter_mut() {
            if w.abs() < 2.0 { // STE clipping window
                *w -= lr * grad[idx];
            }
            idx += 1;
        }
        for b in layer.b.iter_mut() {
            *b -= lr * grad[idx];
            idx += 1;
        }
    }
}

/// Apply gradient to adaptive model (STE for weights, separate lr for basis).
pub fn apply_gradient_adaptive(
    model: &mut HNNAdaptive,
    weight_grad: &[f32], basis_grad: &[f32],
    lr: f32, basis_lr: f32,
) {
    let mut idx = 0;
    for layer in model.layers_mut() {
        for w in layer.w_latent.iter_mut() {
            if w.abs() < 2.0 {
                *w -= lr * weight_grad[idx];
            }
            idx += 1;
        }
        for b in layer.b.iter_mut() {
            *b -= lr * weight_grad[idx];
            idx += 1;
        }
    }
    // Apply basis gradient
    let mut bidx = 0;
    for act in model.activations_mut() {
        for i in 0..3 {
            if bidx < basis_grad.len() {
                let g = basis_grad[bidx].clamp(-1.0, 1.0);
                act.weights[i] -= basis_lr * g;
            }
            bidx += 1;
        }
    }
}

/// Apply gradient to MLP model.
pub fn apply_gradient_mlp(model: &mut MLPFP32, grad: &[f32], lr: f32) {
    let mut idx = 0;
    for layer in model.layers_mut() {
        for w in layer.w.iter_mut() {
            *w -= lr * grad[idx];
            idx += 1;
        }
        for b in layer.b.iter_mut() {
            *b -= lr * grad[idx];
            idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_derivative() {
        let x = 1.0f32;
        let eps = 1e-5;
        let numerical = (silu(x + eps) - silu(x - eps)) / (2.0 * eps);
        let analytical = silu_derivative(x);
        assert!((numerical - analytical).abs() < 1e-3,
            "SiLU' at {x}: numerical={numerical:.6} analytical={analytical:.6}");
    }

    #[test]
    fn test_gradient_check_fp32() {
        // Compare analytical gradient with numerical gradient for HNNFP32
        let model = HNNFP32::new(1, 8, 2, 42);
        let state = vec![1.0f32, 0.5];

        let analytical = gradient_hnn_fp32(&model, &state);
        let h0 = model.hamiltonian(&state);

        // Numerical gradient: perturb each param one at a time
        let eps = 1e-4;
        let mut max_rel_err = 0.0f32;
        let n_params = analytical.len();

        for idx in 0..n_params {
            let mut model_copy = model.clone();
            // Perturb param idx
            let mut param_idx = 0;
            let mut done = false;
            for layer in model_copy.layers_mut() {
                for w in layer.w.iter_mut() {
                    if param_idx == idx {
                        *w += eps;
                        done = true;
                    }
                    param_idx += 1;
                    if done { break; }
                }
                if !done {
                    for b in layer.b.iter_mut() {
                        if param_idx == idx {
                            *b += eps;
                            done = true;
                        }
                        param_idx += 1;
                        if done { break; }
                    }
                }
                if done { break; }
            }

            let h1 = model_copy.hamiltonian(&state);
            let numerical = (h1 - h0) / eps;
            let err = (numerical - analytical[idx]).abs()
                / (numerical.abs().max(analytical[idx].abs()).max(1e-7));
            max_rel_err = max_rel_err.max(err);
        }

        assert!(max_rel_err < 0.05,
            "Max relative error between analytical and numerical gradient: {max_rel_err:.6}");
    }

    #[test]
    fn test_gradient_check_ternary_ste() {
        // STE gradient should be non-zero (unlike numerical gradient of quantized forward)
        let model = HNNTernary::new(1, 8, 2, 42);
        let state = vec![1.0f32, 0.5];

        let grad = gradient_hnn_ternary(&model, &state);
        let nonzero_count = grad.iter().filter(|&&g| g.abs() > 1e-10).count();

        // STE should produce non-zero gradients even though forward is quantized
        assert!(nonzero_count > grad.len() / 2,
            "STE gradient should have many non-zero entries: {nonzero_count}/{} = {:.1}%",
            grad.len(), 100.0 * nonzero_count as f64 / grad.len() as f64);
    }

    #[test]
    fn test_loss_gradient_fp32() {
        let model = HNNFP32::new(1, 8, 2, 42);
        let pairs = vec![
            (vec![1.0f32, 0.0], vec![0.0, -1.0]),
            (vec![0.5, 0.5], vec![0.5, -0.5]),
        ];

        let grad = loss_gradient_fp32(&model, &pairs);
        assert!(!grad.is_empty());
        assert!(grad.iter().all(|g| g.is_finite()), "Gradient should be finite");
    }

    #[test]
    fn test_loss_gradient_ternary_nonzero() {
        let model = HNNTernary::new(1, 8, 2, 42);
        let pairs = vec![
            (vec![1.0f32, 0.0], vec![0.0, -1.0]),
        ];

        let grad = loss_gradient_ternary(&model, &pairs);
        let nonzero = grad.iter().filter(|&&g| g.abs() > 1e-10).count();
        assert!(nonzero > 0, "STE loss gradient should have non-zero entries");
    }
}
