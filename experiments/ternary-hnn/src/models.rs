//! 4 models for the Hamiltonian experiment.
//!
//! Model 1: HNN-FP32    — f32 weights, SiLU, Hamiltonian
//! Model 2: HNN-Ternary  — ternary weights, SiLU, Hamiltonian
//! Model 3: HNN-Adaptive — ternary weights, AdaptiveBasis, Hamiltonian
//! Model 4: MLP-FP32     — f32 weights, SiLU, direct prediction (no Hamiltonian)

use rand::Rng;

// ── Shared ────────────────────────────────────────────────────

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// Simple f32 linear layer.
#[derive(Clone)]
struct LinearFP32 {
    w: Vec<f32>, // [out × in]
    b: Vec<f32>, // [out]
    in_features: usize,
    out_features: usize,
}

impl LinearFP32 {
    fn new(in_f: usize, out_f: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / (in_f + out_f) as f32).sqrt();
        let w: Vec<f32> = (0..out_f * in_f).map(|_| rng.gen_range(-scale..scale)).collect();
        let b = vec![0.0f32; out_f];
        Self { w, b, in_features: in_f, out_features: out_f }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = self.b.clone();
        for m in 0..self.out_features {
            for n in 0..self.in_features {
                y[m] += self.w[m * self.in_features + n] * x[n];
            }
        }
        y
    }

    fn param_count(&self) -> usize { self.w.len() + self.b.len() }
    fn memory_bytes(&self) -> usize { (self.w.len() + self.b.len()) * 4 }
}

/// Ternary linear layer (quantized from latent f32 via STE).
#[derive(Clone)]
struct LinearTernary {
    w_latent: Vec<f32>,  // shadow weights
    b: Vec<f32>,
    in_features: usize,
    out_features: usize,
    gamma: f32,
}

impl LinearTernary {
    fn new(in_f: usize, out_f: usize, rng: &mut impl Rng) -> Self {
        let scale = (2.0 / (in_f + out_f) as f32).sqrt();
        let w_latent: Vec<f32> = (0..out_f * in_f).map(|_| rng.gen_range(-scale..scale)).collect();
        let b = vec![0.0f32; out_f];
        Self { w_latent, b, in_features: in_f, out_features: out_f, gamma: 0.7 }
    }

    fn quantize_weight(&self, w: f32, threshold: f32) -> f32 {
        if w >= threshold { 1.0 }
        else if w <= -threshold { -1.0 }
        else { 0.0 }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = self.b.clone();
        for m in 0..self.out_features {
            // Compute row threshold
            let row_start = m * self.in_features;
            let abs_mean: f32 = self.w_latent[row_start..row_start + self.in_features]
                .iter().map(|w| w.abs()).sum::<f32>() / self.in_features as f32;
            let threshold = self.gamma * abs_mean;

            for n in 0..self.in_features {
                let w = self.quantize_weight(self.w_latent[row_start + n], threshold);
                y[m] += w * x[n];
            }
        }
        y
    }

    fn param_count(&self) -> usize { self.w_latent.len() + self.b.len() }

    fn memory_bytes_quantized(&self) -> usize {
        // Ternary: 2 bits per weight
        (self.w_latent.len() * 2 + 7) / 8 + self.b.len() * 4
    }
}

/// AdaptiveBasis activation (trimmed 3-basis: identity + sin + tanh).
#[derive(Clone)]
struct AdaptiveActivation {
    weights: [f32; 3], // [identity, sin, tanh]
}

impl AdaptiveActivation {
    fn new() -> Self {
        // SiLU decomposition init
        Self { weights: [0.5, 0.0, 0.5] }
    }

    fn forward(&self, x: f32) -> f32 {
        self.weights[0] * x
            + self.weights[1] * x.sin()
            + self.weights[2] * x.tanh()
    }
}

// ── Model 1: HNN-FP32 ────────────────────────────────────────

/// Hamiltonian NN with float32 weights.
#[derive(Clone)]
pub struct HNNFP32 {
    layers: Vec<LinearFP32>,
    d_state: usize,
}

impl HNNFP32 {
    pub fn new(d_state: usize, hidden: usize, n_layers: usize, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let input_dim = 2 * d_state;
        let mut layers = Vec::new();
        // First layer
        layers.push(LinearFP32::new(input_dim, hidden, &mut rng));
        // Hidden layers
        for _ in 1..n_layers {
            layers.push(LinearFP32::new(hidden, hidden, &mut rng));
        }
        // Output layer (scalar energy)
        layers.push(LinearFP32::new(hidden, 1, &mut rng));
        Self { layers, d_state }
    }
}

impl HNNModel for HNNFP32 {
    fn hamiltonian(&self, state: &[f32]) -> f32 {
        let mut x = state.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < self.layers.len() - 1 {
                x.iter_mut().for_each(|v| *v = silu(*v));
            }
        }
        x[0]
    }
    fn d_state(&self) -> usize { self.d_state }
    fn param_count(&self) -> usize { self.layers.iter().map(|l| l.param_count()).sum() }
    fn memory_bytes(&self) -> usize { self.layers.iter().map(|l| l.memory_bytes()).sum() }
    fn name(&self) -> &str { "HNN-FP32" }

    fn update_weights(&mut self, grad_states: &[(Vec<f32>, Vec<f32>)], lr: f32) {
        // Simple numerical gradient descent on loss w.r.t. all parameters
        let eps = 1e-4;
        for layer_idx in 0..self.layers.len() {
            let n_w = self.layers[layer_idx].w.len();
            for wi in 0..n_w {
                let loss0 = self.compute_loss(grad_states);
                self.layers[layer_idx].w[wi] += eps;
                let loss1 = self.compute_loss(grad_states);
                self.layers[layer_idx].w[wi] -= eps;
                let grad = (loss1 - loss0) / eps;
                self.layers[layer_idx].w[wi] -= lr * grad;
            }
        }
    }
}

// ── Model 2: HNN-Ternary ─────────────────────────────────────

/// Hamiltonian NN with ternary (STE) weights.
#[derive(Clone)]
pub struct HNNTernary {
    layers: Vec<LinearTernary>,
    d_state: usize,
}

impl HNNTernary {
    pub fn new(d_state: usize, hidden: usize, n_layers: usize, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let input_dim = 2 * d_state;
        let mut layers = Vec::new();
        layers.push(LinearTernary::new(input_dim, hidden, &mut rng));
        for _ in 1..n_layers {
            layers.push(LinearTernary::new(hidden, hidden, &mut rng));
        }
        layers.push(LinearTernary::new(hidden, 1, &mut rng));
        Self { layers, d_state }
    }
}

impl HNNModel for HNNTernary {
    fn hamiltonian(&self, state: &[f32]) -> f32 {
        let mut x = state.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < self.layers.len() - 1 {
                x.iter_mut().for_each(|v| *v = silu(*v));
            }
        }
        x[0]
    }
    fn d_state(&self) -> usize { self.d_state }
    fn param_count(&self) -> usize { self.layers.iter().map(|l| l.param_count()).sum() }
    fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes_quantized()).sum()
    }
    fn name(&self) -> &str { "HNN-Ternary" }

    fn update_weights(&mut self, grad_states: &[(Vec<f32>, Vec<f32>)], lr: f32) {
        let eps = 1e-4;
        for layer_idx in 0..self.layers.len() {
            let n_w = self.layers[layer_idx].w_latent.len();
            for wi in 0..n_w {
                let loss0 = self.compute_loss(grad_states);
                self.layers[layer_idx].w_latent[wi] += eps;
                let loss1 = self.compute_loss(grad_states);
                self.layers[layer_idx].w_latent[wi] -= eps;
                let grad = (loss1 - loss0) / eps;
                // STE: update latent weights (clipped)
                if self.layers[layer_idx].w_latent[wi].abs() < 2.0 {
                    self.layers[layer_idx].w_latent[wi] -= lr * grad;
                }
            }
        }
    }
}

// ── Model 3: HNN-Adaptive ────────────────────────────────────

/// Hamiltonian NN with ternary weights + AdaptiveBasis.
#[derive(Clone)]
pub struct HNNAdaptive {
    layers: Vec<LinearTernary>,
    activations: Vec<AdaptiveActivation>,
    d_state: usize,
}

impl HNNAdaptive {
    pub fn new(d_state: usize, hidden: usize, n_layers: usize, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let input_dim = 2 * d_state;
        let mut layers = Vec::new();
        let mut activations = Vec::new();
        layers.push(LinearTernary::new(input_dim, hidden, &mut rng));
        activations.push(AdaptiveActivation::new());
        for _ in 1..n_layers {
            layers.push(LinearTernary::new(hidden, hidden, &mut rng));
            activations.push(AdaptiveActivation::new());
        }
        layers.push(LinearTernary::new(hidden, 1, &mut rng));
        Self { layers, activations, d_state }
    }
}

impl HNNModel for HNNAdaptive {
    fn hamiltonian(&self, state: &[f32]) -> f32 {
        let mut x = state.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < self.layers.len() - 1 {
                x.iter_mut().for_each(|v| *v = self.activations[i].forward(*v));
            }
        }
        x[0]
    }
    fn d_state(&self) -> usize { self.d_state }
    fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum::<usize>()
            + self.activations.len() * 3
    }
    fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes_quantized()).sum::<usize>()
            + self.activations.len() * 12
    }
    fn name(&self) -> &str { "HNN-Adaptive" }

    fn update_weights(&mut self, grad_states: &[(Vec<f32>, Vec<f32>)], lr: f32) {
        let eps = 1e-4;
        // Update ternary latent weights
        for layer_idx in 0..self.layers.len() {
            let n_w = self.layers[layer_idx].w_latent.len();
            for wi in 0..n_w {
                let loss0 = self.compute_loss(grad_states);
                self.layers[layer_idx].w_latent[wi] += eps;
                let loss1 = self.compute_loss(grad_states);
                self.layers[layer_idx].w_latent[wi] -= eps;
                let grad = (loss1 - loss0) / eps;
                if self.layers[layer_idx].w_latent[wi].abs() < 2.0 {
                    self.layers[layer_idx].w_latent[wi] -= lr * grad;
                }
            }
        }
        // Update basis weights with 10× higher lr (clamped gradient)
        let basis_lr = lr * 10.0;
        for act_idx in 0..self.activations.len() {
            for bi in 0..3 {
                let loss0 = self.compute_loss(grad_states);
                self.activations[act_idx].weights[bi] += eps;
                let loss1 = self.compute_loss(grad_states);
                self.activations[act_idx].weights[bi] -= eps;
                let grad = ((loss1 - loss0) / eps).clamp(-1.0, 1.0);
                self.activations[act_idx].weights[bi] -= basis_lr * grad;
            }
        }
    }
}

impl HNNAdaptive {
    /// Update only ternary weights (freeze basis). Used during warmup.
    pub fn update_weights_only(&mut self, grad_states: &[(Vec<f32>, Vec<f32>)], lr: f32) {
        let eps = 1e-4;
        for layer_idx in 0..self.layers.len() {
            let n_w = self.layers[layer_idx].w_latent.len();
            for wi in 0..n_w {
                let loss0 = self.compute_loss(grad_states);
                self.layers[layer_idx].w_latent[wi] += eps;
                let loss1 = self.compute_loss(grad_states);
                self.layers[layer_idx].w_latent[wi] -= eps;
                let grad = (loss1 - loss0) / eps;
                if self.layers[layer_idx].w_latent[wi].abs() < 2.0 {
                    self.layers[layer_idx].w_latent[wi] -= lr * grad;
                }
            }
        }
    }

    /// Get current basis weights (for logging).
    pub fn basis_weights(&self) -> Vec<[f32; 3]> {
        self.activations.iter().map(|a| a.weights).collect()
    }
}

// ── Model 4: MLP-FP32 (no Hamiltonian) ───────────────────────

/// Standard MLP that directly predicts next state. NOT Hamiltonian.
#[derive(Clone)]
pub struct MLPFP32 {
    layers: Vec<LinearFP32>,
    d_state: usize,
}

impl MLPFP32 {
    pub fn new(d_state: usize, hidden: usize, n_layers: usize, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let io_dim = 2 * d_state;
        let mut layers = Vec::new();
        layers.push(LinearFP32::new(io_dim, hidden, &mut rng));
        for _ in 1..n_layers {
            layers.push(LinearFP32::new(hidden, hidden, &mut rng));
        }
        layers.push(LinearFP32::new(hidden, io_dim, &mut rng)); // output = state dim
        Self { layers, d_state }
    }

    /// Direct next-state prediction: (q,p) → (dq/dt, dp/dt)
    pub fn predict_derivatives(&self, state: &[f32]) -> Vec<f32> {
        let mut x = state.to_vec();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            if i < self.layers.len() - 1 {
                x.iter_mut().for_each(|v| *v = silu(*v));
            }
        }
        x
    }

    pub fn d_state(&self) -> usize { self.d_state }
    pub fn param_count(&self) -> usize { self.layers.iter().map(|l| l.param_count()).sum() }
    pub fn memory_bytes(&self) -> usize { self.layers.iter().map(|l| l.memory_bytes()).sum() }
    pub fn name(&self) -> &str { "MLP-FP32" }

    pub fn update_weights(&mut self, training_pairs: &[(Vec<f32>, Vec<f32>)], lr: f32) {
        let eps = 1e-4;
        for layer_idx in 0..self.layers.len() {
            let n_w = self.layers[layer_idx].w.len();
            for wi in 0..n_w {
                let loss0 = self.loss(training_pairs);
                self.layers[layer_idx].w[wi] += eps;
                let loss1 = self.loss(training_pairs);
                self.layers[layer_idx].w[wi] -= eps;
                let grad = (loss1 - loss0) / eps;
                self.layers[layer_idx].w[wi] -= lr * grad;
            }
        }
    }

    fn loss(&self, pairs: &[(Vec<f32>, Vec<f32>)]) -> f32 {
        let n = pairs.len() as f32;
        pairs.iter().map(|(state, target_deriv)| {
            let pred = self.predict_derivatives(state);
            pred.iter().zip(target_deriv.iter())
                .map(|(p, t)| (p - t) * (p - t))
                .sum::<f32>()
        }).sum::<f32>() / n
    }
}

// ── HNN Trait ─────────────────────────────────────────────────

/// Trait for Hamiltonian models.
pub trait HNNModel {
    /// Compute scalar Hamiltonian (energy).
    fn hamiltonian(&self, state: &[f32]) -> f32;
    fn d_state(&self) -> usize;
    fn param_count(&self) -> usize;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &str;

    /// Update weights given training pairs (state, target_derivatives).
    fn update_weights(&mut self, grad_states: &[(Vec<f32>, Vec<f32>)], lr: f32);

    /// Compute dynamics via finite-difference Hamiltonian gradient.
    fn dynamics(&self, state: &[f32]) -> Vec<f32> {
        let d = self.d_state();
        let eps = 1e-4;
        let mut derivs = vec![0.0f32; 2 * d];
        let h0 = self.hamiltonian(state);

        // dq_i/dt = ∂H/∂p_i
        for i in 0..d {
            let mut s = state.to_vec();
            s[d + i] += eps;
            derivs[i] = (self.hamiltonian(&s) - h0) / eps;
        }
        // dp_i/dt = -∂H/∂q_i
        for i in 0..d {
            let mut s = state.to_vec();
            s[i] += eps;
            derivs[d + i] = -(self.hamiltonian(&s) - h0) / eps;
        }
        derivs
    }

    /// Leapfrog integration using learned Hamiltonian.
    fn integrate(&self, state: &[f32], dt: f32, steps: usize) -> Vec<Vec<f32>> {
        let d = self.d_state();
        let mut traj = Vec::with_capacity(steps + 1);
        let mut s = state.to_vec();
        traj.push(s.clone());

        for _ in 0..steps {
            // Half step p
            let derivs1 = self.dynamics(&s);
            for i in d..2 * d {
                s[i] += 0.5 * dt * derivs1[i];
            }
            // Full step q
            let derivs2 = self.dynamics(&s);
            for i in 0..d {
                s[i] += dt * derivs2[i];
            }
            // Half step p
            let derivs3 = self.dynamics(&s);
            for i in d..2 * d {
                s[i] += 0.5 * dt * derivs3[i];
            }
            traj.push(s.clone());
        }
        traj
    }

    /// Loss: MSE of predicted vs target derivatives.
    fn compute_loss(&self, pairs: &[(Vec<f32>, Vec<f32>)]) -> f32 {
        let n = pairs.len() as f32;
        if n == 0.0 { return 0.0; }
        pairs.iter().map(|(state, target_deriv)| {
            let pred_deriv = self.dynamics(state);
            pred_deriv.iter().zip(target_deriv.iter())
                .map(|(p, t)| (p - t) * (p - t))
                .sum::<f32>()
        }).sum::<f32>() / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnn_fp32_forward() {
        let m = HNNFP32::new(1, 16, 2, 42);
        let h = m.hamiltonian(&[1.0, 0.5]);
        assert!(h.is_finite());
    }

    #[test]
    fn test_hnn_ternary_forward() {
        let m = HNNTernary::new(1, 16, 2, 42);
        let h = m.hamiltonian(&[1.0, 0.5]);
        assert!(h.is_finite());
    }

    #[test]
    fn test_hnn_adaptive_forward() {
        let m = HNNAdaptive::new(1, 16, 2, 42);
        let h = m.hamiltonian(&[1.0, 0.5]);
        assert!(h.is_finite());
    }

    #[test]
    fn test_mlp_forward() {
        let m = MLPFP32::new(1, 16, 2, 42);
        let d = m.predict_derivatives(&[1.0, 0.5]);
        assert_eq!(d.len(), 2);
        assert!(d.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_hnn_dynamics() {
        let m = HNNFP32::new(1, 16, 2, 42);
        let d = m.dynamics(&[1.0, 0.5]);
        assert_eq!(d.len(), 2);
        assert!(d.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_hnn_integration() {
        let m = HNNFP32::new(1, 16, 2, 42);
        let traj = m.integrate(&[1.0, 0.0], 0.01, 10);
        assert_eq!(traj.len(), 11);
        assert!(traj.iter().all(|s| s.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn test_memory_ternary_smaller() {
        let fp32 = HNNFP32::new(1, 64, 3, 42);
        let ternary = HNNTernary::new(1, 64, 3, 42);
        assert!(ternary.memory_bytes() < fp32.memory_bytes(),
            "Ternary {} should be < FP32 {}", ternary.memory_bytes(), fp32.memory_bytes());
    }
}
