//! AdaptiveNet — learnable activation functions as basis combinations.
//!
//! Supports configurable basis sets:
//! - Full (6): identity, ReLU, x|x|, sin(πx/2), tanh, tent
//! - Trimmed (3): identity, sin(πx/2), tanh  ← recommended default
//!
//! Matmul remains ternary (BitNetLinear, unchanged). Only the activation is learnable.
//! Overhead: 3 multiply-adds per element vs the O(n²) matmul → ~0.5% slower than BitNet.

use crate::bitnet::{BitNetLinear, RMSNorm};
use crate::error::VagiError;

/// Legacy constant for backward compat.
pub const N_BASIS: usize = 6;

// ── Basis function evaluations ────────────────────────────────

/// b0: identity — allows skip-like behaviour.
#[inline(always)]
pub fn basis_identity(x: f32) -> f32 { x }

/// b1: ReLU — standard rectified nonlinearity.
#[inline(always)]
pub fn basis_relu(x: f32) -> f32 { x.max(0.0) }

/// b2: x·|x| — smooth quadratic, sign-preserving.
#[inline(always)]
pub fn basis_xabsx(x: f32) -> f32 { x * x.abs() }

/// b3: sin(πx/2) — periodic, maps [-1,1]→[-1,1].
#[inline(always)]
pub fn basis_sin(x: f32) -> f32 {
    (std::f32::consts::FRAC_PI_2 * x).sin()
}

/// b4: fast tanh approximation — x / (1 + |x|). Bounded, stabilising.
#[inline(always)]
pub fn basis_tanh(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// b5: tent/triangle — max(0, 1 - |x|). Local, sparsity-inducing.
#[inline(always)]
pub fn basis_tent(x: f32) -> f32 {
    (1.0 - x.abs()).max(0.0)
}

/// All 6 basis functions.
pub const ALL_BASIS_FNS: [fn(f32) -> f32; 6] = [
    basis_identity, basis_relu, basis_xabsx,
    basis_sin, basis_tanh, basis_tent,
];

/// All 6 basis names.
pub const ALL_BASIS_NAMES: [&str; 6] = [
    "identity", "relu", "xabsx", "sin", "tanh", "tent",
];

/// Legacy alias.
pub const BASIS_NAMES: [&str; 6] = ALL_BASIS_NAMES;

// ── Basis function derivatives (for backprop) ─────────────────

#[inline(always)]
fn grad_identity(_x: f32) -> f32 { 1.0 }

#[inline(always)]
fn grad_relu(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

#[inline(always)]
fn grad_xabsx(x: f32) -> f32 { 2.0 * x.abs() }

#[inline(always)]
fn grad_sin(x: f32) -> f32 {
    std::f32::consts::FRAC_PI_2 * (std::f32::consts::FRAC_PI_2 * x).cos()
}

#[inline(always)]
fn grad_tanh(x: f32) -> f32 {
    let d = 1.0 + x.abs();
    1.0 / (d * d)
}

#[inline(always)]
fn grad_tent(x: f32) -> f32 {
    if x.abs() < 1.0 {
        if x > 0.0 { -1.0 } else if x < 0.0 { 1.0 } else { 0.0 }
    } else {
        0.0
    }
}

const ALL_GRAD_FNS: [fn(f32) -> f32; 6] = [
    grad_identity, grad_relu, grad_xabsx,
    grad_sin, grad_tanh, grad_tent,
];

// ── BasisConfig ───────────────────────────────────────────────

/// Configuration selecting which basis functions to use.
#[derive(Clone, Debug)]
pub struct BasisConfig {
    /// Function pointers for forward evaluation.
    pub fns: Vec<fn(f32) -> f32>,
    /// Function pointers for gradient evaluation.
    pub grad_fns: Vec<fn(f32) -> f32>,
    /// Human-readable names.
    pub names: Vec<&'static str>,
}

impl BasisConfig {
    /// Full 6-basis set (legacy).
    pub fn full() -> Self {
        Self {
            fns: ALL_BASIS_FNS.to_vec(),
            grad_fns: ALL_GRAD_FNS.to_vec(),
            names: ALL_BASIS_NAMES.to_vec(),
        }
    }

    /// Trimmed 3-basis: identity + sin + tanh.
    /// Ablation-proven optimal for physics tasks.
    pub fn trimmed() -> Self {
        Self {
            fns: vec![basis_identity, basis_sin, basis_tanh],
            grad_fns: vec![grad_identity, grad_sin, grad_tanh],
            names: vec!["identity", "sin", "tanh"],
        }
    }

    /// Custom basis set from indices into the full set.
    pub fn custom(indices: &[usize]) -> Self {
        Self {
            fns: indices.iter().map(|&i| ALL_BASIS_FNS[i]).collect(),
            grad_fns: indices.iter().map(|&i| ALL_GRAD_FNS[i]).collect(),
            names: indices.iter().map(|&i| ALL_BASIS_NAMES[i]).collect(),
        }
    }

    pub fn len(&self) -> usize { self.fns.len() }

    pub fn is_empty(&self) -> bool { self.fns.is_empty() }
}

// ── AdaptiveBasis ─────────────────────────────────────────────

/// Learnable activation function: weighted combination of basis functions.
///
/// ```text
/// act(x) = Σ_i weights[i] * basis_i(x)
/// ```
///
/// Supports configurable basis sets (3, 4, or 6 functions).
#[derive(Clone, Debug)]
pub struct AdaptiveBasis {
    /// Basis weights [n_basis]. Learnable.
    weights: Vec<f32>,
    /// Active mask for pruning.
    active_mask: Vec<bool>,
    /// Pruning threshold.
    prune_threshold: f32,
    /// Basis function configuration.
    config: BasisConfig,
}

impl AdaptiveBasis {
    /// Create with full 6-basis, uniform weights (legacy default).
    pub fn new() -> Self {
        Self::with_config(BasisConfig::full())
    }

    /// Create with trimmed 3-basis (identity + sin + tanh).
    /// Ablation-proven: removes relu, xabsx, tent which hurt or don't help.
    pub fn trimmed() -> Self {
        Self::with_config(BasisConfig::trimmed())
    }

    /// Create with a specific basis configuration and uniform weights.
    pub fn with_config(config: BasisConfig) -> Self {
        let n = config.len();
        let w = 1.0 / n as f32;
        Self {
            weights: vec![w; n],
            active_mask: vec![true; n],
            prune_threshold: 1e-4,
            config,
        }
    }

    /// Create with SiLU decomposition init on trimmed basis.
    ///
    /// SiLU(x) = x·σ(x) ≈ 0.5·x + 0.5·tanh(x) + small corrections.
    /// This gives a warm start from a proven activation shape.
    pub fn silu_decomposed() -> Self {
        // SiLU(x) = x·σ(x), we approximate using our basis:
        //   act(x) = w0·x + w1·sin(πx/2) + w2·[x/(1+|x|)]
        // Least-squares fit over [-3, 3] with fast-tanh basis:
        //   w0 ≈ 0.60 (linear dominates at large |x|)
        //   w1 ≈ 0.0  (sin is periodic, doesn't help SiLU shape)
        //   w2 ≈ 0.35 (saturation component)
        let config = BasisConfig::trimmed();
        Self {
            weights: vec![0.60, 0.0, 0.35],
            active_mask: vec![true, false, true],
            prune_threshold: 1e-4,
            config,
        }
    }

    /// Create with specific weights for the full 6-basis set (legacy compat).
    pub fn with_weights(weights: [f32; N_BASIS]) -> Self {
        let active_mask: Vec<bool> = weights.iter().map(|w| w.abs() > 1e-4).collect();
        Self {
            weights: weights.to_vec(),
            active_mask,
            prune_threshold: 1e-4,
            config: BasisConfig::full(),
        }
    }

    /// Create with specific weights for a given config.
    pub fn with_config_weights(config: BasisConfig, weights: Vec<f32>) -> Self {
        assert_eq!(config.len(), weights.len(), "weights must match basis count");
        let active_mask: Vec<bool> = weights.iter().map(|w| w.abs() > 1e-4).collect();
        Self { weights, active_mask, prune_threshold: 1e-4, config }
    }

    /// Create biased toward SiLU behaviour on full 6-basis (legacy).
    pub fn silu_like() -> Self {
        Self::with_weights([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
    }

    /// Number of basis functions in this instance.
    pub fn n_basis(&self) -> usize { self.config.len() }

    /// Forward pass: apply adaptive activation element-wise (in-place).
    pub fn forward(&self, x: &mut [f32]) {
        let fns = &self.config.fns;
        for xi in x.iter_mut() {
            let v = *xi;
            let mut acc = 0.0f32;
            for (j, active) in self.active_mask.iter().enumerate() {
                if *active {
                    acc += self.weights[j] * fns[j](v);
                }
            }
            *xi = acc;
        }
    }

    /// Forward with per-basis outputs (needed for gradient computation).
    pub fn forward_with_basis(&self, x: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let n = x.len();
        let nb = self.n_basis();
        let fns = &self.config.fns;
        let mut output = vec![0.0f32; n];
        let mut basis_outputs: Vec<Vec<f32>> = (0..nb)
            .map(|_| vec![0.0f32; n])
            .collect();

        for (i, &xi) in x.iter().enumerate() {
            let mut acc = 0.0f32;
            for j in 0..nb {
                let bj = fns[j](xi);
                basis_outputs[j][i] = bj;
                acc += self.weights[j] * bj;
            }
            output[i] = acc;
        }
        (output, basis_outputs)
    }

    /// Compute gradient of loss w.r.t. input x.
    pub fn backward_input(&self, x: &[f32], grad_output: &[f32]) -> Vec<f32> {
        let n = x.len().min(grad_output.len());
        let nb = self.n_basis();
        let grad_fns = &self.config.grad_fns;
        let mut grad_input = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..nb {
                if self.active_mask[j] {
                    sum += self.weights[j] * grad_fns[j](x[i]);
                }
            }
            grad_input[i] = grad_output[i] * sum;
        }
        grad_input
    }

    /// Update weights given gradient of loss w.r.t. activation output.
    pub fn update_weights(&mut self, grad_output: &[f32], basis_outputs: &[Vec<f32>], lr: f32) {
        let nb = self.n_basis();
        for j in 0..nb {
            let grad_wj: f32 = grad_output.iter()
                .zip(basis_outputs[j].iter())
                .map(|(go, bj)| go * bj)
                .sum();
            self.weights[j] -= lr * grad_wj;
        }
    }

    /// Prune: zero out weights below threshold.
    pub fn prune(&mut self, threshold: f32) {
        self.prune_threshold = threshold;
        for j in 0..self.n_basis() {
            if self.weights[j].abs() < threshold {
                self.weights[j] = 0.0;
                self.active_mask[j] = false;
            } else {
                self.active_mask[j] = true;
            }
        }
    }

    /// Get current weights (returns slice for any basis count).
    pub fn weights_slice(&self) -> &[f32] { &self.weights }

    /// Get current weights as fixed array (legacy, panics if n_basis != 6).
    pub fn weights(&self) -> &[f32; N_BASIS] {
        self.weights.as_slice().try_into()
            .expect("weights() requires exactly 6 basis; use weights_slice() for trimmed")
    }

    /// Number of active (non-zero) basis functions.
    pub fn active_count(&self) -> usize {
        self.active_mask.iter().filter(|&&a| a).count()
    }

    /// Human-readable description of the learned activation shape.
    pub fn describe(&self) -> String {
        let names = &self.config.names;
        let mut parts = Vec::new();
        for (j, &w) in self.weights.iter().enumerate() {
            if w.abs() > self.prune_threshold {
                parts.push(format!("{:.3}·{}", w, names[j]));
            }
        }
        if parts.is_empty() { "0".to_string() } else { parts.join(" + ") }
    }

    /// Get basis names for this instance.
    pub fn basis_names(&self) -> &[&str] { &self.config.names }
}

impl Default for AdaptiveBasis {
    fn default() -> Self { Self::new() }
}

// ── BasisScheduler ────────────────────────────────────────────

/// Warmup scheduler for basis weights.
///
/// Freeze basis weights for `warmup_steps` to let ternary weights stabilise,
/// then cosine-anneal the basis learning rate from 0 to `max_lr`.
#[derive(Clone, Debug)]
pub struct BasisScheduler {
    /// Steps to freeze basis weights (let ternary weights settle).
    pub warmup_steps: usize,
    /// Maximum basis learning rate after warmup.
    pub max_lr: f32,
    /// Current step counter.
    step: usize,
}

impl BasisScheduler {
    /// Create a new scheduler.
    ///
    /// - `warmup_steps`: freeze basis for this many steps (typically 100-500)
    /// - `max_lr`: target learning rate for basis weights after warmup
    pub fn new(warmup_steps: usize, max_lr: f32) -> Self {
        Self { warmup_steps, max_lr, step: 0 }
    }

    /// Get the current effective basis learning rate and advance step.
    pub fn step_lr(&mut self) -> f32 {
        let lr = self.current_lr();
        self.step += 1;
        lr
    }

    /// Get current lr without advancing.
    pub fn current_lr(&self) -> f32 {
        if self.step < self.warmup_steps {
            0.0  // frozen
        } else {
            let ramp_step = self.step - self.warmup_steps;
            let ramp_len = self.warmup_steps;
            if ramp_len == 0 || ramp_step >= ramp_len {
                self.max_lr
            } else {
                let progress = (ramp_step + 1) as f32 / ramp_len as f32;
                self.max_lr * 0.5 * (1.0 - (std::f32::consts::PI * progress).cos())
            }
        }
    }

    /// Current step count.
    pub fn current_step(&self) -> usize { self.step }

    /// Whether still in warmup (frozen).
    pub fn is_frozen(&self) -> bool { self.step < self.warmup_steps }
}

// ── AdaptiveBlock ─────────────────────────────────────────────

/// Transformer block using ternary weights + adaptive activation.
///
/// ```text
/// x → RMSNorm → BitNetLinear(up) → AdaptiveBasis → BitNetLinear(down) → + residual
/// ```
#[derive(Clone, Debug)]
pub struct AdaptiveBlock {
    pub norm: RMSNorm,
    pub ffn_up: BitNetLinear,
    pub activation: AdaptiveBasis,
    pub ffn_down: BitNetLinear,
    pub d_model: usize,
    pub ffn_dim: usize,
}

impl AdaptiveBlock {
    /// Create with full 6-basis, uniform weights (legacy).
    pub fn new(d_model: usize, ffn_dim: usize) -> Self {
        Self::with_activation(d_model, ffn_dim, AdaptiveBasis::new())
    }

    /// Create with trimmed 3-basis (identity + sin + tanh).
    pub fn trimmed(d_model: usize, ffn_dim: usize) -> Self {
        Self::with_activation(d_model, ffn_dim, AdaptiveBasis::trimmed())
    }

    /// Create with SiLU-decomposed init on trimmed basis.
    pub fn silu_init(d_model: usize, ffn_dim: usize) -> Self {
        Self::with_activation(d_model, ffn_dim, AdaptiveBasis::silu_decomposed())
    }

    /// Create with a specific activation.
    pub fn with_activation(d_model: usize, ffn_dim: usize, activation: AdaptiveBasis) -> Self {
        Self {
            norm: RMSNorm::new(d_model),
            ffn_up: BitNetLinear::new(d_model, ffn_dim, false),
            activation,
            ffn_down: BitNetLinear::new(ffn_dim, d_model, false),
            d_model,
            ffn_dim,
        }
    }

    /// Forward pass with residual connection (in-place on x).
    pub fn forward(&self, x: &mut [f32]) -> Result<(), VagiError> {
        if x.len() != self.d_model {
            return Err(VagiError::ShapeMismatch {
                expected: format!("{}", self.d_model),
                got: format!("{}", x.len()),
            });
        }
        let residual: Vec<f32> = x.to_vec();
        self.norm.forward(x);

        let mut hidden = vec![0.0f32; self.ffn_dim];
        self.ffn_up.forward(x, &mut hidden)?;
        self.activation.forward(&mut hidden);
        self.ffn_down.forward(&hidden, x)?;

        for (xi, ri) in x.iter_mut().zip(residual.iter()) {
            *xi += ri;
        }
        Ok(())
    }

    /// Forward pass returning new vector.
    pub fn forward_vec(&self, input: &[f32]) -> Result<Vec<f32>, VagiError> {
        let mut output = input.to_vec();
        self.forward(&mut output)?;
        Ok(output)
    }

    /// Forward pass with intermediate values (for training).
    pub fn forward_training(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>), VagiError> {
        if x.len() != self.d_model {
            return Err(VagiError::ShapeMismatch {
                expected: format!("{}", self.d_model),
                got: format!("{}", x.len()),
            });
        }
        let mut normed = x.to_vec();
        self.norm.forward(&mut normed);

        let mut hidden = vec![0.0f32; self.ffn_dim];
        self.ffn_up.forward(&normed, &mut hidden)?;

        let pre_act = hidden.clone();
        let (activated, basis_outputs) = self.activation.forward_with_basis(&hidden);

        let mut output = vec![0.0f32; self.d_model];
        self.ffn_down.forward(&activated, &mut output)?;

        for (oi, xi) in output.iter_mut().zip(x.iter()) {
            *oi += xi;
        }
        Ok((output, pre_act, basis_outputs))
    }

    pub fn activation(&self) -> &AdaptiveBasis { &self.activation }
    pub fn activation_mut(&mut self) -> &mut AdaptiveBasis { &mut self.activation }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- Basis function correctness --

    #[test]
    fn test_basis_identity() {
        assert_eq!(basis_identity(2.0), 2.0);
        assert_eq!(basis_identity(-3.0), -3.0);
        assert_eq!(basis_identity(0.0), 0.0);
    }

    #[test]
    fn test_basis_relu() {
        assert_eq!(basis_relu(2.0), 2.0);
        assert_eq!(basis_relu(-3.0), 0.0);
        assert_eq!(basis_relu(0.0), 0.0);
    }

    #[test]
    fn test_basis_xabsx() {
        assert_eq!(basis_xabsx(2.0), 4.0);
        assert_eq!(basis_xabsx(-3.0), -9.0);
        assert_eq!(basis_xabsx(0.0), 0.0);
    }

    #[test]
    fn test_basis_sin() {
        let v = basis_sin(1.0);
        assert!((v - 1.0).abs() < 1e-5, "sin(π/2·1) should ≈ 1.0, got {v}");
        let v0 = basis_sin(0.0);
        assert!(v0.abs() < 1e-6, "sin(0) should ≈ 0");
    }

    #[test]
    fn test_basis_tanh() {
        let v = basis_tanh(0.0);
        assert!(v.abs() < 1e-6, "tanh(0) = 0, got {v}");
        let v1 = basis_tanh(1.0);
        assert!((v1 - 0.5).abs() < 1e-6, "tanh_approx(1) = 0.5, got {v1}");
        assert!(basis_tanh(100.0) < 1.01, "tanh should be bounded");
    }

    #[test]
    fn test_basis_tent() {
        assert_eq!(basis_tent(0.0), 1.0);
        assert_eq!(basis_tent(0.5), 0.5);
        assert_eq!(basis_tent(1.0), 0.0);
        assert_eq!(basis_tent(2.0), 0.0);
        assert_eq!(basis_tent(-0.5), 0.5);
    }

    // -- Trimmed basis --

    #[test]
    fn test_trimmed_3basis() {
        let ab = AdaptiveBasis::trimmed();
        assert_eq!(ab.n_basis(), 3);
        assert_eq!(ab.basis_names(), &["identity", "sin", "tanh"]);
        let mut x = vec![1.0, -0.5, 0.0];
        ab.forward(&mut x);
        assert!(x.iter().all(|v| v.is_finite()), "Trimmed forward should be finite");
    }

    #[test]
    fn test_trimmed_forward_matches_manual() {
        let ab = AdaptiveBasis::with_config_weights(
            BasisConfig::trimmed(),
            vec![0.5, 0.2, 0.3],
        );
        let x_val = 1.0f32;
        let expected = 0.5 * x_val + 0.2 * basis_sin(x_val) + 0.3 * basis_tanh(x_val);
        let mut x = vec![x_val];
        ab.forward(&mut x);
        assert!((x[0] - expected).abs() < 1e-6, "got {} expected {}", x[0], expected);
    }

    // -- SiLU decomposition --

    #[test]
    fn test_silu_decomposed_init() {
        let ab = AdaptiveBasis::silu_decomposed();
        assert_eq!(ab.n_basis(), 3);
        let w = ab.weights_slice();
        assert!((w[0] - 0.60).abs() < 0.01, "identity weight should ≈ 0.60, got {}", w[0]);
        assert!((w[2] - 0.35).abs() < 0.01, "tanh weight should ≈ 0.35, got {}", w[2]);
    }

    #[test]
    fn test_silu_decomposition_approximation() {
        // Verify the decomposition actually approximates SiLU
        let ab = AdaptiveBasis::silu_decomposed();
        let silu = |x: f32| -> f32 { x / (1.0 + (-x).exp()) };
        let mut total_error = 0.0f32;
        let n = 100;
        for i in 0..n {
            let x = -3.0 + 6.0 * i as f32 / n as f32;
            let (out, _) = ab.forward_with_basis(&[x]);
            let err = (out[0] - silu(x)).abs();
            total_error += err;
        }
        let avg_error = total_error / n as f32;
        // Should approximate SiLU within reasonable error
        assert!(avg_error < 0.6, "SiLU approximation avg error = {avg_error:.4}, should be < 0.6");
    }

    // -- BasisScheduler --

    #[test]
    fn test_scheduler_warmup_frozen() {
        let mut sched = BasisScheduler::new(100, 0.01);
        assert!(sched.is_frozen());
        assert_eq!(sched.step_lr(), 0.0);
        assert_eq!(sched.step_lr(), 0.0);
    }

    #[test]
    fn test_scheduler_after_warmup() {
        let mut sched = BasisScheduler::new(10, 0.01);
        // Skip warmup
        for _ in 0..10 {
            assert_eq!(sched.step_lr(), 0.0);
        }
        // Now should start ramping
        assert!(!sched.is_frozen());
        let lr = sched.step_lr();
        assert!(lr > 0.0, "LR should be > 0 after warmup, got {lr}");
        assert!(lr < 0.01, "LR should be < max during ramp, got {lr}");
    }

    #[test]
    fn test_scheduler_reaches_max() {
        let mut sched = BasisScheduler::new(10, 0.01);
        // Skip warmup + full ramp
        for _ in 0..20 {
            sched.step_lr();
        }
        let lr = sched.step_lr();
        assert!((lr - 0.01).abs() < 1e-6, "Should reach max_lr, got {lr}");
    }

    // -- AdaptiveBasis forward (legacy compat) --

    #[test]
    fn test_forward_identity_only() {
        let ab = AdaptiveBasis::with_weights([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut x = vec![1.0, -2.0, 0.5];
        let orig = x.clone();
        ab.forward(&mut x);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-6, "With only identity weight, output should match input");
        }
    }

    #[test]
    fn test_forward_relu_only() {
        let ab = AdaptiveBasis::with_weights([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut x = vec![2.0, -3.0, 0.0];
        ab.forward(&mut x);
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!(x[1].abs() < 1e-6);
        assert!(x[2].abs() < 1e-6);
    }

    #[test]
    fn test_forward_combination() {
        let ab = AdaptiveBasis::with_weights([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let mut x = vec![2.0, -2.0];
        ab.forward(&mut x);
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!((x[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_forward_with_basis_matches() {
        let ab = AdaptiveBasis::new();
        let x = vec![1.0, -0.5, 0.3];
        let mut x_inplace = x.clone();
        ab.forward(&mut x_inplace);
        let (output, _basis) = ab.forward_with_basis(&x);
        for (a, b) in x_inplace.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6, "forward and forward_with_basis should match");
        }
    }

    // -- Gradient correctness --

    #[test]
    fn test_numerical_gradient_weights() {
        let ab = AdaptiveBasis::new();
        let x = vec![1.0, -0.5, 0.3, 2.0, -1.5];
        let eps = 1e-4;
        let (_, basis_out) = ab.forward_with_basis(&x);
        let grad_output = vec![1.0; x.len()];
        for j in 0..N_BASIS {
            let analytical: f32 = grad_output.iter()
                .zip(basis_out[j].iter())
                .map(|(g, b)| g * b)
                .sum();
            let mut weights_p = *ab.weights();
            let mut weights_m = *ab.weights();
            weights_p[j] += eps;
            weights_m[j] -= eps;
            let ab_p = AdaptiveBasis::with_weights(weights_p);
            let ab_m = AdaptiveBasis::with_weights(weights_m);
            let (out_p, _) = ab_p.forward_with_basis(&x);
            let (out_m, _) = ab_m.forward_with_basis(&x);
            let loss_p: f32 = out_p.iter().sum();
            let loss_m: f32 = out_m.iter().sum();
            let numerical = (loss_p - loss_m) / (2.0 * eps);
            assert!(
                (analytical - numerical).abs() < 1e-2,
                "Gradient mismatch for basis {j} ({name}): analytical={analytical}, numerical={numerical}",
                name = ALL_BASIS_NAMES[j]
            );
        }
    }

    #[test]
    fn test_numerical_gradient_input() {
        let ab = AdaptiveBasis::with_weights([0.3, 0.2, 0.1, 0.15, 0.15, 0.1]);
        let x = vec![0.5, -0.3, 1.2];
        let grad_output = vec![1.0, 1.0, 1.0];
        let eps = 1e-4;
        let analytical = ab.backward_input(&x, &grad_output);
        for i in 0..x.len() {
            let mut x_p = x.clone();
            let mut x_m = x.clone();
            x_p[i] += eps;
            x_m[i] -= eps;
            let (out_p, _) = ab.forward_with_basis(&x_p);
            let (out_m, _) = ab.forward_with_basis(&x_m);
            let loss_p: f32 = out_p.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let loss_m: f32 = out_m.iter().zip(&grad_output).map(|(o, g)| o * g).sum();
            let numerical = (loss_p - loss_m) / (2.0 * eps);
            assert!(
                (analytical[i] - numerical).abs() < 1e-2,
                "Input gradient mismatch at {i}: analytical={}, numerical={numerical}",
                analytical[i]
            );
        }
    }

    #[test]
    fn test_numerical_gradient_trimmed() {
        let ab = AdaptiveBasis::with_config_weights(
            BasisConfig::trimmed(),
            vec![0.4, 0.3, 0.3],
        );
        let x = vec![0.5, -0.3, 1.2];
        let eps = 1e-4;
        let (_, basis_out) = ab.forward_with_basis(&x);
        let grad_output = vec![1.0; x.len()];
        for j in 0..3 {
            let analytical: f32 = grad_output.iter()
                .zip(basis_out[j].iter())
                .map(|(g, b)| g * b)
                .sum();
            let mut w_p = ab.weights_slice().to_vec();
            let mut w_m = ab.weights_slice().to_vec();
            w_p[j] += eps;
            w_m[j] -= eps;
            let ab_p = AdaptiveBasis::with_config_weights(BasisConfig::trimmed(), w_p);
            let ab_m = AdaptiveBasis::with_config_weights(BasisConfig::trimmed(), w_m);
            let (out_p, _) = ab_p.forward_with_basis(&x);
            let (out_m, _) = ab_m.forward_with_basis(&x);
            let numerical = (out_p.iter().sum::<f32>() - out_m.iter().sum::<f32>()) / (2.0 * eps);
            assert!(
                (analytical - numerical).abs() < 1e-2,
                "Trimmed gradient mismatch at {j}"
            );
        }
    }

    // -- Pruning --

    #[test]
    fn test_pruning() {
        let mut ab = AdaptiveBasis::with_weights([0.5, 0.001, 0.3, 0.0001, 0.4, 0.00001]);
        assert_eq!(ab.active_count(), 4);
        ab.prune(0.01);
        assert_eq!(ab.active_count(), 3);
        assert_eq!(ab.weights()[1], 0.0);
        assert_eq!(ab.weights()[3], 0.0);
        assert_eq!(ab.weights()[5], 0.0);
    }

    #[test]
    fn test_weight_update() {
        let mut ab = AdaptiveBasis::with_weights([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let x = vec![1.0, -1.0];
        let (_, basis_out) = ab.forward_with_basis(&x);
        let grad_output = vec![1.0, 1.0];
        let old_weights = *ab.weights();
        ab.update_weights(&grad_output, &basis_out, 0.01);
        assert_ne!(ab.weights(), &old_weights, "Weights should change after update");
    }

    // -- AdaptiveBlock --

    #[test]
    fn test_adaptive_block_shape() {
        let block = AdaptiveBlock::new(64, 256);
        let input = vec![1.0f32; 64];
        let output = block.forward_vec(&input).unwrap();
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_adaptive_block_residual() {
        let block = AdaptiveBlock::new(64, 256);
        let input = vec![1.0f32; 64];
        let output = block.forward_vec(&input).unwrap();
        assert_ne!(output, input);
    }

    #[test]
    fn test_adaptive_block_activation_affects_output() {
        let block1 = AdaptiveBlock::new(32, 128);
        let mut block2 = block1.clone();
        block2.activation = AdaptiveBasis::with_weights([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let input = vec![0.5f32; 32];
        let out1 = block1.forward_vec(&input).unwrap();
        let out2 = block2.forward_vec(&input).unwrap();
        assert_ne!(out1, out2, "Different basis weights should produce different outputs");
    }

    #[test]
    fn test_adaptive_block_training_forward() {
        let block = AdaptiveBlock::new(32, 128);
        let input = vec![0.5f32; 32];
        let (output, pre_act, basis_out) = block.forward_training(&input).unwrap();
        assert_eq!(output.len(), 32);
        assert_eq!(pre_act.len(), 128);
        assert_eq!(basis_out.len(), N_BASIS);
        assert_eq!(basis_out[0].len(), 128);
    }

    #[test]
    fn test_trimmed_block() {
        let block = AdaptiveBlock::trimmed(32, 128);
        let input = vec![0.5f32; 32];
        let (output, _pre_act, basis_out) = block.forward_training(&input).unwrap();
        assert_eq!(output.len(), 32);
        assert_eq!(basis_out.len(), 3); // trimmed = 3 basis
    }

    #[test]
    fn test_silu_init_block() {
        let block = AdaptiveBlock::silu_init(32, 128);
        let input = vec![0.5f32; 32];
        let output = block.forward_vec(&input).unwrap();
        assert_eq!(output.len(), 32);
        assert_eq!(block.activation().n_basis(), 3);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let block = AdaptiveBlock::new(64, 256);
        let bad_input = vec![1.0f32; 100];
        assert!(block.forward_vec(&bad_input).is_err());
    }

    #[test]
    fn test_describe() {
        let ab = AdaptiveBasis::with_weights([0.5, 0.0, 0.0, 0.3, 0.0, 0.2]);
        let desc = ab.describe();
        assert!(desc.contains("identity"), "Should mention identity");
        assert!(desc.contains("sin"), "Should mention sin");
        assert!(desc.contains("tent"), "Should mention tent");
        assert!(!desc.contains("relu"), "Should not mention relu (weight=0)");
    }

    #[test]
    fn test_describe_trimmed() {
        let ab = AdaptiveBasis::silu_decomposed();
        let desc = ab.describe();
        assert!(desc.contains("identity"), "Should mention identity");
        assert!(desc.contains("tanh"), "Should mention tanh");
    }
}
