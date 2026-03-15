//! AdaptiveNet — learnable activation functions as basis combinations.
//!
//! Replaces fixed SiLU in BitNetBlock with a weighted sum of 6 basis functions:
//!   act(x) = w0·x + w1·ReLU(x) + w2·x|x| + w3·sin(πx/2) + w4·tanh(x) + w5·tent(x)
//!
//! Matmul remains ternary (BitNetLinear, unchanged). Only the activation is learnable.
//! Overhead: 6 multiply-adds per element vs the O(n²) matmul → ~0.8% slower than BitNet.

use crate::bitnet::{BitNetLinear, RMSNorm};
use crate::error::VagiError;

/// Number of basis functions.
pub const N_BASIS: usize = 6;

/// Basis function names (for display/logging).
pub const BASIS_NAMES: [&str; N_BASIS] = [
    "identity", "relu", "xabsx", "sin", "tanh", "tent",
];

// ── Basis function evaluations ────────────────────────────────

/// b0: identity — allows skip-like behaviour.
#[inline(always)]
pub fn basis_identity(x: f32) -> f32 { x }

/// b1: ReLU — standard rectified nonlinearity.
#[inline(always)]
pub fn basis_relu(x: f32) -> f32 { x.max(0.0) }

/// b2: x·|x| — smooth quadratic, sign-preserving. Good for physics.
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

/// All basis functions as an array.
pub const BASIS_FNS: [fn(f32) -> f32; N_BASIS] = [
    basis_identity, basis_relu, basis_xabsx,
    basis_sin, basis_tanh, basis_tent,
];

// ── Basis function derivatives (for backprop) ─────────────────

/// d/dx[identity] = 1
#[inline(always)]
fn grad_identity(_x: f32) -> f32 { 1.0 }

/// d/dx[ReLU] = 1 if x > 0, else 0
#[inline(always)]
fn grad_relu(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

/// d/dx[x|x|] = 2|x|
#[inline(always)]
fn grad_xabsx(x: f32) -> f32 { 2.0 * x.abs() }

/// d/dx[sin(πx/2)] = (π/2)·cos(πx/2)
#[inline(always)]
fn grad_sin(x: f32) -> f32 {
    std::f32::consts::FRAC_PI_2 * (std::f32::consts::FRAC_PI_2 * x).cos()
}

/// d/dx[tanh_approx] = 1 / (1+|x|)²
#[inline(always)]
fn grad_tanh(x: f32) -> f32 {
    let d = 1.0 + x.abs();
    1.0 / (d * d)
}

/// d/dx[tent] = -sign(x) if |x| < 1, else 0
#[inline(always)]
fn grad_tent(x: f32) -> f32 {
    if x.abs() < 1.0 {
        if x > 0.0 { -1.0 } else if x < 0.0 { 1.0 } else { 0.0 }
    } else {
        0.0
    }
}

const GRAD_FNS: [fn(f32) -> f32; N_BASIS] = [
    grad_identity, grad_relu, grad_xabsx,
    grad_sin, grad_tanh, grad_tent,
];

// ── AdaptiveBasis ─────────────────────────────────────────────

/// Learnable activation function: weighted combination of basis functions.
///
/// ```text
/// act(x) = Σ_i weights[i] * basis_i(x)
/// ```
///
/// Basis functions are fixed. Weights are learnable per-layer (6 floats).
#[derive(Clone, Debug)]
pub struct AdaptiveBasis {
    /// Basis weights [N_BASIS]. Learnable.
    weights: [f32; N_BASIS],
    /// Active mask: skip basis with weight ≈ 0 for speed.
    active_mask: [bool; N_BASIS],
    /// Pruning threshold.
    prune_threshold: f32,
}

impl AdaptiveBasis {
    /// Create with uniform weights (all basis equally active).
    pub fn new() -> Self {
        // Start with SiLU-like: bias toward tanh (closest to sigmoid-gated linear)
        let w = 1.0 / N_BASIS as f32;
        Self {
            weights: [w; N_BASIS],
            active_mask: [true; N_BASIS],
            prune_threshold: 1e-4,
        }
    }

    /// Create biased toward SiLU behaviour (identity + tanh dominant).
    pub fn silu_like() -> Self {
        Self {
            weights: [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            active_mask: [true, false, false, false, true, false],
            prune_threshold: 1e-4,
        }
    }

    /// Create with specific initial weights.
    pub fn with_weights(weights: [f32; N_BASIS]) -> Self {
        let active_mask = weights.map(|w| w.abs() > 1e-4);
        Self { weights, active_mask, prune_threshold: 1e-4 }
    }

    /// Forward pass: apply adaptive activation element-wise (in-place).
    pub fn forward(&self, x: &mut [f32]) {
        for xi in x.iter_mut() {
            let v = *xi;
            let mut acc = 0.0f32;
            for (j, active) in self.active_mask.iter().enumerate() {
                if *active {
                    acc += self.weights[j] * BASIS_FNS[j](v);
                }
            }
            *xi = acc;
        }
    }

    /// Forward with per-basis outputs (needed for gradient computation).
    ///
    /// Returns `(output, basis_outputs)` where `basis_outputs[j][i] = basis_j(x[i])`.
    pub fn forward_with_basis(&self, x: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let n = x.len();
        let mut output = vec![0.0f32; n];
        let mut basis_outputs: Vec<Vec<f32>> = (0..N_BASIS)
            .map(|_| vec![0.0f32; n])
            .collect();

        for (i, &xi) in x.iter().enumerate() {
            let mut acc = 0.0f32;
            for j in 0..N_BASIS {
                let bj = BASIS_FNS[j](xi);
                basis_outputs[j][i] = bj;
                acc += self.weights[j] * bj;
            }
            output[i] = acc;
        }
        (output, basis_outputs)
    }

    /// Compute gradient of loss w.r.t. input x.
    ///
    /// `∂L/∂x_i = (∂L/∂act_i) × Σ_j w_j × ∂basis_j(x_i)/∂x_i`
    pub fn backward_input(&self, x: &[f32], grad_output: &[f32]) -> Vec<f32> {
        let n = x.len().min(grad_output.len());
        let mut grad_input = vec![0.0f32; n];
        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..N_BASIS {
                if self.active_mask[j] {
                    sum += self.weights[j] * GRAD_FNS[j](x[i]);
                }
            }
            grad_input[i] = grad_output[i] * sum;
        }
        grad_input
    }

    /// Update weights given gradient of loss w.r.t. activation output.
    ///
    /// `∂L/∂w_j = Σ_i (∂L/∂act_i) × basis_j(x_i)`
    pub fn update_weights(&mut self, grad_output: &[f32], basis_outputs: &[Vec<f32>], lr: f32) {
        for j in 0..N_BASIS {
            let grad_wj: f32 = grad_output.iter()
                .zip(basis_outputs[j].iter())
                .map(|(go, bj)| go * bj)
                .sum();
            self.weights[j] -= lr * grad_wj;
        }
    }

    /// Prune: zero out weights below threshold, update active_mask.
    pub fn prune(&mut self, threshold: f32) {
        self.prune_threshold = threshold;
        for j in 0..N_BASIS {
            if self.weights[j].abs() < threshold {
                self.weights[j] = 0.0;
                self.active_mask[j] = false;
            } else {
                self.active_mask[j] = true;
            }
        }
    }

    /// Get current weights.
    pub fn weights(&self) -> &[f32; N_BASIS] { &self.weights }

    /// Number of active (non-zero) basis functions.
    pub fn active_count(&self) -> usize {
        self.active_mask.iter().filter(|&&a| a).count()
    }

    /// Human-readable description of the learned activation shape.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        for (j, &w) in self.weights.iter().enumerate() {
            if w.abs() > self.prune_threshold {
                parts.push(format!("{:.3}·{}", w, BASIS_NAMES[j]));
            }
        }
        if parts.is_empty() { "0".to_string() } else { parts.join(" + ") }
    }
}

impl Default for AdaptiveBasis {
    fn default() -> Self { Self::new() }
}

// ── AdaptiveBlock ─────────────────────────────────────────────

/// Transformer block using ternary weights + adaptive activation.
///
/// ```text
/// x → RMSNorm → BitNetLinear(up) → AdaptiveBasis → BitNetLinear(down) → + residual
/// ```
///
/// Identical to BitNetBlock except activation is learnable.
/// Matmul overhead: <1% (6 mul-adds per element vs O(n²) matmul).
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
    /// Create a new AdaptiveBlock.
    pub fn new(d_model: usize, ffn_dim: usize) -> Self {
        Self {
            norm: RMSNorm::new(d_model),
            ffn_up: BitNetLinear::new(d_model, ffn_dim, false),
            activation: AdaptiveBasis::new(),
            ffn_down: BitNetLinear::new(ffn_dim, d_model, false),
            d_model,
            ffn_dim,
        }
    }

    /// Create with SiLU-like initial activation (for warm start).
    pub fn silu_init(d_model: usize, ffn_dim: usize) -> Self {
        let mut block = Self::new(d_model, ffn_dim);
        block.activation = AdaptiveBasis::silu_like();
        block
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

        // RMSNorm
        self.norm.forward(x);

        // FFN up: d_model → ffn_dim
        let mut hidden = vec![0.0f32; self.ffn_dim];
        self.ffn_up.forward(x, &mut hidden)?;

        // Adaptive activation (replaces SiLU)
        self.activation.forward(&mut hidden);

        // FFN down: ffn_dim → d_model
        self.ffn_down.forward(&hidden, x)?;

        // Residual
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
    /// Returns (output, pre_activation_hidden, basis_outputs).
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

        // Residual
        for (oi, xi) in output.iter_mut().zip(x.iter()) {
            *oi += xi;
        }

        Ok((output, pre_act, basis_outputs))
    }

    /// Get a reference to the adaptive activation.
    pub fn activation(&self) -> &AdaptiveBasis { &self.activation }

    /// Get a mutable reference to the adaptive activation.
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
        assert_eq!(basis_xabsx(2.0), 4.0);    // 2*|2| = 4
        assert_eq!(basis_xabsx(-3.0), -9.0);  // -3*|-3| = -9
        assert_eq!(basis_xabsx(0.0), 0.0);
    }

    #[test]
    fn test_basis_sin() {
        let v = basis_sin(1.0); // sin(π/2) = 1
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

    // -- AdaptiveBasis forward --

    #[test]
    fn test_forward_identity_only() {
        let ab = AdaptiveBasis::with_weights([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut x = vec![1.0, -2.0, 0.5];
        let orig = x.clone();
        ab.forward(&mut x);
        // Should be identity
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
        assert!(x[1].abs() < 1e-6);     // ReLU(-3) = 0
        assert!(x[2].abs() < 1e-6);     // ReLU(0) = 0
    }

    #[test]
    fn test_forward_combination() {
        // 0.5·identity + 0.5·relu
        let ab = AdaptiveBasis::with_weights([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let mut x = vec![2.0, -2.0];
        ab.forward(&mut x);
        // x=2:  0.5*2 + 0.5*2 = 2.0
        assert!((x[0] - 2.0).abs() < 1e-6);
        // x=-2: 0.5*(-2) + 0.5*0 = -1.0
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

        // Compute analytical gradient for w0
        let (_, basis_out) = ab.forward_with_basis(&x);
        let grad_output = vec![1.0; x.len()]; // dL/dact = 1 (simplest case)
        // ∂L/∂w_j = Σ_i grad_output[i] * basis_j(x[i])
        for j in 0..N_BASIS {
            let analytical: f32 = grad_output.iter()
                .zip(basis_out[j].iter())
                .map(|(g, b)| g * b)
                .sum();

            // Numerical: (L(w+ε) - L(w-ε)) / 2ε
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
                name = BASIS_NAMES[j]
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

    // -- Pruning --

    #[test]
    fn test_pruning() {
        let mut ab = AdaptiveBasis::with_weights([0.5, 0.001, 0.3, 0.0001, 0.4, 0.00001]);
        assert_eq!(ab.active_count(), 4); // 0.001 and 0.0001 > 1e-4 threshold
        ab.prune(0.01);
        assert_eq!(ab.active_count(), 3); // Only 0.5, 0.3, 0.4 survive
        assert_eq!(ab.weights()[1], 0.0);
        assert_eq!(ab.weights()[3], 0.0);
        assert_eq!(ab.weights()[5], 0.0);
    }

    // -- Weight update --

    #[test]
    fn test_weight_update() {
        let mut ab = AdaptiveBasis::with_weights([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let x = vec![1.0, -1.0];
        let (_, basis_out) = ab.forward_with_basis(&x);
        let grad_output = vec![1.0, 1.0];
        let old_weights = *ab.weights();
        ab.update_weights(&grad_output, &basis_out, 0.01);
        // Weights should have changed
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
        // Output should differ from input (FFN processes it)
        assert_ne!(output, input);
    }

    #[test]
    fn test_adaptive_block_activation_affects_output() {
        let mut block1 = AdaptiveBlock::new(32, 128);
        let mut block2 = block1.clone();
        // Change block2's activation to pure identity
        block2.activation = AdaptiveBasis::with_weights([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let input = vec![0.5f32; 32];
        let out1 = block1.forward_vec(&input).unwrap();
        let out2 = block2.forward_vec(&input).unwrap();
        // Different activations should give different outputs
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
    fn test_shape_mismatch_error() {
        let block = AdaptiveBlock::new(64, 256);
        let bad_input = vec![1.0f32; 100];
        assert!(block.forward_vec(&bad_input).is_err());
    }

    // -- Describe --

    #[test]
    fn test_describe() {
        let ab = AdaptiveBasis::with_weights([0.5, 0.0, 0.0, 0.3, 0.0, 0.2]);
        let desc = ab.describe();
        assert!(desc.contains("identity"), "Should mention identity");
        assert!(desc.contains("sin"), "Should mention sin");
        assert!(desc.contains("tent"), "Should mention tent");
        assert!(!desc.contains("relu"), "Should not mention relu (weight=0)");
    }
}
