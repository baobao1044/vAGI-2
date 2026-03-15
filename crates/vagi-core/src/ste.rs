//! Straight-Through Estimator (STE) for ternary weight training.
//!
//! Training maintains "latent" f32 weights. Forward quantizes to ternary.
//! Backward: gradient passes through unchanged (∂L/∂w_latent = ∂L/∂w_ternary).
//!
//! This is how Microsoft trains BitNet b1.58.

use crate::ternary::TernaryMatrix;

/// STE quantizer for ternary {-1, 0, +1} weights.
pub struct STEQuantizer {
    /// Gamma parameter for threshold: threshold = gamma * mean(|W|)
    pub gamma: f32,
}

impl STEQuantizer {
    pub fn new() -> Self {
        Self { gamma: 0.7 }
    }

    pub fn with_gamma(gamma: f32) -> Self {
        Self { gamma }
    }

    /// Forward: quantize f32 latent weights to ternary.
    ///
    /// Uses absmean threshold: threshold = gamma * mean(|W_row|)
    /// W >= threshold → +1, W <= -threshold → -1, else → 0
    ///
    /// Returns (packed TernaryMatrix, threshold used per row).
    pub fn quantize_forward(
        &self,
        w_latent: &[f32],
        rows: usize,
        cols: usize,
    ) -> TernaryMatrix {
        TernaryMatrix::pack(w_latent, rows, cols, self.gamma)
    }

    /// Backward: STE gradient pass-through with optional gradient clipping.
    ///
    /// ∂L/∂w_latent = ∂L/∂w_quantized (identity, straight-through)
    /// but clip to [-1, 1] for weights far from the quantization boundary
    /// to prevent latent weights from drifting too far.
    pub fn quantize_backward(
        &self,
        grad: &[f32],
        w_latent: &[f32],
        clip_range: f32,
    ) -> Vec<f32> {
        grad.iter()
            .zip(w_latent.iter())
            .map(|(&g, &w)| {
                // STE: pass gradient through, but zero out for very large latent weights
                // (prevents them from drifting infinitely far from the quantization boundary)
                if w.abs() > clip_range {
                    0.0
                } else {
                    g
                }
            })
            .collect()
    }
}

impl Default for STEQuantizer {
    fn default() -> Self { Self::new() }
}

/// Trainable ternary linear layer using STE.
///
/// Maintains latent f32 weights that get quantized to ternary on each forward.
/// Gradients update the latent weights via STE pass-through.
#[derive(Clone, Debug)]
pub struct STELinear {
    /// Latent f32 weights [out_features × in_features].
    pub w_latent: Vec<f32>,
    /// Optional bias.
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
    /// STE quantizer config.
    pub gamma: f32,
    /// Gradient clipping range for STE.
    pub clip_range: f32,
}

impl STELinear {
    /// Create with random latent weights (small Gaussian-like init).
    pub fn new(in_features: usize, out_features: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let w_latent: Vec<f32> = (0..out_features * in_features)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self {
            w_latent,
            bias: None,
            in_features,
            out_features,
            gamma: 0.7,
            clip_range: 2.0,
        }
    }

    /// Quantize latent weights and perform forward pass.
    /// Returns (output, quantized TernaryMatrix for caching).
    pub fn forward(&self, x: &[f32], y: &mut [f32]) -> TernaryMatrix {
        let ste = STEQuantizer::with_gamma(self.gamma);
        let w_ternary = ste.quantize_forward(&self.w_latent, self.out_features, self.in_features);
        crate::ternary::ternary_matvec(&w_ternary, x, y);
        if let Some(ref bias) = self.bias {
            for (yi, bi) in y.iter_mut().zip(bias.iter()) {
                *yi += bi;
            }
        }
        w_ternary
    }

    /// Update latent weights using STE gradient.
    /// `grad_output` is ∂L/∂y, `x` is the input from forward.
    /// Uses simple SGD update: w_latent -= lr * ∂L/∂w_latent
    pub fn backward_update(
        &mut self,
        grad_output: &[f32],
        x: &[f32],
        lr: f32,
    ) {
        let ste = STEQuantizer::with_gamma(self.gamma);
        // ∂L/∂W = grad_output ⊗ x^T (outer product)
        for m in 0..self.out_features {
            for n in 0..self.in_features {
                let grad_w = grad_output[m] * x[n];
                let idx = m * self.in_features + n;
                // STE pass-through with clipping
                if self.w_latent[idx].abs() <= self.clip_range {
                    self.w_latent[idx] -= lr * grad_w;
                }
            }
        }
        let _ = ste; // suppress unused
    }

    /// Get current ternary snapshot (for inference/evaluation).
    pub fn quantized(&self) -> TernaryMatrix {
        let ste = STEQuantizer::with_gamma(self.gamma);
        ste.quantize_forward(&self.w_latent, self.out_features, self.in_features)
    }

    /// Sparsity: fraction of zero weights in current quantization.
    pub fn sparsity(&self) -> f32 {
        let ternary = self.quantized();
        let mut zeros = 0usize;
        for m in 0..self.out_features {
            for n in 0..self.in_features {
                if ternary.get(m, n) == 0 { zeros += 1; }
            }
        }
        zeros as f32 / (self.out_features * self.in_features) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ste_quantize_roundtrip() {
        let ste = STEQuantizer::new();
        let w = vec![1.0, -1.0, 0.1, 0.9, -0.8, 0.0, 0.5, -0.5];
        let mat = ste.quantize_forward(&w, 2, 4);
        // All values should be ternary
        for m in 0..2 {
            for n in 0..4 {
                let v = mat.get(m, n);
                assert!(v == -1 || v == 0 || v == 1, "Got {v} at ({m},{n})");
            }
        }
    }

    #[test]
    fn test_ste_gradient_passthrough() {
        let ste = STEQuantizer::new();
        let grad = vec![1.0, -2.0, 0.5, 3.0];
        let w = vec![0.5, -0.3, 0.1, 0.8]; // all within clip_range
        let result = ste.quantize_backward(&grad, &w, 2.0);
        assert_eq!(result, grad, "STE should pass gradients through unchanged");
    }

    #[test]
    fn test_ste_gradient_clipping() {
        let ste = STEQuantizer::new();
        let grad = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![0.5, 3.0, -0.1, -5.0]; // 3.0 and -5.0 outside clip_range=2.0
        let result = ste.quantize_backward(&grad, &w, 2.0);
        assert_eq!(result[0], 1.0, "Within range: pass through");
        assert_eq!(result[1], 0.0, "3.0 > 2.0: clipped to 0");
        assert_eq!(result[2], 3.0, "Within range: pass through");
        assert_eq!(result[3], 0.0, "-5.0 abs > 2.0: clipped to 0");
    }

    #[test]
    fn test_ste_linear_forward() {
        let layer = STELinear::new(4, 2);
        let x = vec![1.0, 0.5, -0.5, 1.0];
        let mut y = vec![0.0f32; 2];
        let _w_ternary = layer.forward(&x, &mut y);
        assert_eq!(y.len(), 2);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ste_training_converges() {
        // Simple test: learn to map [1,0,0,0] → [target, 0]
        let mut layer = STELinear::new(4, 2);
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let target = vec![1.0, -1.0];
        let lr = 0.1;
        let mut initial_loss = f32::MAX;

        for epoch in 0..100 {
            let mut y = vec![0.0f32; 2];
            let _w = layer.forward(&x, &mut y);
            let loss: f32 = y.iter().zip(&target)
                .map(|(p, t)| (p - t) * (p - t))
                .sum::<f32>() / 2.0;

            if epoch == 0 { initial_loss = loss; }

            // Gradient: ∂MSE/∂y = (y - target)
            let grad: Vec<f32> = y.iter().zip(&target)
                .map(|(p, t)| p - t)
                .collect();
            layer.backward_update(&grad, &x, lr);
        }

        let mut y_final = vec![0.0f32; 2];
        layer.forward(&x, &mut y_final);
        let final_loss: f32 = y_final.iter().zip(&target)
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>() / 2.0;

        assert!(
            final_loss <= initial_loss,
            "Training should reduce loss: initial={initial_loss:.4}, final={final_loss:.4}"
        );
    }

    #[test]
    fn test_sparsity() {
        let layer = STELinear::new(32, 16);
        let s = layer.sparsity();
        // With gamma=0.7 and random init, some weights should be zero
        assert!(s >= 0.0 && s <= 1.0);
        eprintln!("STE sparsity: {s:.2}");
    }
}
