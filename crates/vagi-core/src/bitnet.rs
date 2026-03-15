//! BitNet blocks — ternary neural network building blocks.
//!
//! All weights are ternary {-1, 0, +1}. Activations remain f32.
//! Matrix multiplications use packed ternary matvec (addition-only).

use crate::error::VagiError;
use crate::ternary::{TernaryMatrix, ternary_matvec};
use rand::Rng;

/// Model configuration.
#[derive(Clone, Debug)]
pub struct BitNetConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_norm_eps: f32,
}

impl BitNetConfig {
    /// 100M parameter configuration.
    pub fn small_100m() -> Self {
        Self {
            d_model: 768,
            n_layers: 12,
            n_heads: 12,
            ffn_dim: 3072,
            vocab_size: 32000,
            max_seq_len: 2048,
            rms_norm_eps: 1e-6,
        }
    }

    /// Estimated parameter count.
    pub fn param_count(&self) -> usize {
        let attn_params = 4 * self.d_model * self.d_model;
        let ffn_params = 2 * self.d_model * self.ffn_dim;
        self.n_layers * (attn_params + ffn_params) + self.vocab_size * self.d_model
    }
}

/// Root Mean Square Layer Normalization.
///
/// y_i = (x_i / rms(x)) * γ_i
/// where rms(x) = sqrt(mean(x²) + ε)
#[derive(Clone, Debug)]
pub struct RMSNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            weight: vec![1.0; dim],
            eps: 1e-6,
        }
    }

    pub fn forward(&self, x: &mut [f32]) {
        let n = x.len();
        if n == 0 { return; }
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f32 + self.eps).sqrt();
        let inv_rms = 1.0 / rms;
        for (xi, wi) in x.iter_mut().zip(self.weight.iter()) {
            *xi = *xi * inv_rms * wi;
        }
    }

    pub fn dim(&self) -> usize { self.weight.len() }
}

/// BitNet linear layer with packed ternary weights.
///
/// Weights stored as TernaryMatrix (2-bit packed, ~16× smaller than f32).
/// Forward: y = scale * Σ ternary(W) * x + bias (addition-only inner loop).
#[derive(Clone, Debug)]
pub struct BitNetLinear {
    /// Packed ternary weight matrix [out_features × in_features].
    packed: TernaryMatrix,
    /// Optional bias [out_features].
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl BitNetLinear {
    /// Create with random ternary weights.
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let mut rng = rand::thread_rng();
        let ternary: Vec<i8> = (0..out_features * in_features)
            .map(|_| match rng.gen_range(0u8..3) {
                0 => -1i8,
                1 => 0,
                _ => 1,
            })
            .collect();
        let packed = TernaryMatrix::from_ternary(&ternary, out_features, in_features);
        let bias = if use_bias { Some(vec![0.0; out_features]) } else { None };
        Self { packed, bias, in_features, out_features }
    }

    /// Create with all-zero weights.
    pub fn zeros(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let packed = TernaryMatrix::zeros(out_features, in_features);
        let bias = if use_bias { Some(vec![0.0; out_features]) } else { None };
        Self { packed, bias, in_features, out_features }
    }

    /// Create from an existing TernaryMatrix.
    pub fn from_packed(packed: TernaryMatrix, use_bias: bool) -> Self {
        let out_features = packed.rows();
        let in_features = packed.cols();
        let bias = if use_bias { Some(vec![0.0; out_features]) } else { None };
        Self { packed, bias, in_features, out_features }
    }

    /// Forward pass: y = W * x + bias, using packed ternary matvec.
    pub fn forward(&self, x: &[f32], y: &mut [f32]) -> Result<(), VagiError> {
        if x.len() != self.in_features {
            return Err(VagiError::ShapeMismatch {
                expected: format!("{}", self.in_features),
                got: format!("{}", x.len()),
            });
        }
        if y.len() != self.out_features {
            return Err(VagiError::ShapeMismatch {
                expected: format!("{}", self.out_features),
                got: format!("{}", y.len()),
            });
        }

        ternary_matvec(&self.packed, x, y);

        if let Some(ref bias) = self.bias {
            for (yi, bi) in y.iter_mut().zip(bias.iter()) {
                *yi += bi;
            }
        }
        Ok(())
    }

    /// Get a single weight value as f32 (-1.0, 0.0, or +1.0).
    /// Used for gradient backpropagation through frozen ternary weights.
    #[inline]
    pub fn get_weight(&self, row: usize, col: usize) -> f32 {
        self.packed.get(row, col) as f32
    }

    /// Access the underlying TernaryMatrix.
    pub fn packed(&self) -> &TernaryMatrix { &self.packed }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.packed.memory_bytes()
            + self.bias.as_ref().map_or(0, |b| b.len() * 4)
    }
}

/// SiLU activation: x * sigmoid(x).
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// BitNet feed-forward block with residual connection.
///
/// x → RMSNorm → FFN_up → SiLU → FFN_down → + x
#[derive(Clone, Debug)]
pub struct BitNetBlock {
    pub norm: RMSNorm,
    pub ffn_up: BitNetLinear,
    pub ffn_down: BitNetLinear,
    pub d_model: usize,
    pub ffn_dim: usize,
}

impl BitNetBlock {
    pub fn new(d_model: usize, ffn_dim: usize) -> Self {
        Self {
            norm: RMSNorm::new(d_model),
            ffn_up: BitNetLinear::new(d_model, ffn_dim, false),
            ffn_down: BitNetLinear::new(ffn_dim, d_model, false),
            d_model,
            ffn_dim,
        }
    }

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

        for h in hidden.iter_mut() {
            *h = silu(*h);
        }

        self.ffn_down.forward(&hidden, x)?;

        for (xi, ri) in x.iter_mut().zip(residual.iter()) {
            *xi += ri;
        }
        Ok(())
    }

    pub fn forward_vec(&self, input: &[f32]) -> Result<Vec<f32>, VagiError> {
        let mut output = input.to_vec();
        self.forward(&mut output)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_unit_magnitude() {
        let norm = RMSNorm::new(768);
        let mut x: Vec<f32> = (0..768).map(|i| (i as f32) * 0.01 - 3.84).collect();
        norm.forward(&mut x);
        let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 0.01, "RMS after norm should be ~1.0, got {rms}");
    }

    #[test]
    fn test_bitnet_linear_shape_check() {
        let linear = BitNetLinear::zeros(768, 3072, false);
        let x = vec![1.0f32; 768];
        let mut y = vec![0.0f32; 3072];
        assert!(linear.forward(&x, &mut y).is_ok());

        let x_bad = vec![1.0f32; 100];
        assert!(linear.forward(&x_bad, &mut y).is_err());
    }

    #[test]
    fn test_bitnet_block_residual() {
        let block = BitNetBlock::new(64, 256);
        let input = vec![1.0f32; 64];
        let output = block.forward_vec(&input).unwrap();
        assert_eq!(output.len(), 64);
        assert!(output != input);
    }

    #[test]
    fn test_config_param_count() {
        let config = BitNetConfig::small_100m();
        let count = config.param_count();
        assert!(count > 50_000_000, "Param count too low: {count}");
        assert!(count < 200_000_000, "Param count too high: {count}");
    }

    #[test]
    fn test_get_weight() {
        let linear = BitNetLinear::new(4, 2, false);
        for r in 0..2 {
            for c in 0..4 {
                let w = linear.get_weight(r, c);
                assert!(w == -1.0 || w == 0.0 || w == 1.0,
                    "Weight at ({r},{c}) should be ternary, got {w}");
            }
        }
    }

    #[test]
    fn test_memory_savings() {
        let linear = BitNetLinear::new(768, 3072, false);
        let mem = linear.memory_bytes();
        let f32_mem = 768 * 3072 * 4;
        let ratio = f32_mem as f64 / mem as f64;
        assert!(ratio > 10.0, "Should be >10× smaller, got {ratio:.1}×");
    }
}
