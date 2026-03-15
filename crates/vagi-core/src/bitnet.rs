//! BitNet blocks — ternary neural network building blocks.
//!
//! All weights are ternary {-1, 0, +1}. Activations remain f32.
//! Matrix multiplications use conditional add/subtract only (no float multiply).

use crate::error::VagiError;
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
        // Rough estimate: 4 * d_model^2 per layer (Q,K,V,O) + 2 * d_model * ffn_dim (FFN)
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
    /// Create a new RMSNorm with all-ones scale.
    pub fn new(dim: usize) -> Self {
        Self {
            weight: vec![1.0; dim],
            eps: 1e-6,
        }
    }

    /// Forward pass (in-place).
    pub fn forward(&self, x: &mut [f32]) {
        let n = x.len();
        if n == 0 {
            return;
        }
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f32 + self.eps).sqrt();
        let inv_rms = 1.0 / rms;
        for (xi, wi) in x.iter_mut().zip(self.weight.iter()) {
            *xi = *xi * inv_rms * wi;
        }
    }

    /// Dimension of this norm layer.
    pub fn dim(&self) -> usize {
        self.weight.len()
    }
}

/// BitNet linear layer with ternary weights.
///
/// Stores weights as f32 for now (will be replaced with TernaryMatrix
/// for production). Forward: y = W * quantize(x) + bias.
#[derive(Clone, Debug)]
pub struct BitNetLinear {
    /// Weight matrix [out_features × in_features], stored row-major.
    pub weight: Vec<f32>,
    /// Optional bias [out_features].
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl BitNetLinear {
    /// Create with random ternary weights.
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let mut rng = rand::thread_rng();
        let weight: Vec<f32> = (0..out_features * in_features)
            .map(|_| match rng.gen_range(0u8..3) {
                0 => -1.0,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();
        let bias = if use_bias {
            Some(vec![0.0; out_features])
        } else {
            None
        };
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Create with zeros.
    pub fn zeros(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let weight = vec![0.0; out_features * in_features];
        let bias = if use_bias {
            Some(vec![0.0; out_features])
        } else {
            None
        };
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass: y = W * x + bias.
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
        for i in 0..self.out_features {
            let mut sum = 0.0f32;
            let row_start = i * self.in_features;
            for j in 0..self.in_features {
                let w = self.weight[row_start + j];
                // Ternary: only add/subtract, skip zeros
                if w > 0.5 {
                    sum += x[j];
                } else if w < -0.5 {
                    sum -= x[j];
                }
            }
            if let Some(ref bias) = self.bias {
                sum += bias[i];
            }
            y[i] = sum;
        }
        Ok(())
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
    /// Create a new BitNet block.
    pub fn new(d_model: usize, ffn_dim: usize) -> Self {
        Self {
            norm: RMSNorm::new(d_model),
            ffn_up: BitNetLinear::new(d_model, ffn_dim, false),
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
        // Save residual
        let residual: Vec<f32> = x.to_vec();

        // RMSNorm
        self.norm.forward(x);

        // FFN up: d_model → ffn_dim
        let mut hidden = vec![0.0f32; self.ffn_dim];
        self.ffn_up.forward(x, &mut hidden)?;

        // SiLU activation
        for h in hidden.iter_mut() {
            *h = silu(*h);
        }

        // FFN down: ffn_dim → d_model
        self.ffn_down.forward(&hidden, x)?;

        // Residual connection
        for (xi, ri) in x.iter_mut().zip(residual.iter()) {
            *xi += ri;
        }

        Ok(())
    }

    /// Forward pass returning new vector (non-mutating).
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
        // Output should differ from input due to FFN processing
        assert!(output != input);
    }

    #[test]
    fn test_config_param_count() {
        let config = BitNetConfig::small_100m();
        let count = config.param_count();
        // Should be roughly 100M
        assert!(count > 50_000_000, "Param count too low: {count}");
        assert!(count < 200_000_000, "Param count too high: {count}");
    }
}
