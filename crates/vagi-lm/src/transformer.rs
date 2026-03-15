//! Transformer layer = Attention + FFN with residual connections.
//!
//! ```text
//! x → RMSNorm → CausalAttention → + residual
//!   → RMSNorm → FFN(up → AdaptiveBasis → down) → + residual
//! ```
//!
//! FFN uses AdaptiveBlock from vagi-core (ternary weights + learnable activation).

use vagi_core::bitnet::RMSNorm;
use vagi_core::ste::STELinear;
use vagi_core::adaptive::AdaptiveBasis;
use crate::attention::CausalAttention;

/// Single transformer layer.
#[derive(Clone, Debug)]
pub struct TransformerLayer {
    /// Pre-attention normalization.
    pub attn_norm: RMSNorm,
    /// Multi-head causal self-attention.
    pub attention: CausalAttention,
    /// Pre-FFN normalization.
    pub ffn_norm: RMSNorm,
    /// FFN up projection [d_model → ffn_dim] (trainable ternary).
    pub ffn_up: STELinear,
    /// Learnable activation.
    pub activation: AdaptiveBasis,
    /// FFN down projection [ffn_dim → d_model] (trainable ternary).
    pub ffn_down: STELinear,
    pub d_model: usize,
    pub ffn_dim: usize,
}

impl TransformerLayer {
    pub fn new(d_model: usize, n_heads: usize, ffn_dim: usize, max_seq_len: usize) -> Self {
        Self {
            attn_norm: RMSNorm::new(d_model),
            attention: CausalAttention::new(d_model, n_heads, max_seq_len),
            ffn_norm: RMSNorm::new(d_model),
            ffn_up: STELinear::new(d_model, ffn_dim),
            activation: AdaptiveBasis::trimmed(), // identity + sin + tanh
            ffn_down: STELinear::new(ffn_dim, d_model),
            d_model,
            ffn_dim,
        }
    }

    /// Forward pass over a sequence.
    ///
    /// Input/output: flat `[seq_len × d_model]`.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let d = self.d_model;

        // ── Attention block ──
        // x_normed = RMSNorm(x)  per token
        let mut normed = x.to_vec();
        for t in 0..seq_len {
            self.attn_norm.forward(&mut normed[t * d..(t + 1) * d]);
        }

        // attn_out = CausalAttention(x_normed)
        let attn_out = self.attention.forward(&normed, seq_len);

        // residual: h = x + attn_out
        let mut h: Vec<f32> = x.iter().zip(attn_out.iter()).map(|(a, b)| a + b).collect();

        // ── FFN block ──
        // h_normed = RMSNorm(h)  per token
        let mut h_normed = h.clone();
        for t in 0..seq_len {
            self.ffn_norm.forward(&mut h_normed[t * d..(t + 1) * d]);
        }

        // FFN: up → activation → down, per token
        for t in 0..seq_len {
            let tok = &h_normed[t * d..(t + 1) * d];
            let mut up = vec![0.0f32; self.ffn_dim];
            self.ffn_up.forward(tok, &mut up);
            self.activation.forward(&mut up);
            let mut down = vec![0.0f32; d];
            self.ffn_down.forward(&up, &mut down);

            // residual: h = h + ffn_out
            for j in 0..d {
                h[t * d + j] += down[j];
            }
        }

        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_layer_shape() {
        let layer = TransformerLayer::new(64, 4, 256, 128);
        let seq_len = 8;
        let x = vec![0.1f32; seq_len * 64];
        let out = layer.forward(&x, seq_len);
        assert_eq!(out.len(), seq_len * 64);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_residual_connection() {
        let layer = TransformerLayer::new(32, 2, 128, 64);
        let x = vec![1.0f32; 32];
        let out = layer.forward(&x, 1);
        // Output should differ from input (attention + FFN change it)
        assert_ne!(x, out);
        // But should be close-ish (residual keeps input signal)
        let diff: f32 = x.iter().zip(out.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / x.len() as f32;
        assert!(diff < 100.0, "Residual should keep output in reasonable range, diff={diff}");
    }
}
