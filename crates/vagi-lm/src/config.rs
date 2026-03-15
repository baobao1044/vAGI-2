//! Language model configuration.

use crate::tokenizer::VOCAB_SIZE;

/// Language model configuration.
#[derive(Clone, Debug)]
pub struct LMConfig {
    /// Model dimension (embedding size).
    pub d_model: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// FFN hidden dimension (typically 4 × d_model).
    pub ffn_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
}

impl LMConfig {
    /// Tiny config for testing (~200K params).
    pub fn tiny() -> Self {
        Self {
            d_model: 64,
            n_layers: 4,
            n_heads: 4,
            ffn_dim: 256,
            vocab_size: VOCAB_SIZE,
            max_seq_len: 256,
            rms_eps: 1e-6,
        }
    }

    /// Small config (~5M params).
    pub fn small() -> Self {
        Self {
            d_model: 256,
            n_layers: 6,
            n_heads: 8,
            ffn_dim: 1024,
            vocab_size: VOCAB_SIZE,
            max_seq_len: 512,
            rms_eps: 1e-6,
        }
    }

    /// Base config (~50M params).
    pub fn base() -> Self {
        Self {
            d_model: 512,
            n_layers: 12,
            n_heads: 8,
            ffn_dim: 2048,
            vocab_size: VOCAB_SIZE,
            max_seq_len: 1024,
            rms_eps: 1e-6,
        }
    }

    /// Head dimension (d_model / n_heads).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Estimated parameter count.
    pub fn param_count(&self) -> usize {
        let embed = self.vocab_size * self.d_model;
        let attn = 4 * self.d_model * self.d_model; // Q, K, V, O projections
        let ffn = 2 * self.d_model * self.ffn_dim;  // up + down
        let per_layer = attn + ffn;
        embed + self.n_layers * per_layer + embed // +embed for LM head (tied)
    }
}

impl Default for LMConfig {
    fn default() -> Self { Self::tiny() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_params() {
        let c = LMConfig::tiny();
        let p = c.param_count();
        eprintln!("Tiny params: {p}");
        assert!(p > 100_000 && p < 500_000);
    }

    #[test]
    fn test_base_params() {
        let c = LMConfig::base();
        let p = c.param_count();
        eprintln!("Base params: {p}");
        assert!(p > 30_000_000 && p < 80_000_000);
    }

    #[test]
    fn test_head_dim() {
        let c = LMConfig::tiny();
        assert_eq!(c.head_dim(), 16); // 64 / 4
    }
}
