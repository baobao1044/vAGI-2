//! Full language model: Embedding → Transformer layers → LM Head.
//!
//! Produces next-token logits. Supports autoregressive generation.


use crate::config::LMConfig;
use crate::embedding::Embedding;
use crate::transformer::TransformerLayer;
use vagi_core::bitnet::RMSNorm;
use vagi_core::ste::STELinear;

/// Ternary language model.
///
/// ```text
/// tokens → Embedding → [TransformerLayer × N] → RMSNorm → LM Head → logits
/// ```
#[derive(Clone)]
pub struct VagiLM {
    pub config: LMConfig,
    /// Token embedding table (f32).
    pub embedding: Embedding,
    /// Transformer layers.
    pub layers: Vec<TransformerLayer>,
    /// Final normalization.
    pub final_norm: RMSNorm,
    /// LM head: projects d_model → vocab_size.
    pub lm_head: STELinear,
}

impl VagiLM {
    /// Build a new model from config.
    pub fn new(config: LMConfig) -> Self {
        let layers = (0..config.n_layers)
            .map(|_| TransformerLayer::new(
                config.d_model,
                config.n_heads,
                config.ffn_dim,
                config.max_seq_len,
            ))
            .collect();

        Self {
            embedding: Embedding::new(config.vocab_size, config.d_model),
            layers,
            final_norm: RMSNorm::new(config.d_model),
            lm_head: STELinear::new(config.d_model, config.vocab_size),
            config,
        }
    }

    /// Forward pass: tokens → logits.
    ///
    /// Input: token IDs `[seq_len]`.
    /// Output: logits `[seq_len × vocab_size]` (flat).
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let seq_len = tokens.len();
        let d = self.config.d_model;
        let v = self.config.vocab_size;

        // 1. Embedding lookup: [seq_len × d_model]
        let mut hidden = self.embedding.forward(tokens);

        // 2. Transformer layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        // 3. Final RMSNorm per token
        for t in 0..seq_len {
            self.final_norm.forward(&mut hidden[t * d..(t + 1) * d]);
        }

        // 4. LM head: project to vocab logits per token
        let mut logits = vec![0.0f32; seq_len * v];
        for t in 0..seq_len {
            self.lm_head.forward(
                &hidden[t * d..(t + 1) * d],
                &mut logits[t * v..(t + 1) * v],
            );
        }

        logits
    }

    /// Get logits for the last token only (for generation).
    pub fn forward_last(&self, tokens: &[u32]) -> Vec<f32> {
        let logits = self.forward(tokens);
        let v = self.config.vocab_size;
        let seq_len = tokens.len();
        logits[(seq_len - 1) * v..seq_len * v].to_vec()
    }

    /// Autoregressive text generation.
    ///
    /// Returns generated token IDs (excluding prompt).
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Vec<u32> {
        let mut rng = rand::thread_rng();
        let mut tokens = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            // Truncate to max_seq_len if needed
            let start = if tokens.len() > self.config.max_seq_len {
                tokens.len() - self.config.max_seq_len
            } else {
                0
            };
            let context = &tokens[start..];

            // Get logits for last position
            let logits = self.forward_last(context);

            // Sample next token
            let next = if temperature < 1e-6 {
                // Greedy: argmax
                argmax(&logits)
            } else {
                // Temperature sampling
                sample_temperature(&logits, temperature, &mut rng)
            };

            tokens.push(next);
            generated.push(next);

            // Stop at EOS
            if next == crate::tokenizer::EOS_ID {
                break;
            }
        }

        generated
    }

    /// Estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let embed = self.embedding.memory_bytes();
        // STELinear stores f32 latent weights
        let lm_head = self.config.d_model * self.config.vocab_size * 4;
        let per_layer = {
            let attn = 4 * self.config.d_model * self.config.d_model * 4; // 4 projections, f32 latent
            let ffn = 2 * self.config.d_model * self.config.ffn_dim * 4;  // up + down, f32 latent
            let norms = 2 * self.config.d_model * 4;
            attn + ffn + norms
        };
        embed + self.config.n_layers * per_layer + lm_head
    }

    /// Parameter count.
    pub fn param_count(&self) -> usize {
        self.config.param_count()
    }
}

/// Argmax over a slice.
fn argmax(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Temperature-scaled sampling.
fn sample_temperature(logits: &[f32], temperature: f32, rng: &mut impl rand::Rng) -> u32 {
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Stable softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Sample from distribution
    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::ByteTokenizer;

    #[test]
    fn test_model_forward_shape() {
        let config = LMConfig::tiny();
        let model = VagiLM::new(config.clone());
        let tokens = vec![65, 66, 67, 68]; // "ABCD"
        let logits = model.forward(&tokens);
        assert_eq!(logits.len(), 4 * config.vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_model_forward_last() {
        let config = LMConfig::tiny();
        let model = VagiLM::new(config.clone());
        let tokens = vec![65, 66];
        let logits = model.forward_last(&tokens);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_model_generate() {
        let config = LMConfig::tiny();
        let model = VagiLM::new(config);
        let tok = ByteTokenizer::new();
        let prompt = tok.encode_raw("Hi");
        let generated = model.generate(&prompt, 5, 1.0);
        assert!(!generated.is_empty());
        assert!(generated.len() <= 5);
        // All tokens should be valid IDs
        for &t in &generated {
            assert!((t as usize) < tok.vocab_size(),
                "Generated invalid token: {t}");
        }
    }

    #[test]
    fn test_model_greedy_deterministic() {
        let config = LMConfig::tiny();
        let model = VagiLM::new(config);
        let prompt = vec![65, 66]; // "AB"
        let gen1 = model.generate(&prompt, 3, 0.0);
        let gen2 = model.generate(&prompt, 3, 0.0);
        assert_eq!(gen1, gen2, "Greedy generation should be deterministic");
    }

    #[test]
    fn test_param_count() {
        let config = LMConfig::tiny();
        let model = VagiLM::new(config);
        let params = model.param_count();
        eprintln!("Tiny model params: {params}");
        assert!(params > 0);
    }

    #[test]
    fn test_memory_usage() {
        let config = LMConfig::tiny();
        let model = VagiLM::new(config);
        let mem = model.memory_bytes();
        eprintln!("Tiny model memory: {} bytes ({:.1} KB)", mem, mem as f64 / 1024.0);
        assert!(mem > 0);
    }

    #[test]
    fn test_end_to_end_text() {
        let tok = ByteTokenizer::new();
        let config = LMConfig::tiny();
        let model = VagiLM::new(config);

        let prompt = tok.encode("Hello");
        let generated = model.generate(&prompt, 10, 0.8);
        let full_tokens: Vec<u32> = prompt.iter().chain(generated.iter()).cloned().collect();
        let text = tok.decode(&full_tokens);
        eprintln!("Generated text: '{text}'");
        assert!(text.starts_with("Hello"));
    }
}
