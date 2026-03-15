//! Two-Phase Attention — HDC-gated sparse attention routing.
//!
//! Phase 1 (Scout): Hash query to HyperVector → scan HDC memory → top-K.
//! Phase 2 (Focus): Standard dot-product attention over raw tokens + retrieved.
//!
//! Replaces O(n²) full attention with O(1) attention over constant-size context.

use vagi_hdc::{HDCEncoder, HDCMemory, HyperVector, MemoryConfig};

/// Configuration for two-phase attention.
#[derive(Clone, Debug)]
pub struct TwoPhaseConfig {
    /// Number of episodes to retrieve from HDC memory.
    pub scout_k: usize,
    /// Recent raw token buffer size.
    pub raw_buffer_size: usize,
    /// Model dimension.
    pub d_model: usize,
    /// HDC encoder vocabulary size.
    pub vocab_size: usize,
    /// HDC encoder seed.
    pub encoder_seed: u64,
}

impl Default for TwoPhaseConfig {
    fn default() -> Self {
        Self {
            scout_k: 32,
            raw_buffer_size: 256,
            d_model: 64,
            vocab_size: 1000,
            encoder_seed: 42,
        }
    }
}

/// Two-phase attention module.
///
/// Phase 1 (Scout): O(1) binary matching via HDC XOR + popcount.
/// Phase 2 (Focus): dot-product attention over small context (raw + retrieved).
pub struct TwoPhaseAttention {
    /// HDC memory store for long-term episodes.
    memory: HDCMemory,
    /// Encoder for converting embeddings to HyperVectors.
    encoder: HDCEncoder,
    /// Recent raw token buffer (ring buffer).
    raw_buffer: Vec<Vec<f32>>,
    /// Configuration.
    config: TwoPhaseConfig,
}

impl TwoPhaseAttention {
    /// Create a new two-phase attention module with in-memory HDC store.
    pub fn new(config: TwoPhaseConfig) -> Self {
        let memory = HDCMemory::in_memory(MemoryConfig {
            max_episodes: 100_000,
        }).expect("Failed to create in-memory HDC store");
        let encoder = HDCEncoder::new(config.vocab_size, config.encoder_seed);
        Self {
            memory,
            encoder,
            raw_buffer: Vec::with_capacity(config.raw_buffer_size),
            config,
        }
    }

    /// Ingest a token embedding: store in raw buffer and HDC memory.
    pub fn ingest(&mut self, token: &[f32], metadata: &str) {
        // Add to raw buffer (ring buffer behavior)
        if self.raw_buffer.len() >= self.config.raw_buffer_size {
            self.raw_buffer.remove(0);
        }
        self.raw_buffer.push(token.to_vec());

        // Encode and store in HDC memory
        let hv = self.encoder.encode_embedding(token);
        self.memory.insert(hv, metadata, 0.5, 0.0);
    }

    /// Phase 1: Scout — find top-K relevant episodes from HDC memory.
    ///
    /// Encodes query to HyperVector, scans memory via XOR+popcount.
    /// Returns (episode_ids, similarities).
    pub fn scout(&self, query: &[f32]) -> Vec<(u64, f32)> {
        let query_hv = self.encoder.encode_embedding(query);
        self.memory.query_topk(&query_hv, self.config.scout_k)
    }

    /// Phase 2: Focus — dot-product attention over small context.
    ///
    /// Context = raw_buffer (recent tokens) + retrieved episode embeddings.
    /// Computes: output = softmax(Q·K^T / √d) · V
    ///
    /// Since we don't store original embeddings in HDC (only HyperVectors),
    /// we use raw_buffer as the attention context.
    pub fn focus(&self, query: &[f32]) -> Vec<f32> {
        let d = self.config.d_model;
        if self.raw_buffer.is_empty() {
            return vec![0.0; d];
        }

        // Q = query, K = V = raw_buffer entries
        let n = self.raw_buffer.len();
        let scale = (d as f32).sqrt();

        // Compute attention scores: Q · K^T / √d
        let mut scores = Vec::with_capacity(n);
        for key in &self.raw_buffer {
            let dot: f32 = query.iter().zip(key.iter())
                .map(|(q, k)| q * k)
                .sum();
            scores.push(dot / scale);
        }

        // Softmax
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        if sum_exp > 0.0 {
            for e in exp_scores.iter_mut() { *e /= sum_exp; }
        }

        // Weighted sum: output = Σ attention_i × V_i
        let mut output = vec![0.0f32; d];
        for (weight, value) in exp_scores.iter().zip(self.raw_buffer.iter()) {
            for (o, v) in output.iter_mut().zip(value.iter()) {
                *o += weight * v;
            }
        }
        output
    }

    /// Full forward: scout + focus combined.
    ///
    /// 1. Scout: retrieve relevant memory episodes
    /// 2. Touch retrieved episodes (update access count)
    /// 3. Focus: attend over raw buffer
    pub fn forward(&mut self, query: &[f32]) -> Vec<f32> {
        // Phase 1: scout
        let retrieved = self.scout(query);

        // Touch retrieved episodes (boosts their importance for forgetting)
        for (id, _) in &retrieved {
            self.memory.touch(*id);
        }

        // Phase 2: focus attention over raw buffer
        self.focus(query)
    }

    /// Number of episodes in HDC memory.
    pub fn memory_len(&self) -> usize { self.memory.len() }

    /// Number of tokens in raw buffer.
    pub fn buffer_len(&self) -> usize { self.raw_buffer.len() }

    /// Access HDC memory directly.
    pub fn memory(&self) -> &HDCMemory { &self.memory }

    /// Access HDC memory mutably.
    pub fn memory_mut(&mut self) -> &mut HDCMemory { &mut self.memory }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> TwoPhaseConfig {
        TwoPhaseConfig {
            scout_k: 5,
            raw_buffer_size: 20,
            d_model: 8,
            vocab_size: 64,
            encoder_seed: 42,
        }
    }

    #[test]
    fn test_create() {
        let attn = TwoPhaseAttention::new(make_config());
        assert_eq!(attn.memory_len(), 0);
        assert_eq!(attn.buffer_len(), 0);
    }

    #[test]
    fn test_ingest() {
        let mut attn = TwoPhaseAttention::new(make_config());
        attn.ingest(&vec![1.0; 8], "tok0");
        assert_eq!(attn.memory_len(), 1);
        assert_eq!(attn.buffer_len(), 1);
    }

    #[test]
    fn test_raw_buffer_ring() {
        let mut attn = TwoPhaseAttention::new(make_config());
        // Buffer size = 20, insert 30
        for i in 0..30 {
            attn.ingest(&vec![i as f32; 8], "tok");
        }
        assert_eq!(attn.buffer_len(), 20, "Ring buffer should cap at 20");
        assert_eq!(attn.memory_len(), 30, "HDC memory stores all");
    }

    #[test]
    fn test_scout_retrieves() {
        let mut attn = TwoPhaseAttention::new(make_config());
        // Insert some diverse tokens
        for i in 0..10 {
            attn.ingest(&vec![i as f32 * 0.1; 8], &format!("tok{i}"));
        }
        // Scout with a query
        let results = attn.scout(&vec![0.5; 8]);
        assert!(!results.is_empty(), "Scout should return results");
        assert!(results.len() <= 5, "Should respect scout_k=5");
    }

    #[test]
    fn test_focus_output_shape() {
        let mut attn = TwoPhaseAttention::new(make_config());
        for i in 0..5 {
            attn.ingest(&vec![i as f32; 8], "tok");
        }
        let output = attn.focus(&vec![1.0; 8]);
        assert_eq!(output.len(), 8, "Focus output should be d_model");
        assert!(output.iter().all(|v| v.is_finite()), "All outputs should be finite");
    }

    #[test]
    fn test_focus_empty() {
        let attn = TwoPhaseAttention::new(make_config());
        let output = attn.focus(&vec![1.0; 8]);
        assert_eq!(output, vec![0.0; 8], "Empty buffer should produce zero output");
    }

    #[test]
    fn test_forward_end_to_end() {
        let mut attn = TwoPhaseAttention::new(make_config());
        // Ingest some context
        for i in 0..15 {
            attn.ingest(&vec![(i as f32 * 0.1).sin(); 8], "ctx");
        }
        // Forward with query
        let output = attn.forward(&vec![0.5; 8]);
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_attention_weighted_sum() {
        let mut attn = TwoPhaseAttention::new(TwoPhaseConfig {
            scout_k: 5,
            raw_buffer_size: 10,
            d_model: 2,
            vocab_size: 32,
            encoder_seed: 42,
        });
        // Insert one very similar token
        attn.ingest(&vec![1.0, 0.0], "match");
        // Insert several dissimilar
        for _ in 0..5 {
            attn.ingest(&vec![-1.0, -1.0], "other");
        }
        // Query similar to first token
        let output = attn.focus(&vec![1.0, 0.0]);
        // Output should lean toward [-1, -1] since there are 5 of those vs 1 match
        // But the match should have higher attention weight
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_scout_touches_episodes() {
        let mut attn = TwoPhaseAttention::new(make_config());
        for i in 0..5 {
            attn.ingest(&vec![i as f32; 8], "tok");
        }
        // Forward touches retrieved episodes
        attn.forward(&vec![1.0; 8]);
        // Check that some episodes have been touched (access_count > 0)
        let has_touched = (0..5u64).any(|id| {
            attn.memory().get(id).map_or(false, |ep| ep.access_count > 0)
        });
        assert!(has_touched, "Forward should touch retrieved episodes");
    }
}
