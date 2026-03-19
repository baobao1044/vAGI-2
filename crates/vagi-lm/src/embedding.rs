//! Embedding table — f32 token vectors.
//!
//! Embeddings use full precision (not ternary) because token representations
//! need fine-grained distinctions. Consistent with BitNet papers.

use rand::Rng;

/// Token embedding table.
#[derive(Clone, Debug)]
pub struct Embedding {
    /// Weight matrix [vocab_size × d_model], row-major.
    pub weight: Vec<f32>,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl Embedding {
    /// Create with small random init (Xavier-like).
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / d_model as f32).sqrt();
        let weight: Vec<f32> = (0..vocab_size * d_model)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self { weight, vocab_size, d_model }
    }

    /// Look up embedding for a single token.
    #[inline]
    pub fn forward_one(&self, token_id: u32) -> &[f32] {
        let idx = token_id as usize;
        let start = idx * self.d_model;
        &self.weight[start..start + self.d_model]
    }

    /// Look up embeddings for a sequence of tokens.
    /// Returns [seq_len × d_model] as flat Vec.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(tokens.len() * self.d_model);
        for &tok in tokens {
            out.extend_from_slice(self.forward_one(tok));
        }
        out
    }

    /// Look up embedding for a single token (owned copy).
    #[inline]
    pub fn forward_single(&self, token_id: u32) -> Vec<f32> {
        self.forward_one(token_id).to_vec()
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.weight.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_shape() {
        let emb = Embedding::new(259, 64);
        let vec = emb.forward_one(65); // 'A'
        assert_eq!(vec.len(), 64);
        assert!(vec.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_embedding_batch() {
        let emb = Embedding::new(259, 64);
        let tokens = vec![65, 66, 67]; // "ABC"
        let out = emb.forward(&tokens);
        assert_eq!(out.len(), 3 * 64);
    }

    #[test]
    fn test_embedding_deterministic_lookup() {
        let emb = Embedding::new(259, 32);
        let v1 = emb.forward_one(100);
        let v2 = emb.forward_one(100);
        assert_eq!(v1, v2, "Same token should give same embedding");
    }
}
