//! HDC Encoder — convert token sequences or f32 embeddings into HyperVectors.
//!
//! Uses positional encoding via cyclic permutation and majority-rule bundling.

use crate::vector::HyperVector;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Encoder that converts token sequences or embeddings to HyperVectors.
pub struct HDCEncoder {
    /// Random codebook: one HyperVector per token ID.
    codebook: Vec<HyperVector>,
    /// Vocabulary size.
    pub vocab_size: usize,
}

impl HDCEncoder {
    /// Create encoder with deterministic random codebook from seed.
    pub fn new(vocab_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let codebook: Vec<HyperVector> = (0..vocab_size)
            .map(|_| HyperVector::random(&mut rng))
            .collect();
        Self { codebook, vocab_size }
    }

    /// Encode token ID sequence → single HyperVector.
    ///
    /// Method: positional encoding via permutation + bundling.
    /// For tokens [t0, t1, t2]:
    ///   encoded = bundle([codebook[t0].permute(0), codebook[t1].permute(1), ...])
    ///
    /// This preserves order: [1,2,3] ≠ [3,2,1].
    pub fn encode_tokens(&self, tokens: &[u32]) -> HyperVector {
        if tokens.is_empty() {
            return HyperVector::zero();
        }
        let shifted: Vec<HyperVector> = tokens.iter().enumerate()
            .map(|(i, &t)| {
                let idx = (t as usize) % self.vocab_size;
                self.codebook[idx].permute(i as i32)
            })
            .collect();
        let refs: Vec<&HyperVector> = shifted.iter().collect();
        HyperVector::bundle(&refs)
    }

    /// Encode f32 embedding → HyperVector via random projection.
    ///
    /// For each bit position, compute a dot product with a pseudo-random
    /// weight vector derived from the codebook. Bit = 1 if dot > 0.
    pub fn encode_embedding(&self, embedding: &[f32]) -> HyperVector {
        let dim = embedding.len();
        let mut result = HyperVector::zero();

        for word_idx in 0..HyperVector::WORDS {
            let mut word = 0u64;
            for bit in 0..64 {
                let global_bit = word_idx * 64 + bit;
                // Pseudo-random projection: use bit position to select
                // codebook entry and bit within it as sign
                let cb_idx = global_bit % self.vocab_size;
                let mut dot = 0.0f32;
                for j in 0..dim {
                    // Use codebook bits as random signs: bit set → +1, else → -1
                    let sign_word = (j / 64) % HyperVector::WORDS;
                    let sign_bit = j % 64;
                    let sign = if self.codebook[cb_idx].data[sign_word] & (1u64 << sign_bit) != 0 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                    dot += sign * embedding[j];
                }
                if dot > 0.0 {
                    word |= 1u64 << bit;
                }
            }
            result.data[word_idx] = word;
        }
        result
    }

    /// Get a specific codebook entry.
    pub fn get_token_vector(&self, token_id: u32) -> &HyperVector {
        &self.codebook[(token_id as usize) % self.vocab_size]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_encoding() {
        let enc = HDCEncoder::new(100, 42);
        let tokens = vec![1, 5, 10];
        let v1 = enc.encode_tokens(&tokens);
        let v2 = enc.encode_tokens(&tokens);
        assert_eq!(v1, v2, "Same tokens should produce same HyperVector");
    }

    #[test]
    fn test_order_matters() {
        let enc = HDCEncoder::new(100, 42);
        let v1 = enc.encode_tokens(&[1, 2, 3]);
        let v2 = enc.encode_tokens(&[3, 2, 1]);
        let sim = v1.similarity(&v2);
        assert!(sim < 0.8, "Different order should produce different vectors, sim={sim}");
    }

    #[test]
    fn test_similar_sequences() {
        let enc = HDCEncoder::new(100, 42);
        let v1 = enc.encode_tokens(&[10, 20, 30]);
        let v2 = enc.encode_tokens(&[10, 20, 31]); // one token different
        let v3 = enc.encode_tokens(&[50, 60, 70]); // totally different
        let sim_12 = v1.similarity(&v2);
        let sim_13 = v1.similarity(&v3);
        assert!(sim_12 > sim_13,
            "Similar seqs should be more similar ({sim_12}) than different ({sim_13})");
    }

    #[test]
    fn test_empty_tokens() {
        let enc = HDCEncoder::new(100, 42);
        let v = enc.encode_tokens(&[]);
        assert_eq!(v, HyperVector::zero());
    }

    #[test]
    fn test_embedding_encoding() {
        let enc = HDCEncoder::new(64, 42);
        let emb1 = vec![1.0, 0.0, -1.0, 0.5];
        let emb2 = vec![1.0, 0.0, -1.0, 0.5];
        let emb3 = vec![-1.0, 0.0, 1.0, -0.5]; // opposite
        let v1 = enc.encode_embedding(&emb1);
        let v2 = enc.encode_embedding(&emb2);
        let v3 = enc.encode_embedding(&emb3);
        assert_eq!(v1, v2, "Same embedding → same HyperVector");
        let sim_13 = v1.similarity(&v3);
        assert!(sim_13 < 0.3, "Opposite embeddings should be dissimilar: {sim_13}");
    }
}
