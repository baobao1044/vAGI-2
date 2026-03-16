//! Text dataset for language model training.
//!
//! Loads text, tokenizes, and creates overlapping sequences for training.

use crate::tokenizer::ByteTokenizer;

/// A text dataset: sequences of token IDs for next-token prediction.
pub struct TextDataset {
    /// All training sequences, each a Vec of token IDs.
    sequences: Vec<Vec<u32>>,
    /// Sequence length used for splitting.
    seq_len: usize,
}

impl TextDataset {
    /// Create from a raw string, splitting into overlapping sequences.
    ///
    /// Each sequence is `seq_len` tokens long (with BOS/EOS).
    pub fn from_string(text: &str, seq_len: usize) -> Self {
        let tok = ByteTokenizer::new();
        let all_tokens = tok.encode(text);
        let mut sequences = Vec::new();

        if all_tokens.len() <= seq_len {
            sequences.push(all_tokens);
        } else {
            let stride = seq_len / 2; // 50% overlap
            let mut start = 0;
            while start + seq_len <= all_tokens.len() {
                sequences.push(all_tokens[start..start + seq_len].to_vec());
                start += stride;
            }
            // Include the last chunk if there's remaining data
            if start < all_tokens.len() && all_tokens.len() - start >= 4 {
                sequences.push(all_tokens[all_tokens.len() - seq_len.min(all_tokens.len())..].to_vec());
            }
        }

        Self { sequences, seq_len }
    }

    /// Create from multiple text samples.
    pub fn from_samples(texts: &[&str], seq_len: usize) -> Self {
        let tok = ByteTokenizer::new();
        let sequences: Vec<Vec<u32>> = texts.iter()
            .map(|text| {
                let tokens = tok.encode(text);
                if tokens.len() > seq_len {
                    tokens[..seq_len].to_vec()
                } else {
                    tokens
                }
            })
            .filter(|seq| seq.len() >= 2)
            .collect();

        Self { sequences, seq_len }
    }

    /// Number of training sequences.
    pub fn len(&self) -> usize { self.sequences.len() }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool { self.sequences.is_empty() }

    /// Get a sequence by index.
    pub fn get(&self, idx: usize) -> &[u32] { &self.sequences[idx] }

    /// Get all sequences (for train_epoch).
    pub fn sequences(&self) -> &[Vec<u32>] { &self.sequences }

    /// Sequence length.
    pub fn seq_len(&self) -> usize { self.seq_len }

    /// Shuffle with a given RNG.
    pub fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        use rand::seq::SliceRandom;
        self.sequences.shuffle(rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_from_string() {
        let ds = TextDataset::from_string("Hello, world! This is a test.", 16);
        assert!(!ds.is_empty());
        for i in 0..ds.len() {
            assert!(ds.get(i).len() >= 2);
            assert!(ds.get(i).len() <= 16);
        }
    }

    #[test]
    fn test_dataset_from_samples() {
        let ds = TextDataset::from_samples(&["Hello", "World", "Test"], 32);
        assert_eq!(ds.len(), 3);
    }

    #[test]
    fn test_dataset_short_text() {
        let ds = TextDataset::from_string("Hi", 256);
        assert_eq!(ds.len(), 1);
        assert!(ds.get(0).len() >= 2); // at least BOS + bytes + EOS
    }

    #[test]
    fn test_dataset_shuffle() {
        let mut ds = TextDataset::from_samples(
            &["AAA", "BBB", "CCC", "DDD", "EEE"],
            32,
        );
        let before = ds.sequences().to_vec();
        let mut rng = rand::thread_rng();
        ds.shuffle(&mut rng);
        // After shuffle, at least some sequences should be in different positions
        // (probabilistic but very likely with 5 elements)
        let after = ds.sequences().to_vec();
        assert_eq!(before.len(), after.len());
    }
}
