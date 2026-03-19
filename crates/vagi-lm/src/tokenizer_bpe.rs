//! Simple Byte-Pair Encoding (BPE) tokenizer for Vietnamese.
//!
//! Learns common byte-pair merges from a corpus, producing
//! a compact vocabulary that represents Vietnamese syllables
//! and common words as single tokens.

use std::collections::HashMap;

/// BPE tokenizer with learned merge rules.
pub struct BPETokenizer {
    /// Merge rules: (pair_a, pair_b) → merged_id
    merges: Vec<(u32, u32, u32)>,
    /// Token to bytes mapping (for decoding)
    vocab: Vec<Vec<u8>>,
    /// Bytes to token mapping (for encoding)
    byte_to_id: HashMap<Vec<u8>, u32>,
    /// Vocabulary size
    vocab_size: usize,
}

impl BPETokenizer {
    /// Base vocabulary: 256 bytes + PAD + BOS + EOS = 259 tokens.
    const BASE_VOCAB: usize = 259;
    const PAD_ID: u32 = 256;
    const BOS_ID: u32 = 257;
    const EOS_ID: u32 = 258;

    /// Train BPE tokenizer from corpus.
    ///
    /// `corpus`: training text.
    /// `n_merges`: number of merge operations to learn.
    pub fn train(corpus: &str, n_merges: usize) -> Self {
        // Initialize vocabulary with byte-level tokens
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(Self::BASE_VOCAB + n_merges);
        for b in 0u8..=255 {
            vocab.push(vec![b]);
        }
        // PAD, BOS, EOS
        vocab.push(vec![]); // PAD
        vocab.push(vec![]); // BOS
        vocab.push(vec![]); // EOS

        // Tokenize corpus into byte sequences (one per word)
        let mut words: Vec<Vec<u32>> = corpus.split_whitespace()
            .map(|w| w.bytes().map(|b| b as u32).collect())
            .collect();

        let mut merges = Vec::with_capacity(n_merges);

        for _ in 0..n_merges {
            // Count all adjacent pairs
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for word in &words {
                for pair in word.windows(2) {
                    *pair_counts.entry((pair[0], pair[1])).or_default() += 1;
                }
            }

            // Find most frequent pair
            let best = match pair_counts.into_iter().max_by_key(|&(_, count)| count) {
                Some((pair, count)) if count >= 2 => pair,
                _ => break, // No more mergeable pairs
            };

            // Create new token
            let new_id = vocab.len() as u32;
            let mut new_bytes = vocab[best.0 as usize].clone();
            new_bytes.extend_from_slice(&vocab[best.1 as usize]);
            vocab.push(new_bytes);
            merges.push((best.0, best.1, new_id));

            // Apply merge to all words
            for word in &mut words {
                let mut i = 0;
                while i + 1 < word.len() {
                    if word[i] == best.0 && word[i + 1] == best.1 {
                        word[i] = new_id;
                        word.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Build reverse lookup
        let mut byte_to_id = HashMap::new();
        for (id, bytes) in vocab.iter().enumerate() {
            if !bytes.is_empty() {
                byte_to_id.insert(bytes.clone(), id as u32);
            }
        }

        let vocab_size = vocab.len();
        Self { merges, vocab, byte_to_id, vocab_size }
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = vec![Self::BOS_ID];

        for word in text.split_whitespace() {
            let mut tokens: Vec<u32> = word.bytes().map(|b| b as u32).collect();

            // Apply merges in order
            for &(a, b, merged) in &self.merges {
                let mut i = 0;
                while i + 1 < tokens.len() {
                    if tokens[i] == a && tokens[i + 1] == b {
                        tokens[i] = merged;
                        tokens.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }

            result.extend_from_slice(&tokens);
            result.push(b' ' as u32); // space separator
        }

        // Remove trailing space, add EOS
        if result.last() == Some(&(b' ' as u32)) {
            result.pop();
        }
        result.push(Self::EOS_ID);
        result
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            let idx = id as usize;
            if idx < self.vocab.len() && !self.vocab[idx].is_empty() {
                bytes.extend_from_slice(&self.vocab[idx]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize { self.vocab_size }

    /// Number of learned merges.
    pub fn n_merges(&self) -> usize { self.merges.len() }

    /// Compression ratio (bytes per token) on given text.
    pub fn compression_ratio(&self, text: &str) -> f32 {
        let tokens = self.encode(text);
        let n_tokens = tokens.len().max(1);
        text.len() as f32 / n_tokens as f32
    }

    /// Save merge rules to file (one merge per line: "a b merged_id").
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "{}", self.merges.len())?;
        for &(a, b, m) in &self.merges {
            writeln!(f, "{} {} {}", a, b, m)?;
        }
        Ok(())
    }

    /// Load merge rules from file and reconstruct tokenizer.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let mut lines = data.lines();
        let n: usize = lines.next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Empty file"))?
            .trim().parse()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Base vocab: 256 bytes + PAD + BOS + EOS
        let mut vocab: Vec<Vec<u8>> = Vec::with_capacity(Self::BASE_VOCAB + n);
        for b in 0u8..=255 { vocab.push(vec![b]); }
        vocab.push(vec![]); // PAD
        vocab.push(vec![]); // BOS
        vocab.push(vec![]); // EOS

        let mut merges = Vec::with_capacity(n);
        for line in lines.take(n) {
            let parts: Vec<u32> = line.split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            if parts.len() != 3 { continue; }
            let (a, b, m) = (parts[0], parts[1], parts[2]);
            // Reconstruct merged token bytes
            let mut new_bytes = vocab[a as usize].clone();
            new_bytes.extend_from_slice(&vocab[b as usize]);
            // Ensure vocab is large enough
            while vocab.len() <= m as usize { vocab.push(vec![]); }
            vocab[m as usize] = new_bytes;
            merges.push((a, b, m));
        }

        let mut byte_to_id = HashMap::new();
        for (id, bytes) in vocab.iter().enumerate() {
            if !bytes.is_empty() {
                byte_to_id.insert(bytes.clone(), id as u32);
            }
        }
        let vocab_size = vocab.len();
        Ok(Self { merges, vocab, byte_to_id, vocab_size })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_train() {
        let corpus = "sinh viên sinh viên đi học đi học thầy giảng bài thầy giảng bài hay";
        let bpe = BPETokenizer::train(corpus, 20);
        assert!(bpe.vocab_size() > BPETokenizer::BASE_VOCAB);
        assert!(bpe.n_merges() > 0);
    }

    #[test]
    fn test_bpe_encode_decode() {
        let corpus = "thầy giảng bài hay thầy giảng bài sinh viên đi học";
        let bpe = BPETokenizer::train(corpus, 30);
        let tokens = bpe.encode("thầy giảng bài");
        let decoded = bpe.decode(&tokens);
        assert!(decoded.contains("thầy"));
        assert!(decoded.contains("giảng"));
    }

    #[test]
    fn test_bpe_compression() {
        let corpus = "sinh viên ".repeat(100) + &"thầy giảng ".repeat(100);
        let bpe = BPETokenizer::train(&corpus, 50);
        let ratio = bpe.compression_ratio("sinh viên thầy giảng");
        // BPE should compress better than byte-level (ratio > 1.0)
        assert!(ratio > 1.0, "BPE should compress better: ratio={ratio}");
    }
}
