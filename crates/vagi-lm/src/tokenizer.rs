//! Byte-level tokenizer — 259 vocab, zero training required.
//!
//! Token IDs 0-255 = raw UTF-8 bytes.
//! 256 = PAD, 257 = BOS, 258 = EOS.

/// Special token IDs.
pub const PAD_ID: u32 = 256;
pub const BOS_ID: u32 = 257;
pub const EOS_ID: u32 = 258;
pub const VOCAB_SIZE: usize = 259;

/// Byte-level tokenizer. Each UTF-8 byte becomes one token.
#[derive(Clone, Debug)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self { Self }

    /// Encode text to token IDs. Prepends BOS, appends EOS.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::with_capacity(text.len() + 2);
        tokens.push(BOS_ID);
        for byte in text.as_bytes() {
            tokens.push(*byte as u32);
        }
        tokens.push(EOS_ID);
        tokens
    }

    /// Encode without special tokens (for continuation).
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|&b| b as u32).collect()
    }

    /// Decode token IDs back to text. Skips special tokens.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter()
            .filter(|&&t| t < 256)  // skip PAD/BOS/EOS
            .map(|&t| t as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize { VOCAB_SIZE }
}

impl Default for ByteTokenizer {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_ascii() {
        let tok = ByteTokenizer::new();
        let text = "hello world";
        let tokens = tok.encode(text);
        assert_eq!(tokens[0], BOS_ID);
        assert_eq!(*tokens.last().unwrap(), EOS_ID);
        assert_eq!(tokens.len(), text.len() + 2);
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_decode_utf8() {
        let tok = ByteTokenizer::new();
        let text = "xin chào";
        let tokens = tok.encode(text);
        let decoded = tok.decode(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_raw() {
        let tok = ByteTokenizer::new();
        let tokens = tok.encode_raw("hi");
        assert_eq!(tokens, vec![104, 105]);
    }

    #[test]
    fn test_special_tokens_skipped() {
        let tok = ByteTokenizer::new();
        let tokens = vec![BOS_ID, 65, 66, PAD_ID, EOS_ID];
        assert_eq!(tok.decode(&tokens), "AB");
    }

    #[test]
    fn test_vocab_size() {
        assert_eq!(ByteTokenizer::new().vocab_size(), 259);
    }
}
