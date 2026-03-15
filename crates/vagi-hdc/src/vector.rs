//! HyperVector — 10,240-bit binary hypervectors for HDC.

use rand::Rng;

/// 10,240-bit hypervector, stored as 160 × u64.
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct HyperVector {
    pub data: [u64; 160],
}

impl HyperVector {
    /// Dimension in bits.
    pub const DIM: usize = 10240;
    /// Number of u64 words.
    pub const WORDS: usize = 160;
    /// Byte size for serialization.
    pub const BYTES: usize = 160 * 8; // 1280

    /// Create zero vector.
    pub fn zero() -> Self {
        Self { data: [0u64; Self::WORDS] }
    }

    /// Create random binary vector.
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut data = [0u64; Self::WORDS];
        for d in data.iter_mut() {
            *d = rng.gen();
        }
        Self { data }
    }

    /// XOR binding (self-inverse: a ⊕ b ⊕ b = a).
    pub fn bind(&self, other: &Self) -> Self {
        let mut result = [0u64; Self::WORDS];
        for i in 0..Self::WORDS {
            result[i] = self.data[i] ^ other.data[i];
        }
        Self { data: result }
    }

    /// Hamming distance (number of differing bits).
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        let mut dist = 0u32;
        for i in 0..Self::WORDS {
            dist += (self.data[i] ^ other.data[i]).count_ones();
        }
        dist
    }

    /// Cosine-like similarity: 1.0 - (hamming / total_bits).
    /// 0.5 = random, 1.0 = identical, 0.0 = complement.
    pub fn similarity(&self, other: &Self) -> f32 {
        1.0 - (self.hamming_distance(other) as f32 / Self::DIM as f32)
    }

    /// Population count (number of 1-bits).
    pub fn popcount(&self) -> u32 {
        self.data.iter().map(|d| d.count_ones()).sum()
    }

    /// Majority-rule bundling: merge N vectors.
    ///
    /// For each bit position: output = 1 if majority of inputs have 1.
    /// Ties (even count, exactly half): bit = 0 (deterministic).
    pub fn bundle(vectors: &[&HyperVector]) -> HyperVector {
        let n = vectors.len();
        if n == 0 { return HyperVector::zero(); }
        if n == 1 { return vectors[0].clone(); }

        let threshold = n / 2; // strict majority: count > n/2
        let mut result = [0u64; Self::WORDS];

        for word_idx in 0..Self::WORDS {
            let mut word = 0u64;
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let count: usize = vectors.iter()
                    .filter(|v| v.data[word_idx] & mask != 0)
                    .count();
                if count > threshold {
                    word |= mask;
                }
            }
            result[word_idx] = word;
        }
        HyperVector { data: result }
    }

    /// Cyclic bit permutation by k positions.
    ///
    /// Shifts all 10240 bits cyclically. permute(A, k) then permute(result, -k) = A.
    /// Used for positional encoding in sequences.
    pub fn permute(&self, k: i32) -> HyperVector {
        let total_bits = Self::DIM as i32;
        // Normalize k to positive [0, total_bits)
        let shift = ((k % total_bits) + total_bits) as usize % Self::DIM;
        if shift == 0 { return self.clone(); }

        let mut result = HyperVector::zero();

        for src_bit in 0..Self::DIM {
            let dst_bit = (src_bit + shift) % Self::DIM;
            let src_word = src_bit / 64;
            let src_pos = src_bit % 64;
            let dst_word = dst_bit / 64;
            let dst_pos = dst_bit % 64;

            if self.data[src_word] & (1u64 << src_pos) != 0 {
                result.data[dst_word] |= 1u64 << dst_pos;
            }
        }
        result
    }

    /// Serialize to bytes (little-endian). 160 × 8 = 1280 bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.data.iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::BYTES { return None; }
        let mut data = [0u64; Self::WORDS];
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            data[i] = u64::from_le_bytes(chunk.try_into().ok()?);
        }
        Some(HyperVector { data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_self_inverse() {
        let mut rng = rand::thread_rng();
        let a = HyperVector::random(&mut rng);
        let b = HyperVector::random(&mut rng);
        let result = a.bind(&b).bind(&b);
        assert_eq!(a, result);
    }

    #[test]
    fn test_self_similarity() {
        let mut rng = rand::thread_rng();
        let a = HyperVector::random(&mut rng);
        assert!((a.similarity(&a) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_random_similarity() {
        let mut rng = rand::thread_rng();
        let a = HyperVector::random(&mut rng);
        let b = HyperVector::random(&mut rng);
        let sim = a.similarity(&b);
        // Random vectors should have similarity ≈ 0.5
        assert!((sim - 0.5).abs() < 0.05,
            "Random similarity should be ~0.5, got {sim}");
    }

    #[test]
    fn test_bundle_majority() {
        let mut rng = rand::thread_rng();
        let a = HyperVector::random(&mut rng);
        let b = HyperVector::random(&mut rng);
        // bundle([A, A, B]) should be closer to A than B
        let bundled = HyperVector::bundle(&[&a, &a, &b]);
        let sim_a = bundled.similarity(&a);
        let sim_b = bundled.similarity(&b);
        assert!(sim_a > sim_b,
            "bundle(A,A,B) should be closer to A ({sim_a}) than B ({sim_b})");
        assert!(sim_a > 0.6, "Should be well above random: {sim_a}");
    }

    #[test]
    fn test_bundle_single() {
        let mut rng = rand::thread_rng();
        let a = HyperVector::random(&mut rng);
        let bundled = HyperVector::bundle(&[&a]);
        assert_eq!(a, bundled);
    }

    #[test]
    fn test_bundle_empty() {
        let bundled = HyperVector::bundle(&[]);
        assert_eq!(bundled, HyperVector::zero());
    }

    #[test]
    fn test_permute_inverse() {
        let mut rng = rand::thread_rng();
        let a = HyperVector::random(&mut rng);
        let shifted = a.permute(7);
        let restored = shifted.permute(-7);
        assert_eq!(a, restored, "permute(k) then permute(-k) should be identity");
    }

    #[test]
    fn test_permute_changes_vector() {
        let mut rng = rand::thread_rng();
        let a = HyperVector::random(&mut rng);
        let shifted = a.permute(1);
        assert_ne!(a, shifted, "permute(1) should change the vector");
    }

    #[test]
    fn test_permute_zero_is_identity() {
        let mut rng = rand::thread_rng();
        let a = HyperVector::random(&mut rng);
        assert_eq!(a, a.permute(0));
    }

    #[test]
    fn test_bytes_roundtrip() {
        let mut rng = rand::thread_rng();
        let a = HyperVector::random(&mut rng);
        let bytes = a.to_bytes();
        assert_eq!(bytes.len(), HyperVector::BYTES);
        let b = HyperVector::from_bytes(&bytes).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_bytes_wrong_length() {
        assert!(HyperVector::from_bytes(&[0u8; 100]).is_none());
        assert!(HyperVector::from_bytes(&[]).is_none());
    }

    #[test]
    fn test_popcount() {
        let zero = HyperVector::zero();
        assert_eq!(zero.popcount(), 0);
        let mut rng = rand::thread_rng();
        let r = HyperVector::random(&mut rng);
        let pc = r.popcount();
        // Random vector should have ~50% bits set
        assert!((pc as f32 / HyperVector::DIM as f32 - 0.5).abs() < 0.05,
            "Random popcount ratio should be ~0.5, got {}", pc as f32 / HyperVector::DIM as f32);
    }
}
