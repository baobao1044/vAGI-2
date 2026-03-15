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

    /// Create zero vector.
    pub fn zero() -> Self {
        Self { data: [0u64; 160] }
    }

    /// Create random binary vector.
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut data = [0u64; 160];
        for d in data.iter_mut() {
            *d = rng.gen();
        }
        Self { data }
    }

    /// XOR binding (self-inverse: a ⊕ b ⊕ b = a).
    pub fn bind(&self, other: &Self) -> Self {
        let mut result = [0u64; 160];
        for i in 0..160 {
            result[i] = self.data[i] ^ other.data[i];
        }
        Self { data: result }
    }

    /// Hamming distance (number of differing bits).
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        let mut dist = 0u32;
        for i in 0..160 {
            dist += (self.data[i] ^ other.data[i]).count_ones();
        }
        dist
    }

    /// Cosine-like similarity: 1.0 - (hamming / total_bits).
    pub fn similarity(&self, other: &Self) -> f32 {
        1.0 - (self.hamming_distance(other) as f32 / Self::DIM as f32)
    }

    /// Population count (number of 1-bits).
    pub fn popcount(&self) -> u32 {
        self.data.iter().map(|d| d.count_ones()).sum()
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
}
