//! Ternary weight packing and matrix-vector operations.
//!
//! Stores ternary weights {-1, 0, +1} as 2-bit packed values in u64s.
//! Encoding: 00 = 0, 01 = +1, 11 = -1
//! Each u64 holds 32 weights. Rows are 32-weight aligned.
//!
//! Memory: 100M params → ~25MB (vs ~400MB for f32).

/// 2-bit encoding for ternary values.
/// 00 = 0, 01 = +1, 11 = -1
const TERNARY_ZERO: u64 = 0b00;
const TERNARY_POS: u64 = 0b01;
const TERNARY_NEG: u64 = 0b11;

/// Number of weights packed per u64.
const WEIGHTS_PER_U64: usize = 32;

/// A matrix of ternary {-1, 0, +1} weights packed into 2-bit encoding.
///
/// Storage layout: row-major, each row padded to a multiple of 32 weights.
/// Per-row scale factor supports absmax quantization.
#[derive(Clone, Debug)]
pub struct TernaryMatrix {
    /// Packed 2-bit weights. Each u64 holds 32 weights.
    data: Vec<u64>,
    /// Number of rows (output features).
    rows: usize,
    /// Number of columns (input features, logical).
    cols: usize,
    /// Columns padded to multiple of 32.
    cols_padded: usize,
    /// Per-row scale factor for absmax quantization.
    scale: Vec<f32>,
}

impl TernaryMatrix {
    /// Create a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let cols_padded = (cols + WEIGHTS_PER_U64 - 1) / WEIGHTS_PER_U64 * WEIGHTS_PER_U64;
        let u64s_per_row = cols_padded / WEIGHTS_PER_U64;
        Self {
            data: vec![0u64; rows * u64s_per_row],
            rows,
            cols,
            cols_padded,
            scale: vec![1.0; rows],
        }
    }

    /// Number of u64s per row.
    #[inline]
    fn u64s_per_row(&self) -> usize {
        self.cols_padded / WEIGHTS_PER_U64
    }

    /// Pack f32 weights into ternary using absmax threshold.
    ///
    /// threshold = γ × mean(|W_row|)
    /// W > threshold → +1, W < -threshold → -1, else → 0
    /// scale = mean(|W_nonzero|) for reconstruction.
    pub fn pack(weights: &[f32], rows: usize, cols: usize, gamma: f32) -> Self {
        assert_eq!(weights.len(), rows * cols, "Weight count mismatch");
        let cols_padded = (cols + WEIGHTS_PER_U64 - 1) / WEIGHTS_PER_U64 * WEIGHTS_PER_U64;
        let u64s_per_row = cols_padded / WEIGHTS_PER_U64;
        let mut data = vec![0u64; rows * u64s_per_row];
        let mut scale = vec![1.0f32; rows];

        for m in 0..rows {
            let row_start = m * cols;
            let row = &weights[row_start..row_start + cols];

            // Compute threshold = gamma * mean(|w|)
            let abs_mean: f32 = row.iter().map(|w| w.abs()).sum::<f32>() / cols as f32;
            let threshold = gamma * abs_mean;

            // Compute scale = mean of absolute values of non-zero quantized weights
            let mut nonzero_sum = 0.0f32;
            let mut nonzero_count = 0usize;

            let data_start = m * u64s_per_row;
            for n in 0..cols {
                let w = row[n];
                let word_idx = n / WEIGHTS_PER_U64;
                let bit_idx = (n % WEIGHTS_PER_U64) * 2;

                let ternary = if w >= threshold {
                    nonzero_sum += w.abs();
                    nonzero_count += 1;
                    TERNARY_POS
                } else if w <= -threshold {
                    nonzero_sum += w.abs();
                    nonzero_count += 1;
                    TERNARY_NEG
                } else {
                    TERNARY_ZERO
                };
                data[data_start + word_idx] |= ternary << bit_idx;
            }

            scale[m] = if nonzero_count > 0 {
                nonzero_sum / nonzero_count as f32
            } else {
                1.0
            };
        }

        Self { data, rows, cols, cols_padded, scale }
    }

    /// Pack from pre-quantized ternary values (i8: -1, 0, +1).
    pub fn from_ternary(ternary: &[i8], rows: usize, cols: usize) -> Self {
        assert_eq!(ternary.len(), rows * cols);
        let cols_padded = (cols + WEIGHTS_PER_U64 - 1) / WEIGHTS_PER_U64 * WEIGHTS_PER_U64;
        let u64s_per_row = cols_padded / WEIGHTS_PER_U64;
        let mut data = vec![0u64; rows * u64s_per_row];

        for m in 0..rows {
            let data_start = m * u64s_per_row;
            for n in 0..cols {
                let w = ternary[m * cols + n];
                let word_idx = n / WEIGHTS_PER_U64;
                let bit_idx = (n % WEIGHTS_PER_U64) * 2;
                let enc = match w {
                    1 => TERNARY_POS,
                    -1 => TERNARY_NEG,
                    _ => TERNARY_ZERO,
                };
                data[data_start + word_idx] |= enc << bit_idx;
            }
        }

        Self { data, rows, cols, cols_padded, scale: vec![1.0; rows] }
    }

    /// Get single weight at (row, col) as i8 (-1, 0, +1).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> i8 {
        debug_assert!(row < self.rows && col < self.cols);
        let u64s = self.u64s_per_row();
        let word_idx = col / WEIGHTS_PER_U64;
        let bit_idx = (col % WEIGHTS_PER_U64) * 2;
        let bits = (self.data[row * u64s + word_idx] >> bit_idx) & 0b11;
        match bits {
            TERNARY_POS => 1,
            TERNARY_NEG => -1,
            _ => 0,
        }
    }

    /// Unpack a full row to f32 (for reference testing).
    pub fn unpack_row(&self, row: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.cols];
        let s = self.scale[row];
        for n in 0..self.cols {
            out[n] = self.get(row, n) as f32 * s;
        }
        out
    }

    /// Unpack entire matrix to f32.
    pub fn unpack_all(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for m in 0..self.rows {
            let s = self.scale[m];
            for n in 0..self.cols {
                out[m * self.cols + n] = self.get(m, n) as f32 * s;
            }
        }
        out
    }

    /// Memory usage in bytes (data only, not counting Vec overhead).
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 8 + self.scale.len() * 4
    }

    /// Dimensions.
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn scale(&self) -> &[f32] { &self.scale }

    /// Raw packed data (for SIMD kernels).
    pub fn raw_data(&self) -> &[u64] { &self.data }

    /// Get row masks for a specific row: (positive_masks, negative_masks).
    /// Each u64 has bit j set if weight j in that group is +1 (or -1).
    /// Used by SIMD kernels.
    pub fn row_masks(&self, row: usize) -> Vec<(u32, u32)> {
        let u64s = self.u64s_per_row();
        let start = row * u64s;
        (0..u64s).map(|wi| {
            let packed = self.data[start + wi];
            let mut pos_mask = 0u32;
            let mut neg_mask = 0u32;
            for j in 0..WEIGHTS_PER_U64 {
                let bits = (packed >> (j * 2)) & 0b11;
                match bits {
                    TERNARY_POS => pos_mask |= 1 << j,
                    TERNARY_NEG => neg_mask |= 1 << j,
                    _ => {}
                }
            }
            (pos_mask, neg_mask)
        }).collect()
    }
}

// ── Scalar Matvec ─────────────────────────────────────────────

/// Scalar (reference) ternary matrix-vector multiply.
///
/// y[m] = scale[m] × Σ_n W[m,n] × x[n]
///
/// Since W ∈ {-1, 0, +1}, this is addition-only: no float multiplies
/// in the inner loop (only multiply by scale once per row).
pub fn ternary_matvec_scalar(w: &TernaryMatrix, x: &[f32], y: &mut [f32]) {
    assert!(x.len() >= w.cols, "Input too short: {} < {}", x.len(), w.cols);
    assert!(y.len() >= w.rows, "Output too short: {} < {}", y.len(), w.rows);

    let u64s_per_row = w.u64s_per_row();

    for m in 0..w.rows {
        let mut acc = 0.0f32;
        let row_start = m * u64s_per_row;

        for wi in 0..u64s_per_row {
            let packed = w.data[row_start + wi];
            if packed == 0 { continue; } // skip all-zero word

            let base_col = wi * WEIGHTS_PER_U64;
            let end_col = (base_col + WEIGHTS_PER_U64).min(w.cols);

            for j in 0..(end_col - base_col) {
                let bits = (packed >> (j * 2)) & 0b11;
                match bits {
                    TERNARY_POS => acc += x[base_col + j],
                    TERNARY_NEG => acc -= x[base_col + j],
                    _ => {}
                }
            }
        }

        y[m] = acc * w.scale[m];
    }
}

/// Matrix-vector multiply with runtime dispatch.
/// Currently dispatches to scalar; SIMD kernels added later.
pub fn ternary_matvec(w: &TernaryMatrix, x: &[f32], y: &mut [f32]) {
    // TODO: AVX2 / NEON dispatch
    ternary_matvec_scalar(w, x, y);
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m = TernaryMatrix::zeros(4, 8);
        assert_eq!(m.rows(), 4);
        assert_eq!(m.cols(), 8);
        for r in 0..4 { for c in 0..8 { assert_eq!(m.get(r, c), 0); } }
    }

    #[test]
    fn test_from_ternary_roundtrip() {
        let ternary: Vec<i8> = vec![
            1, -1, 0, 1, -1, 0, 1, -1,
            0,  0, 1, 1,  0, -1,-1, 0,
        ];
        let mat = TernaryMatrix::from_ternary(&ternary, 2, 8);
        for m in 0..2 {
            for n in 0..8 {
                assert_eq!(mat.get(m, n), ternary[m * 8 + n],
                    "Mismatch at ({m},{n}): expected {}, got {}",
                    ternary[m * 8 + n], mat.get(m, n));
            }
        }
    }

    #[test]
    fn test_pack_quantization() {
        // Weights [2.0, -1.5, 0.1, 3.0] with gamma=1.0
        // abs_mean = (2+1.5+0.1+3)/4 = 1.65, threshold = 1.65
        // 2.0 > 1.65 → +1, -1.5 < -1.65? No (1.5 < 1.65) → 0,
        // 0.1 → 0, 3.0 → +1
        let weights = vec![2.0, -1.5, 0.1, 3.0];
        let mat = TernaryMatrix::pack(&weights, 1, 4, 1.0);
        assert_eq!(mat.get(0, 0), 1);   // 2.0 > 1.65
        assert_eq!(mat.get(0, 1), 0);   // |-1.5| < 1.65
        assert_eq!(mat.get(0, 2), 0);   // 0.1 < 1.65
        assert_eq!(mat.get(0, 3), 1);   // 3.0 > 1.65
    }

    #[test]
    fn test_pack_negative() {
        let weights = vec![-5.0, -5.0, -5.0, -5.0];
        let mat = TernaryMatrix::pack(&weights, 1, 4, 1.0);
        for n in 0..4 {
            assert_eq!(mat.get(0, n), -1, "All large negatives should be -1");
        }
    }

    #[test]
    fn test_unpack_row() {
        let ternary: Vec<i8> = vec![1, -1, 0, 1];
        let mat = TernaryMatrix::from_ternary(&ternary, 1, 4);
        let row = mat.unpack_row(0);
        assert_eq!(row, vec![1.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_padding() {
        // 3 cols → padded to 32
        let ternary: Vec<i8> = vec![1, -1, 0];
        let mat = TernaryMatrix::from_ternary(&ternary, 1, 3);
        assert_eq!(mat.cols_padded, 32);
        assert_eq!(mat.get(0, 0), 1);
        assert_eq!(mat.get(0, 1), -1);
        assert_eq!(mat.get(0, 2), 0);
    }

    #[test]
    fn test_large_matrix() {
        // 768 × 3072 — typical FFN size
        let ternary: Vec<i8> = (0..768 * 3072)
            .map(|i| match i % 3 { 0 => 1i8, 1 => -1, _ => 0 })
            .collect();
        let mat = TernaryMatrix::from_ternary(&ternary, 768, 3072);
        assert_eq!(mat.rows(), 768);
        assert_eq!(mat.cols(), 3072);
        // Spot check
        assert_eq!(mat.get(0, 0), 1);
        assert_eq!(mat.get(0, 1), -1);
        assert_eq!(mat.get(0, 2), 0);
        assert_eq!(mat.get(767, 3071), match (767 * 3072 + 3071) % 3 {
            0 => 1, 1 => -1, _ => 0
        });
        // Memory: should be ~768 * 96(u64s/row) * 8 = ~589KB
        // vs f32: 768 * 3072 * 4 = ~9.4MB (16× smaller)
        let mem = mat.memory_bytes();
        assert!(mem < 600_000, "Memory should be < 600KB, got {}", mem);
    }

    #[test]
    fn test_memory_savings() {
        // 100M params: rows=10000, cols=10000
        let mat = TernaryMatrix::zeros(10000, 10000);
        let f32_bytes = 10000 * 10000 * 4; // 400MB
        let ternary_bytes = mat.memory_bytes();
        let ratio = f32_bytes as f64 / ternary_bytes as f64;
        assert!(ratio > 12.0, "Ternary should be >12× smaller than f32, got {ratio:.1}×");
    }

    // -- Scalar matvec --

    #[test]
    fn test_matvec_identity_like() {
        // 4×4 identity-ish: diagonal +1, rest 0
        let mut ternary = vec![0i8; 16];
        ternary[0] = 1;   // (0,0)
        ternary[5] = 1;   // (1,1)
        ternary[10] = 1;  // (2,2)
        ternary[15] = 1;  // (3,3)
        let mat = TernaryMatrix::from_ternary(&ternary, 4, 4);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 4];
        ternary_matvec_scalar(&mat, &x, &mut y);
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matvec_add_subtract() {
        // Row: [+1, -1, +1, -1]
        let ternary = vec![1i8, -1, 1, -1];
        let mat = TernaryMatrix::from_ternary(&ternary, 1, 4);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 1];
        ternary_matvec_scalar(&mat, &x, &mut y);
        // 1 - 2 + 3 - 4 = -2
        assert_eq!(y[0], -2.0);
    }

    #[test]
    fn test_matvec_with_scale() {
        let weights = vec![10.0, -10.0, 0.1, 10.0]; // will quantize to [+1,-1,0,+1]
        let mat = TernaryMatrix::pack(&weights, 1, 4, 1.0);
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0f32; 1];
        ternary_matvec_scalar(&mat, &x, &mut y);
        // ternary: +1 -1 0 +1 → sum = 1-1+0+1 = 1
        // scale = mean(|10, 10, 10|) = 10.0
        // y = 10.0 * 1 = 10.0
        assert!((y[0] - 10.0).abs() < 0.01, "Expected ~10.0, got {}", y[0]);
    }

    #[test]
    fn test_matvec_matches_f32_reference() {
        // Compare packed ternary matvec against naive f32 matmul
        let rows = 64;
        let cols = 128;
        let ternary: Vec<i8> = (0..rows * cols)
            .map(|i| match i % 5 { 0 => 1, 1 => -1, 2 => 1, 3 => 0, _ => -1 })
            .collect();
        let mat = TernaryMatrix::from_ternary(&ternary, rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01).collect();

        // Ternary matvec
        let mut y_ternary = vec![0.0f32; rows];
        ternary_matvec_scalar(&mat, &x, &mut y_ternary);

        // f32 reference
        let mut y_ref = vec![0.0f32; rows];
        for m in 0..rows {
            let mut acc = 0.0f32;
            for n in 0..cols {
                acc += ternary[m * cols + n] as f32 * x[n];
            }
            y_ref[m] = acc; // scale is 1.0 for from_ternary
        }

        for m in 0..rows {
            assert!(
                (y_ternary[m] - y_ref[m]).abs() < 1e-4,
                "Row {m}: ternary={}, ref={}", y_ternary[m], y_ref[m]
            );
        }
    }

    #[test]
    fn test_matvec_768x3072() {
        // Realistically-sized: FFN up projection
        let rows = 768;
        let cols = 3072;
        let ternary: Vec<i8> = (0..rows * cols)
            .map(|i| match i % 3 { 0 => 1, 1 => -1, _ => 0 })
            .collect();
        let mat = TernaryMatrix::from_ternary(&ternary, rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.001).sin()).collect();
        let mut y = vec![0.0f32; rows];
        ternary_matvec_scalar(&mat, &x, &mut y);

        // Just verify it finishes and produces finite values
        assert!(y.iter().all(|v| v.is_finite()), "All outputs should be finite");
        assert!(y.iter().any(|v| *v != 0.0), "Not all outputs should be zero");
    }

    #[test]
    fn test_row_masks() {
        let ternary = vec![1i8, -1, 0, 1, 0, 0, -1, 1];
        let mat = TernaryMatrix::from_ternary(&ternary, 1, 8);
        let masks = mat.row_masks(0);
        let (pos, neg) = masks[0]; // first (and only) u64 group
        // pos should have bits 0,3,7 set
        assert!(pos & (1 << 0) != 0);
        assert!(pos & (1 << 3) != 0);
        assert!(pos & (1 << 7) != 0);
        // neg should have bits 1,6 set
        assert!(neg & (1 << 1) != 0);
        assert!(neg & (1 << 6) != 0);
    }
}
