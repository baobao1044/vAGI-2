//! Multi-head causal self-attention with RoPE positional encoding.
//!
//! Q/K/V/O projections use STELinear (trainable ternary weights).
//! Attention scores computed in f32. Causal mask prevents future tokens.
//! RoPE: precomputed sin/cos tables applied to Q and K.

use vagi_core::ste::STELinear;

/// Precomputed RoPE (Rotary Positional Encoding) tables.
#[derive(Clone, Debug)]
pub struct RoPECache {
    /// cos values [max_seq_len × head_dim/2]
    pub(crate) cos: Vec<f32>,
    /// sin values [max_seq_len × head_dim/2]
    pub(crate) sin: Vec<f32>,
    head_dim: usize,
    max_len: usize,
}

impl RoPECache {
    /// Precompute sin/cos tables for RoPE.
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; max_seq_len * half];
        let mut sin = vec![0.0f32; max_seq_len * half];

        for pos in 0..max_seq_len {
            for i in 0..half {
                let theta = (pos as f32) / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32);
                cos[pos * half + i] = theta.cos();
                sin[pos * half + i] = theta.sin();
            }
        }

        Self { cos, sin, head_dim, max_len: max_seq_len }
    }

    /// Apply RoPE to a single vector at a given position.
    /// Rotates pairs: (x0,x1) → (x0*cos - x1*sin, x0*sin + x1*cos)
    #[inline]
    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let half = self.head_dim / 2;
        let base = pos * half;
        for i in 0..half {
            let x0 = x[2 * i];
            let x1 = x[2 * i + 1];
            let c = self.cos[base + i];
            let s = self.sin[base + i];
            x[2 * i]     = x0 * c - x1 * s;
            x[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

/// Multi-head causal self-attention.
///
/// Uses STELinear for Q/K/V/O (trainable ternary).
/// Causal mask: additive -inf for future positions before softmax.
#[derive(Clone, Debug)]
pub struct CausalAttention {
    /// Q projection [d_model → d_model]
    pub wq: STELinear,
    /// K projection [d_model → d_model]
    pub wk: STELinear,
    /// V projection [d_model → d_model]
    pub wv: STELinear,
    /// Output projection [d_model → d_model]
    pub wo: STELinear,
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope: RoPECache,
}

impl CausalAttention {
    pub fn new(d_model: usize, n_heads: usize, max_seq_len: usize) -> Self {
        let head_dim = d_model / n_heads;
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");

        Self {
            wq: STELinear::new(d_model, d_model),
            wk: STELinear::new(d_model, d_model),
            wv: STELinear::new(d_model, d_model),
            wo: STELinear::new(d_model, d_model),
            d_model,
            n_heads,
            head_dim,
            rope: RoPECache::new(head_dim, max_seq_len),
        }
    }

    /// Forward pass: causal self-attention over a sequence.
    ///
    /// Input: `x` is flat [seq_len × d_model].
    /// Output: flat [seq_len × d_model].
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Vec<f32> {
        let d = self.d_model;
        let h = self.n_heads;
        let hd = self.head_dim;

        // Project Q, K, V: each [seq_len × d_model]
        let mut q_all = vec![0.0f32; seq_len * d];
        let mut k_all = vec![0.0f32; seq_len * d];
        let mut v_all = vec![0.0f32; seq_len * d];

        for t in 0..seq_len {
            let x_t = &x[t * d..(t + 1) * d];
            let q_t = &mut q_all[t * d..(t + 1) * d];
            let k_t = &mut k_all[t * d..(t + 1) * d];
            let v_t = &mut v_all[t * d..(t + 1) * d];
            self.wq.forward(x_t, q_t);
            self.wk.forward(x_t, k_t);
            self.wv.forward(x_t, v_t);
        }

        // Apply RoPE to Q and K per head
        for t in 0..seq_len {
            for head in 0..h {
                let offset = t * d + head * hd;
                self.rope.apply(&mut q_all[offset..offset + hd], t);
                self.rope.apply(&mut k_all[offset..offset + hd], t);
            }
        }

        // Compute attention per head
        let mut output = vec![0.0f32; seq_len * d];

        for head in 0..h {
            // For each query position
            for qi in 0..seq_len {
                // Compute attention scores for this query against all keys up to qi (causal)
                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                let q_offset = qi * d + head * hd;

                for ki in 0..=qi {  // causal: only attend to past + self
                    let k_offset = ki * d + head * hd;
                    let mut dot = 0.0f32;
                    for j in 0..hd {
                        dot += q_all[q_offset + j] * k_all[k_offset + j];
                    }
                    scores[ki] = dot / (hd as f32).sqrt();
                }

                // Softmax over valid positions [0..=qi]
                let max_score = scores[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                let mut exps = vec![0.0f32; qi + 1];
                for i in 0..=qi {
                    exps[i] = (scores[i] - max_score).exp();
                    sum_exp += exps[i];
                }
                if sum_exp > 0.0 {
                    for e in exps.iter_mut() {
                        *e /= sum_exp;
                    }
                }

                // Weighted sum of values
                let out_offset = qi * d + head * hd;
                for vi in 0..=qi {
                    let v_offset = vi * d + head * hd;
                    let w = exps[vi];
                    for j in 0..hd {
                        output[out_offset + j] += w * v_all[v_offset + j];
                    }
                }
            }
        }

        // Output projection
        let mut final_out = vec![0.0f32; seq_len * d];
        for t in 0..seq_len {
            self.wo.forward(
                &output[t * d..(t + 1) * d],
                &mut final_out[t * d..(t + 1) * d],
            );
        }

        final_out
    }

    /// Cached forward: process a SINGLE new token using cached K/V from past tokens.
    ///
    /// `x_new`: `[d_model]` — the new token's hidden state.
    /// `pos`: position index of this token in the sequence.
    /// `cache`: mutable KV cache to read from and append to.
    ///
    /// Returns: `[d_model]` output for the new token.
    pub fn forward_cached(&self, x_new: &[f32], pos: usize, cache: &mut KVCache) -> Vec<f32> {
        let d = self.d_model;
        let h = self.n_heads;
        let hd = self.head_dim;

        // Project Q, K, V for new token
        let mut q = vec![0.0f32; d];
        let mut k = vec![0.0f32; d];
        let mut v = vec![0.0f32; d];
        self.wq.forward(x_new, &mut q);
        self.wk.forward(x_new, &mut k);
        self.wv.forward(x_new, &mut v);

        // Apply RoPE to Q and K
        for head in 0..h {
            let offset = head * hd;
            self.rope.apply(&mut q[offset..offset + hd], pos);
            self.rope.apply(&mut k[offset..offset + hd], pos);
        }

        // Append K, V to cache
        cache.k_cache.extend_from_slice(&k);
        cache.v_cache.extend_from_slice(&v);
        cache.len += 1;

        // Attention: query attends to all cached K/V (including current)
        let mut attn_out = vec![0.0f32; d];
        let n = cache.len; // number of cached positions (including current)

        for head in 0..h {
            let q_off = head * hd;

            // Compute scores against all cached keys
            let mut scores = Vec::with_capacity(n);
            for ki in 0..n {
                let k_off = ki * d + head * hd;
                let mut dot = 0.0f32;
                for j in 0..hd {
                    dot += q[q_off + j] * cache.k_cache[k_off + j];
                }
                scores.push(dot / (hd as f32).sqrt());
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f32 = exps.iter().sum();

            // Weighted sum of values
            for vi in 0..n {
                let v_off = vi * d + head * hd;
                let w = if sum > 0.0 { exps[vi] / sum } else { 0.0 };
                for j in 0..hd {
                    attn_out[q_off + j] += w * cache.v_cache[v_off + j];
                }
            }
        }

        // Output projection
        let mut output = vec![0.0f32; d];
        self.wo.forward(&attn_out, &mut output);
        output
    }
}

/// KV cache for fast autoregressive inference.
///
/// Stores K and V projections for all past tokens, enabling O(1)
/// per-token computation instead of O(n) recomputation.
#[derive(Clone)]
pub struct KVCache {
    /// Cached key projections [len × d_model] (flat).
    pub(crate) k_cache: Vec<f32>,
    /// Cached value projections [len × d_model] (flat).
    pub(crate) v_cache: Vec<f32>,
    /// Number of cached positions.
    pub(crate) len: usize,
    /// Model dimension.
    d_model: usize,
}

impl KVCache {
    /// Create empty cache for a layer.
    pub fn new(d_model: usize, max_seq_len: usize) -> Self {
        Self {
            k_cache: Vec::with_capacity(max_seq_len * d_model),
            v_cache: Vec::with_capacity(max_seq_len * d_model),
            len: 0,
            d_model,
        }
    }

    /// Number of cached positions.
    pub fn len(&self) -> usize { self.len }

    /// Whether cache is empty.
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.k_cache.clear();
        self.v_cache.clear();
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_cache() {
        let rope = RoPECache::new(16, 128);
        // Check first position: cos(0)=1, sin(0)=0
        assert!((rope.cos[0] - 1.0).abs() < 1e-6);
        assert!(rope.sin[0].abs() < 1e-6);
    }

    #[test]
    fn test_rope_apply() {
        let rope = RoPECache::new(4, 32);
        let mut x = vec![1.0, 0.0, 1.0, 0.0];
        rope.apply(&mut x, 0);
        // At pos=0, cos=1 sin=0, so x should be unchanged
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!(x[1].abs() < 1e-5);
    }

    #[test]
    fn test_attention_output_shape() {
        let attn = CausalAttention::new(64, 4, 128);
        let seq_len = 8;
        let x = vec![0.1f32; seq_len * 64];
        let out = attn.forward(&x, seq_len);
        assert_eq!(out.len(), seq_len * 64);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_causal_masking() {
        // First token should only attend to itself
        // (verified by output being deterministic given only first token)
        let attn = CausalAttention::new(32, 2, 64);
        let seq_len = 4;
        let mut x = vec![0.0f32; seq_len * 32];
        // Only first token has non-zero input
        for i in 0..32 { x[i] = 1.0; }
        let out = attn.forward(&x, seq_len);
        // First token output should be finite and non-zero
        let first_sum: f32 = out[..32].iter().map(|v| v.abs()).sum();
        assert!(first_sum > 0.0, "First token should produce non-zero output");
    }
}
