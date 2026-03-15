//! Expression encoder — embeds Expr trees into vectors.
//!
//! Uses a simple linear layer instead of BitNetBlock to avoid
//! cross-crate dependency. Will be upgraded to use BitNetBlock
//! when the full neural stack is integrated.

use crate::expr::Expr;
use std::collections::HashMap;

/// Simple linear layer for expression encoding (mock for MVP).
struct SimpleLinear {
    weight: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl SimpleLinear {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        // Xavier-like init
        let scale = (2.0 / (in_dim + out_dim) as f32).sqrt();
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let weight: Vec<f32> = (0..in_dim * out_dim)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self { weight, in_dim, out_dim }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.out_dim];
        for i in 0..self.out_dim {
            let mut sum = 0.0f32;
            for j in 0..self.in_dim.min(input.len()) {
                sum += self.weight[i * self.in_dim + j] * input[j];
            }
            output[i] = sum;
        }
        output
    }
}

/// Embeds expression trees into fixed-size vectors.
pub struct ExprEncoder {
    node_embeddings: HashMap<&'static str, Vec<f32>>,
    tree_encoder: SimpleLinear,
    /// Output dimension.
    pub d_model: usize,
}

impl ExprEncoder {
    pub fn new(d_model: usize) -> Self {
        let types = [
            "Const", "Var", "Symbol", "Add", "Mul", "Neg", "Inv", "Pow",
            "Sin", "Cos", "Exp", "Ln", "Deriv", "Integ", "Eq", "Lt",
        ];
        let mut node_embeddings = HashMap::new();
        let mut rng = rand::thread_rng();
        for name in types {
            use rand::Rng;
            let e: Vec<f32> = (0..d_model).map(|_| rng.gen_range(-0.1..0.1)).collect();
            node_embeddings.insert(name, e);
        }
        let tree_encoder = SimpleLinear::new(3 * d_model, d_model);
        Self { node_embeddings, tree_encoder, d_model }
    }

    /// Hash a string into a deterministic float perturbation vector.
    fn hash_leaf(&self, s: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.d_model];
        let mut h: u64 = 0xcbf29ce484222325; // FNV-1a
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for i in 0..self.d_model {
            h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v[i] = ((h >> 33) as f32) / (u32::MAX as f32) * 0.2 - 0.1;
        }
        v
    }

    /// Encode an expression tree into a fixed-size vector.
    pub fn encode(&self, expr: &Expr) -> Vec<f32> {
        let z = vec![0.0f32; self.d_model];
        // For leaf nodes, produce a unique perturbation from the leaf content
        let leaf_hash = match expr {
            Expr::Const(v) => Some(self.hash_leaf(&format!("{v}"))),
            Expr::Var(name) => Some(self.hash_leaf(name)),
            Expr::Symbol(name) => Some(self.hash_leaf(name)),
            _ => None,
        };
        let (nt, ch) = match expr {
            Expr::Const(_) => ("Const", vec![]),
            Expr::Var(_) => ("Var", vec![]),
            Expr::Symbol(_) => ("Symbol", vec![]),
            Expr::Neg(a) => ("Neg", vec![self.encode(a)]),
            Expr::Inv(a) => ("Inv", vec![self.encode(a)]),
            Expr::Sin(a) => ("Sin", vec![self.encode(a)]),
            Expr::Cos(a) => ("Cos", vec![self.encode(a)]),
            Expr::Exp(a) => ("Exp", vec![self.encode(a)]),
            Expr::Ln(a) => ("Ln", vec![self.encode(a)]),
            Expr::Derivative(a, _) => ("Deriv", vec![self.encode(a)]),
            Expr::Integral(a, _) => ("Integ", vec![self.encode(a)]),
            Expr::Add(a, b) => ("Add", vec![self.encode(a), self.encode(b)]),
            Expr::Mul(a, b) => ("Mul", vec![self.encode(a), self.encode(b)]),
            Expr::Pow(a, b) => ("Pow", vec![self.encode(a), self.encode(b)]),
            Expr::Eq(a, b) => ("Eq", vec![self.encode(a), self.encode(b)]),
            Expr::Lt(a, b) => ("Lt", vec![self.encode(a), self.encode(b)]),
            _ => return z,
        };
        let ne = self.node_embeddings.get(nt).unwrap_or(&z);
        // Mix in leaf hash for atoms (so Var("x") ≠ Var("y"))
        let ne_perturbed: Vec<f32> = if let Some(ref lh) = leaf_hash {
            ne.iter().zip(lh.iter()).map(|(n, l)| n + l).collect()
        } else {
            ne.clone()
        };
        let c1 = ch.first().unwrap_or(&z);
        let c2 = ch.get(1).unwrap_or(&z);
        let mut input = Vec::with_capacity(3 * self.d_model);
        input.extend_from_slice(&ne_perturbed);
        input.extend_from_slice(c1);
        input.extend_from_slice(c2);
        self.tree_encoder.forward(&input)
    }

    /// Cosine similarity between two expression embeddings.
    pub fn similarity(&self, a: &Expr, b: &Expr) -> f32 {
        let ea = self.encode(a);
        let eb = self.encode(b);
        let dot: f32 = ea.iter().zip(eb.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = ea.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = eb.iter().map(|v| v * v).sum::<f32>().sqrt();
        if na < 1e-10 || nb < 1e-10 { 0.0 } else { dot / (na * nb) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_output_dim() {
        let enc = ExprEncoder::new(64);
        let e = enc.encode(&Expr::var("x").add(Expr::num(1.0)));
        assert_eq!(e.len(), 64);
    }

    #[test]
    fn test_deterministic() {
        let enc = ExprEncoder::new(64);
        let expr = Expr::var("x").mul(Expr::num(2.0));
        assert_eq!(enc.encode(&expr), enc.encode(&expr));
    }

    #[test]
    fn test_different_exprs() {
        let enc = ExprEncoder::new(64);
        let e1 = enc.encode(&Expr::var("x"));
        let e2 = enc.encode(&Expr::var("y"));
        assert_ne!(e1, e2);
    }
}
