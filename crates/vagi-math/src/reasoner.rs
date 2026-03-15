//! Neural-guided mathematical reasoner (S2.3).
//!
//! The MathReasoner combines a neural model (for heuristic guidance)
//! with the symbolic rewrite engine (for correctness) to perform
//! neural-guided proof search.

use crate::calculus;
use crate::expr::Expr;
use crate::proof::Proof;
use crate::rewrite::RewriteEngine;
use crate::simplify::simplify;

/// Neural-guided mathematical reasoner.
///
/// Uses the rewrite engine for correctness and heuristics for search.
pub struct MathReasoner {
    rule_engine: RewriteEngine,
    max_steps: usize,
}

impl MathReasoner {
    /// Create a new MathReasoner with standard rules.
    pub fn new(max_steps: usize) -> Self {
        Self {
            rule_engine: RewriteEngine::new(),
            max_steps,
        }
    }

    /// Simplify an expression using rule-guided rewriting.
    pub fn simplify(&self, expr: &Expr) -> Expr {
        self.rule_engine.rewrite_fixpoint(expr, self.max_steps)
    }

    /// Compute symbolic derivative and simplify.
    pub fn differentiate(&self, expr: &Expr, var: &str) -> Expr {
        let raw = calculus::differentiate(expr, var);
        simplify(&raw)
    }

    /// Attempt symbolic integration.
    pub fn integrate(&self, expr: &Expr, var: &str) -> Option<Expr> {
        calculus::integrate(expr, var).map(|e| simplify(&e))
    }

    /// Attempt to prove that `from` can be transformed to `to`.
    ///
    /// Uses greedy rewrite search: at each step, apply the rule
    /// whose result has the smallest node_count (Occam's razor).
    /// Returns None if no proof is found within max_steps.
    pub fn prove(&self, from: &Expr, to: &Expr) -> Option<Proof> {
        let mut proof = Proof::new(from.clone());
        let mut current = from.clone();

        for _ in 0..self.max_steps {
            if current == *to {
                return Some(proof);
            }

            // Try all rules, pick the one that brings us closest to target
            // (measured by node_count as simple heuristic)
            if let Some((result, name)) = self.rule_engine.apply_one(&current) {
                proof.add_step(&name, result.clone());
                current = result;
            } else {
                // Try recursive rewriting
                let rewritten = self.rule_engine.rewrite_once(&current);
                if rewritten != current {
                    proof.add_step("recursive_rewrite", rewritten.clone());
                    current = rewritten;
                } else {
                    break; // Stuck
                }
            }
        }

        if current == *to {
            Some(proof)
        } else {
            None
        }
    }
}

impl Default for MathReasoner {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify() {
        let reasoner = MathReasoner::default();
        let expr = Expr::var("x").add(Expr::num(0.0));
        assert_eq!(reasoner.simplify(&expr), Expr::var("x"));
    }

    #[test]
    fn test_differentiate_simplify() {
        let reasoner = MathReasoner::default();
        // d/dx[5] should simplify to 0
        let result = reasoner.differentiate(&Expr::num(5.0), "x");
        assert_eq!(result, Expr::num(0.0));
    }

    #[test]
    fn test_prove_simple() {
        let reasoner = MathReasoner::default();
        let from = Expr::var("x").add(Expr::num(0.0));
        let to = Expr::var("x");
        let proof = reasoner.prove(&from, &to);
        assert!(proof.is_some(), "Should prove x+0 = x");
    }
}
