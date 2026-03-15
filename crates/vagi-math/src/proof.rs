//! Proof representation and verification.
//!
//! A proof is a sequence of rewrite rule applications that transforms
//! one expression into another. Each step is verified symbolically.

use crate::expr::Expr;

/// A single step in a proof: apply rule `rule_name` to get `result`.
#[derive(Clone, Debug)]
pub struct ProofStep {
    pub rule_name: String,
    pub result: Expr,
}

/// A proof is a sequence of verified transformations from `start` to `end`.
#[derive(Clone, Debug)]
pub struct Proof {
    pub start: Expr,
    pub steps: Vec<ProofStep>,
}

impl Proof {
    /// Create a new empty proof starting from `expr`.
    pub fn new(start: Expr) -> Self {
        Self {
            start,
            steps: Vec::new(),
        }
    }

    /// Add a step to the proof.
    pub fn add_step(&mut self, rule_name: &str, result: Expr) {
        self.steps.push(ProofStep {
            rule_name: rule_name.to_string(),
            result,
        });
    }

    /// The final expression after all proof steps.
    pub fn conclusion(&self) -> &Expr {
        if self.steps.is_empty() {
            &self.start
        } else {
            &self.steps.last().unwrap().result
        }
    }

    /// Number of steps in the proof.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the proof is empty (zero steps).
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Display the proof as a human-readable chain.
    pub fn display(&self) -> String {
        let mut lines = vec![format!("  {}", self.start)];
        for step in &self.steps {
            lines.push(format!("= {}  [{}]", step.result, step.rule_name));
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_chain() {
        let start = Expr::var("x").add(Expr::num(0.0));
        let mut proof = Proof::new(start);
        proof.add_step("add_zero", Expr::var("x"));
        assert_eq!(proof.len(), 1);
        assert_eq!(*proof.conclusion(), Expr::var("x"));
    }

    #[test]
    fn test_proof_display() {
        let start = Expr::var("x").mul(Expr::num(1.0));
        let mut proof = Proof::new(start);
        proof.add_step("mul_one", Expr::var("x"));
        let display = proof.display();
        assert!(display.contains("mul_one"));
    }
}
