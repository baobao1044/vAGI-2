//! Abstraction engine — discovers invariants from microworld data (S4.3).

use vagi_math::Expr;

/// A discovered physical invariant.
#[derive(Clone, Debug)]
pub struct DiscoveredInvariant {
    pub name: String,
    pub expression: Expr,
    pub microworld: String,
    pub conservation_error: f64,
    pub generality: f32,
}

/// Engine for discovering invariants from trajectory data.
pub struct AbstractionEngine {
    pub invariants: Vec<DiscoveredInvariant>,
}

impl AbstractionEngine {
    pub fn new() -> Self {
        Self { invariants: Vec::new() }
    }

    /// Record a discovered invariant.
    pub fn add_invariant(&mut self, inv: DiscoveredInvariant) {
        self.invariants.push(inv);
    }

    /// Number of discovered invariants.
    pub fn invariant_count(&self) -> usize {
        self.invariants.len()
    }

    /// Get invariants with generality above threshold.
    pub fn general_invariants(&self, min_generality: f32) -> Vec<&DiscoveredInvariant> {
        self.invariants.iter()
            .filter(|i| i.generality >= min_generality)
            .collect()
    }

    /// Attempt to unify similar invariants across microworlds.
    pub fn unify_invariants(&mut self) -> usize {
        // Placeholder: in full implementation, would compare expression
        // structures across microworlds and merge equivalent ones.
        0
    }
}

impl Default for AbstractionEngine {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_invariant() {
        let mut engine = AbstractionEngine::new();
        engine.add_invariant(DiscoveredInvariant {
            name: "momentum".into(),
            expression: Expr::var("m").mul(Expr::var("v")),
            microworld: "Collision1D".into(),
            conservation_error: 1e-10,
            generality: 0.8,
        });
        assert_eq!(engine.invariant_count(), 1);
        assert_eq!(engine.general_invariants(0.7).len(), 1);
        assert_eq!(engine.general_invariants(0.9).len(), 0);
    }
}
