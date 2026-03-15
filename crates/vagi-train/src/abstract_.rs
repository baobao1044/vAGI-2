//! Stage 2: Abstract — symmetry discovery training (S4.3).

/// Abstraction training results.
#[derive(Clone, Debug, Default)]
pub struct AbstractionResult {
    pub n_invariants_found: usize,
    pub avg_conservation_error: f64,
    pub avg_generality: f32,
}
