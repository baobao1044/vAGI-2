//! Stage 3: Formalize — neural-guided theorem proving training (S4.4).

/// Formalization training result.
#[derive(Clone, Debug, Default)]
pub struct FormalizeResult {
    pub problems_attempted: usize,
    pub problems_solved: usize,
    pub avg_proof_length: f32,
}

impl FormalizeResult {
    pub fn success_rate(&self) -> f32 {
        if self.problems_attempted == 0 { return 0.0; }
        self.problems_solved as f32 / self.problems_attempted as f32
    }
}

/// Problem generator for formalization training.
pub struct ProblemGenerator {
    pub difficulty: f32,
}

impl ProblemGenerator {
    pub fn new(difficulty: f32) -> Self { Self { difficulty } }

    /// Adjust difficulty toward 50-70% success rate.
    pub fn adjust_difficulty(&mut self, success_rate: f32) {
        if success_rate > 0.7 { self.difficulty *= 1.1; }
        else if success_rate < 0.5 { self.difficulty *= 0.9; }
        self.difficulty = self.difficulty.clamp(0.1, 10.0);
    }
}
