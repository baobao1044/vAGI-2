//! Stage 4: Compose — compositional problem solving (S4.5).

/// Composition training result.
#[derive(Clone, Debug, Default)]
pub struct ComposeResult {
    pub problems_attempted: usize,
    pub problems_solved: usize,
    pub avg_concepts_combined: f32,
}

impl ComposeResult {
    pub fn success_rate(&self) -> f32 {
        if self.problems_attempted == 0 { return 0.0; }
        self.problems_solved as f32 / self.problems_attempted as f32
    }
}
