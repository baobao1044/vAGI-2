//! Curriculum manager — difficulty scaling for GENESIS training.

/// Manages difficulty progression across training.
pub struct CurriculumManager {
    pub current_difficulty: f32,
    pub success_history: Vec<f32>,
    pub target_success_rate: f32,
    pub window_size: usize,
}

impl CurriculumManager {
    pub fn new() -> Self {
        Self {
            current_difficulty: 0.1,
            success_history: Vec::new(),
            target_success_rate: 0.6,
            window_size: 100,
        }
    }

    /// Record a problem attempt result.
    pub fn record(&mut self, success: bool) {
        self.success_history.push(if success { 1.0 } else { 0.0 });
        if self.success_history.len() > self.window_size {
            self.success_history.remove(0);
        }
    }

    /// Current success rate.
    pub fn success_rate(&self) -> f32 {
        if self.success_history.is_empty() { return 0.0; }
        self.success_history.iter().sum::<f32>() / self.success_history.len() as f32
    }

    /// Adjust difficulty toward target success rate.
    pub fn adjust(&mut self) {
        let rate = self.success_rate();
        if rate > self.target_success_rate + 0.1 {
            self.current_difficulty *= 1.1;
        } else if rate < self.target_success_rate - 0.1 {
            self.current_difficulty *= 0.9;
        }
        self.current_difficulty = self.current_difficulty.clamp(0.01, 100.0);
    }
}

impl Default for CurriculumManager {
    fn default() -> Self { Self::new() }
}
