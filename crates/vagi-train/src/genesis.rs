//! GENESIS scheduler — orchestrates the 5-stage training cycle (S4.7).

use std::time::Duration;

/// Current stage of GENESIS training.
#[derive(Clone, Debug)]
pub enum GenesisStage {
    Embody { tier: usize, progress: f32 },
    Abstract { n_invariants_target: usize },
    Formalize { difficulty: f32 },
    Compose { n_concepts: usize },
    Consolidate { duration: Duration },
}

/// Training metrics used for stage progression.
#[derive(Clone, Debug, Default)]
pub struct TrainingMetrics {
    pub prediction_loss: f32,
    pub n_invariants: usize,
    pub proof_success_rate: f32,
    pub composition_success_rate: f32,
    pub mdl_improvement: f32,
}

/// GENESIS training scheduler.
pub struct GenesisScheduler {
    pub current_cycle: usize,
    pub current_stage: GenesisStage,
    pub max_tiers: usize,
}

impl GenesisScheduler {
    pub fn new() -> Self {
        Self {
            current_cycle: 0,
            current_stage: GenesisStage::Embody { tier: 1, progress: 0.0 },
            max_tiers: 5,
        }
    }

    /// Check if current stage should advance.
    pub fn should_advance(&self, metrics: &TrainingMetrics) -> bool {
        match &self.current_stage {
            GenesisStage::Embody { progress, .. } => *progress > 0.95,
            GenesisStage::Abstract { n_invariants_target } =>
                metrics.n_invariants >= *n_invariants_target,
            GenesisStage::Formalize { .. } => metrics.proof_success_rate > 0.6,
            GenesisStage::Compose { .. } => metrics.composition_success_rate > 0.5,
            GenesisStage::Consolidate { .. } => metrics.mdl_improvement > 0.0,
        }
    }

    /// Advance to next stage.
    pub fn advance(&mut self) {
        self.current_stage = match &self.current_stage {
            GenesisStage::Embody { tier, .. } => GenesisStage::Abstract {
                n_invariants_target: 5 * tier,
            },
            GenesisStage::Abstract { .. } => GenesisStage::Formalize { difficulty: 0.3 },
            GenesisStage::Formalize { .. } => GenesisStage::Compose { n_concepts: 2 },
            GenesisStage::Compose { .. } => GenesisStage::Consolidate {
                duration: Duration::from_secs(3600),
            },
            GenesisStage::Consolidate { .. } => {
                self.current_cycle += 1;
                let next_tier = (self.current_cycle + 1).min(self.max_tiers);
                GenesisStage::Embody { tier: next_tier, progress: 0.0 }
            }
        };
    }

    /// Get a human-readable stage name.
    pub fn stage_name(&self) -> &str {
        match &self.current_stage {
            GenesisStage::Embody { .. } => "Embody",
            GenesisStage::Abstract { .. } => "Abstract",
            GenesisStage::Formalize { .. } => "Formalize",
            GenesisStage::Compose { .. } => "Compose",
            GenesisStage::Consolidate { .. } => "Consolidate",
        }
    }
}

impl Default for GenesisScheduler {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_progression() {
        let mut sched = GenesisScheduler::new();
        assert_eq!(sched.stage_name(), "Embody");

        sched.current_stage = GenesisStage::Embody { tier: 1, progress: 0.96 };
        let m = TrainingMetrics::default();
        assert!(sched.should_advance(&m));

        sched.advance();
        assert_eq!(sched.stage_name(), "Abstract");
    }

    #[test]
    fn test_full_cycle() {
        let mut sched = GenesisScheduler::new();
        for _ in 0..5 { sched.advance(); }
        // After 5 advances: Embody→Abstract→Formalize→Compose→Consolidate→Embody(next cycle)
        assert_eq!(sched.stage_name(), "Embody");
        assert_eq!(sched.current_cycle, 1);
    }
}
