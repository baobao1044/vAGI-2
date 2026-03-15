//! Stage 5: Consolidate — sleep & compress (S4.6).

/// Consolidation/pruning report.
#[derive(Clone, Debug, Default)]
pub struct ConsolidateReport {
    pub params_pruned: usize,
    pub rules_compressed: usize,
    pub mdl_before: f64,
    pub mdl_after: f64,
    pub dreams_replayed: usize,
}

impl ConsolidateReport {
    pub fn mdl_improvement(&self) -> f64 {
        if self.mdl_before == 0.0 { return 0.0; }
        (self.mdl_before - self.mdl_after) / self.mdl_before
    }
}
