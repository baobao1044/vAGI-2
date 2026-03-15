//! Elastic Weight Consolidation — prevents catastrophic forgetting (S4.2).

/// EWC regularizer.
pub struct EWCRegularizer {
    /// Fisher information diagonal.
    pub fisher_diag: Vec<f32>,
    /// Reference parameters from previous task.
    pub reference_params: Vec<f32>,
    /// Regularization strength.
    pub lambda: f32,
}

impl EWCRegularizer {
    pub fn new(n_params: usize, lambda: f32) -> Self {
        Self {
            fisher_diag: vec![0.0; n_params],
            reference_params: vec![0.0; n_params],
            lambda,
        }
    }

    /// EWC penalty: λ/2 * Σ_i F_i * (θ_i - θ*_i)².
    pub fn penalty(&self, current_params: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.fisher_diag.len().min(current_params.len()) {
            let diff = current_params[i] - self.reference_params[i];
            sum += self.fisher_diag[i] * diff * diff;
        }
        self.lambda * 0.5 * sum
    }

    /// Snapshot current parameters as reference.
    pub fn snapshot(&mut self, params: &[f32]) {
        self.reference_params = params.to_vec();
    }

    /// Merge Fisher from new task (running average).
    pub fn merge_fisher(&mut self, new_fisher: &[f32], alpha: f32) {
        for i in 0..self.fisher_diag.len().min(new_fisher.len()) {
            self.fisher_diag[i] = (1.0 - alpha) * self.fisher_diag[i] + alpha * new_fisher[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ewc_penalty_zero_at_reference() {
        let ewc = EWCRegularizer::new(10, 1.0);
        let params = vec![0.0; 10]; // same as reference
        assert!((ewc.penalty(&params) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ewc_penalty_increases() {
        let mut ewc = EWCRegularizer::new(10, 1.0);
        ewc.fisher_diag = vec![1.0; 10];
        let params1 = vec![0.1; 10];
        let params2 = vec![0.5; 10];
        assert!(ewc.penalty(&params2) > ewc.penalty(&params1));
    }
}
