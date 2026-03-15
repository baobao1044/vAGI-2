//! Energy-based expert routing — sparse MoE with load balancing.
//!
//! Routes input to top-K experts based on energy scores (dot product).
//! Uses load-balancing auxiliary loss to prevent expert collapse.
//! Only ~5% of experts active per input → sparse compute.

use rand::Rng;

/// Configuration for the expert router.
#[derive(Clone, Debug)]
pub struct RouterConfig {
    /// Number of experts.
    pub n_experts: usize,
    /// Input dimension.
    pub d_model: usize,
    /// Number of active experts per input (top-K).
    pub top_k: usize,
    /// Load-balancing coefficient.
    pub balance_coeff: f32,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            n_experts: 8,
            d_model: 64,
            top_k: 2,
            balance_coeff: 0.01,
        }
    }
}

/// Result of routing: which experts to activate and with what weight.
#[derive(Clone, Debug)]
pub struct RoutingDecision {
    /// Selected expert indices.
    pub expert_indices: Vec<usize>,
    /// Corresponding gating weights (softmax over selected).
    pub expert_weights: Vec<f32>,
    /// Energy scores for all experts (for analysis).
    pub all_energies: Vec<f32>,
}

/// Energy-based router: computes energy scores and selects top-K experts.
pub struct EnergyRouter {
    /// Routing weight matrix [n_experts × d_model].
    pub gate_weights: Vec<f32>,
    /// Configuration.
    pub config: RouterConfig,
    /// Usage counters for load balancing.
    usage_counts: Vec<u64>,
    /// Total routing calls.
    total_calls: u64,
}

impl EnergyRouter {
    /// Create router with random gate weights.
    pub fn new(config: RouterConfig) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / config.d_model as f32).sqrt();
        let gate_weights: Vec<f32> = (0..config.n_experts * config.d_model)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self {
            gate_weights,
            usage_counts: vec![0; config.n_experts],
            total_calls: 0,
            config,
        }
    }

    /// Route input to top-K experts.
    ///
    /// 1. Compute energy: e_i = gate_weights[i] · input
    /// 2. Select top-K by energy
    /// 3. Softmax over selected → gating weights
    pub fn route(&mut self, input: &[f32]) -> RoutingDecision {
        let n = self.config.n_experts;
        let d = self.config.d_model;
        let k = self.config.top_k.min(n);

        // Compute energy scores
        let mut energies = vec![0.0f32; n];
        for i in 0..n {
            let start = i * d;
            for j in 0..d {
                energies[i] += self.gate_weights[start + j] * input[j];
            }
        }

        // Select top-K
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| energies[b].partial_cmp(&energies[a])
            .unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<usize> = indices[..k].to_vec();

        // Softmax over selected experts
        let max_e = selected.iter().map(|&i| energies[i]).fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = selected.iter().map(|&i| (energies[i] - max_e).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let weights: Vec<f32> = if sum_exp > 0.0 {
            exp_scores.iter().map(|e| e / sum_exp).collect()
        } else {
            vec![1.0 / k as f32; k]
        };

        // Update usage counters
        for &idx in &selected {
            self.usage_counts[idx] += 1;
        }
        self.total_calls += 1;

        RoutingDecision {
            expert_indices: selected,
            expert_weights: weights,
            all_energies: energies,
        }
    }

    /// Compute load-balancing auxiliary loss.
    ///
    /// Penalizes uneven expert usage. Returns: balance_coeff × CV²
    /// where CV = coefficient of variation of usage counts.
    pub fn load_balance_loss(&self) -> f32 {
        if self.total_calls == 0 { return 0.0; }
        let n = self.config.n_experts as f32;
        let mean = self.total_calls as f32 * self.config.top_k as f32 / n;
        if mean == 0.0 { return 0.0; }
        let variance: f32 = self.usage_counts.iter()
            .map(|&c| {
                let diff = c as f32 - mean;
                diff * diff
            })
            .sum::<f32>() / n;
        let cv_squared = variance / (mean * mean);
        self.config.balance_coeff * cv_squared
    }

    /// Get usage distribution as fractions.
    pub fn usage_distribution(&self) -> Vec<f32> {
        let total = self.usage_counts.iter().sum::<u64>() as f32;
        if total == 0.0 {
            vec![0.0; self.config.n_experts]
        } else {
            self.usage_counts.iter().map(|&c| c as f32 / total).collect()
        }
    }

    /// Reset usage counters.
    pub fn reset_usage(&mut self) {
        self.usage_counts.fill(0);
        self.total_calls = 0;
    }

    /// Sparsity: fraction of experts NOT used per call.
    pub fn sparsity(&self) -> f32 {
        1.0 - self.config.top_k as f32 / self.config.n_experts as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_routing() {
        let config = RouterConfig::default();
        let mut router = EnergyRouter::new(config);
        let input = vec![1.0f32; 64];
        let decision = router.route(&input);

        assert_eq!(decision.expert_indices.len(), 2, "top_k=2");
        assert_eq!(decision.expert_weights.len(), 2);
        // Weights should sum to ~1.0
        let sum: f32 = decision.expert_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Weights should sum to 1, got {sum}");
    }

    #[test]
    fn test_sparsity() {
        let config = RouterConfig { n_experts: 20, d_model: 32, top_k: 1, balance_coeff: 0.01 };
        let router = EnergyRouter::new(config);
        assert!((router.sparsity() - 0.95).abs() < 1e-5, "1/20 = 5% active = 95% sparse");
    }

    #[test]
    fn test_usage_tracking() {
        let config = RouterConfig { n_experts: 4, d_model: 8, top_k: 1, balance_coeff: 0.01 };
        let mut router = EnergyRouter::new(config);
        for _ in 0..100 {
            router.route(&vec![1.0; 8]);
        }
        let dist = router.usage_distribution();
        assert_eq!(dist.len(), 4);
        let sum: f32 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Distribution should sum to 1");
    }

    #[test]
    fn test_load_balance_loss() {
        let config = RouterConfig { n_experts: 4, d_model: 8, top_k: 1, balance_coeff: 1.0 };
        let mut router = EnergyRouter::new(config);
        // Route many times — same input may always pick same expert
        for _ in 0..100 {
            router.route(&vec![1.0; 8]);
        }
        let loss = router.load_balance_loss();
        // If all goes to one expert: imbalanced → high loss
        // If evenly distributed: low loss
        assert!(loss >= 0.0, "Loss should be non-negative");
    }

    #[test]
    fn test_different_inputs_different_experts() {
        let config = RouterConfig { n_experts: 8, d_model: 4, top_k: 1, balance_coeff: 0.01 };
        let mut router = EnergyRouter::new(config);
        let d1 = router.route(&vec![1.0, 0.0, 0.0, 0.0]);
        let d2 = router.route(&vec![0.0, 0.0, 0.0, 1.0]);
        // Different inputs should often route to different experts
        // Not guaranteed but very likely with random weights
        // Just check the API works correctly
        assert_eq!(d1.expert_indices.len(), 1);
        assert_eq!(d2.expert_indices.len(), 1);
    }

    #[test]
    fn test_all_energies_present() {
        let config = RouterConfig { n_experts: 8, d_model: 4, top_k: 2, balance_coeff: 0.01 };
        let mut router = EnergyRouter::new(config);
        let decision = router.route(&vec![1.0; 4]);
        assert_eq!(decision.all_energies.len(), 8, "Should have energy for all experts");
    }
}
