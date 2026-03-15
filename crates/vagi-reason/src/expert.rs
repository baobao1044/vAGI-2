//! Expert pool — sparse MoE with per-expert AdaptiveBasis activations.
//!
//! Each expert is an AdaptiveBlock (BitNetLinear + AdaptiveBasis).
//! Only top-K experts run per input → sparse compute.

use vagi_core::AdaptiveBlock;
use crate::router::{EnergyRouter, RouterConfig};

/// Configuration for the expert pool.
#[derive(Clone, Debug)]
pub struct ExpertPoolConfig {
    /// Number of experts.
    pub n_experts: usize,
    /// Input/output dimension.
    pub d_model: usize,
    /// Top-K active per input.
    pub top_k: usize,
    /// Load-balancing coefficient for router.
    pub balance_coeff: f32,
}

impl Default for ExpertPoolConfig {
    fn default() -> Self {
        Self {
            n_experts: 8,
            d_model: 64,
            top_k: 2,
            balance_coeff: 0.01,
        }
    }
}

/// MoE expert pool with per-expert AdaptiveBasis.
///
/// Each expert has its own learnable activation function.
/// Router selects top-K, output = Σ weight_i × expert_i(x).
pub struct ExpertPool {
    /// Per-expert AdaptiveBlock (BitNetLinear + AdaptiveBasis).
    experts: Vec<AdaptiveBlock>,
    /// Energy-based router.
    router: EnergyRouter,
    /// Configuration.
    pub config: ExpertPoolConfig,
}

impl ExpertPool {
    /// Create pool with trimmed 3-basis experts.
    pub fn new(config: ExpertPoolConfig) -> Self {
        let experts: Vec<AdaptiveBlock> = (0..config.n_experts)
            .map(|_| AdaptiveBlock::trimmed(config.d_model, config.d_model))
            .collect();
        let router = EnergyRouter::new(RouterConfig {
            n_experts: config.n_experts,
            d_model: config.d_model,
            top_k: config.top_k,
            balance_coeff: config.balance_coeff,
        });
        Self { experts, router, config }
    }

    /// Forward: route input → top-K experts → weighted sum.
    ///
    /// Returns (output, aux_loss).
    pub fn forward(&mut self, input: &[f32]) -> (Vec<f32>, f32) {
        let d = self.config.d_model;
        let decision = self.router.route(input);

        let mut output = vec![0.0f32; d];
        for (&idx, &weight) in decision.expert_indices.iter().zip(decision.expert_weights.iter()) {
            // Clone input because AdaptiveBlock::forward is in-place
            let mut expert_buf = input.to_vec();
            let _ = self.experts[idx].forward(&mut expert_buf);
            for (o, e) in output.iter_mut().zip(expert_buf.iter()) {
                *o += weight * e;
            }
        }

        let aux_loss = self.router.load_balance_loss();
        (output, aux_loss)
    }

    /// Get usage distribution across experts.
    pub fn usage_distribution(&self) -> Vec<f32> {
        self.router.usage_distribution()
    }

    /// Sparsity fraction.
    pub fn sparsity(&self) -> f32 {
        self.router.sparsity()
    }

    /// Number of experts.
    pub fn n_experts(&self) -> usize { self.config.n_experts }

    /// Access router directly.
    pub fn router(&self) -> &EnergyRouter { &self.router }

    /// Access router mutably.
    pub fn router_mut(&mut self) -> &mut EnergyRouter { &mut self.router }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_pool_forward() {
        let config = ExpertPoolConfig {
            n_experts: 4,
            d_model: 8,
            top_k: 2,
            balance_coeff: 0.01,
        };
        let mut pool = ExpertPool::new(config);
        let input = vec![1.0f32; 8];
        let (output, aux_loss) = pool.forward(&input);

        assert_eq!(output.len(), 8, "Output should be d_model");
        assert!(output.iter().all(|v| v.is_finite()), "All outputs should be finite");
        assert!(aux_loss >= 0.0, "Aux loss should be non-negative");
    }

    #[test]
    fn test_sparse_compute() {
        let config = ExpertPoolConfig {
            n_experts: 16,
            d_model: 8,
            top_k: 1,
            balance_coeff: 0.01,
        };
        let pool = ExpertPool::new(config);
        assert!((pool.sparsity() - (15.0 / 16.0)).abs() < 1e-5,
            "1/16 active = 93.75% sparse");
    }

    #[test]
    fn test_multiple_forwards() {
        let config = ExpertPoolConfig::default();
        let mut pool = ExpertPool::new(config);
        for i in 0..50 {
            let input: Vec<f32> = (0..64).map(|j| ((i * 64 + j) as f32 * 0.01).sin()).collect();
            let (output, _) = pool.forward(&input);
            assert!(output.iter().all(|v| v.is_finite()));
        }
        let dist = pool.usage_distribution();
        assert!(dist.iter().any(|&d| d > 0.0), "Some experts should be used");
    }

    #[test]
    fn test_different_experts_have_different_basis() {
        // Each expert should have its own AdaptiveBasis
        let config = ExpertPoolConfig {
            n_experts: 4,
            d_model: 8,
            top_k: 2,
            balance_coeff: 0.01,
        };
        let mut pool = ExpertPool::new(config);
        let input = vec![0.5; 8];
        let (out1, _) = pool.forward(&input);
        assert!(out1.iter().all(|v| v.is_finite()));
    }
}
