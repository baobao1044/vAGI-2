//! OODA Loop — Observe, Orient, Decide, Act runtime agent.
//!
//! Wires together all vAGI layers into a single processing loop:
//! - Observe: ingest raw input, encode via HDC
//! - Orient: update streaming state, query memory
//! - Decide: route through MoE experts, apply predictive gate
//! - Act: produce output, update world model

use vagi_memory::StreamingState;
use vagi_reason::{ExpertPool, ExpertPoolConfig, PredictiveGate, PredictiveGateConfig};

/// OODA loop configuration.
#[derive(Clone, Debug)]
pub struct OODAConfig {
    /// Model dimension.
    pub d_model: usize,
    /// Number of MoE experts.
    pub n_experts: usize,
    /// Top-K active experts.
    pub top_k: usize,
}

impl Default for OODAConfig {
    fn default() -> Self {
        Self {
            d_model: 64,
            n_experts: 8,
            top_k: 2,
        }
    }
}

/// Metrics from a single OODA cycle.
#[derive(Clone, Debug)]
pub struct CycleMetrics {
    /// Surprise from predictive gate.
    pub surprise: f32,
    /// Gate value (0=predicted, 1=novel).
    pub gate_value: f32,
    /// MoE auxiliary loss.
    pub aux_loss: f32,
    /// Total cycles completed.
    pub cycle_count: u64,
}

/// OODA Loop agent — the main runtime pipeline.
pub struct OODALoop {
    /// Streaming state for temporal context.
    state: StreamingState,
    /// Expert pool (sparse MoE).
    experts: ExpertPool,
    /// Predictive coding gate.
    gate: PredictiveGate,
    /// Configuration.
    config: OODAConfig,
    /// Total cycles.
    cycle_count: u64,
}

impl OODALoop {
    /// Create a new OODA loop agent.
    pub fn new(config: OODAConfig) -> Self {
        let state = StreamingState::new(config.d_model);
        let experts = ExpertPool::new(ExpertPoolConfig {
            n_experts: config.n_experts,
            d_model: config.d_model,
            top_k: config.top_k,
            balance_coeff: 0.01,
        });
        let gate = PredictiveGate::new(PredictiveGateConfig {
            d_model: config.d_model,
            ..Default::default()
        });
        Self { state, experts, gate, config, cycle_count: 0 }
    }

    /// Run one OODA cycle:
    ///
    /// 1. **Observe**: update streaming state with input
    /// 2. **Orient**: get multi-scale context from streaming state
    /// 3. **Decide**: route through MoE experts, gate output
    /// 4. **Act**: return processed output
    pub fn cycle(&mut self, input: &[f32]) -> (Vec<f32>, CycleMetrics) {
        let d = self.config.d_model;

        // 1. Observe: update streaming state
        self.state.update(input);

        // 2. Orient: use L0 (word-level) state as oriented input
        let oriented = self.state.level_state(0).unwrap_or(input).to_vec();

        // 3. Decide: route through MoE experts
        let (expert_out, aux_loss) = self.experts.forward(&oriented);

        // 4. Apply predictive coding gate
        let (gated_out, surprise, gate_value) = self.gate.forward(&expert_out);

        self.cycle_count += 1;

        let metrics = CycleMetrics {
            surprise,
            gate_value,
            aux_loss,
            cycle_count: self.cycle_count,
        };

        (gated_out, metrics)
    }

    /// Run multiple cycles.
    pub fn run_batch(&mut self, inputs: &[Vec<f32>]) -> Vec<(Vec<f32>, CycleMetrics)> {
        inputs.iter().map(|input| self.cycle(input)).collect()
    }

    /// Get total cycle count.
    pub fn cycle_count(&self) -> u64 { self.cycle_count }

    /// Get expert usage distribution.
    pub fn expert_usage(&self) -> Vec<f32> { self.experts.usage_distribution() }

    /// Get average surprise.
    pub fn average_surprise(&self) -> f32 { self.gate.average_surprise() }

    /// Get streaming state for inspection.
    pub fn streaming_state(&self) -> &StreamingState { &self.state }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.state.reset();
        self.gate.reset();
        self.experts.router_mut().reset_usage();
        self.cycle_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ooda_basic_cycle() {
        let mut agent = OODALoop::new(OODAConfig::default());
        let input = vec![1.0f32; 64];
        let (output, metrics) = agent.cycle(&input);

        assert_eq!(output.len(), 64);
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(metrics.surprise >= 0.0);
        assert!(metrics.gate_value >= 0.0 && metrics.gate_value <= 1.0);
        assert_eq!(metrics.cycle_count, 1);
    }

    #[test]
    fn test_ooda_multiple_cycles() {
        let mut agent = OODALoop::new(OODAConfig::default());
        for i in 0..100 {
            let input: Vec<f32> = (0..64).map(|j| ((i * 64 + j) as f32 * 0.01).sin()).collect();
            let (output, metrics) = agent.cycle(&input);
            assert!(output.iter().all(|v| v.is_finite()));
            assert!(metrics.surprise.is_finite());
        }
        assert_eq!(agent.cycle_count(), 100);
    }

    #[test]
    fn test_surprise_detection() {
        let mut agent = OODALoop::new(OODAConfig::default());
        // Feed constant input to establish low surprise
        for _ in 0..50 {
            agent.cycle(&vec![1.0; 64]);
        }
        let (_, stable_metrics) = agent.cycle(&vec![1.0; 64]);

        // Feed novel input → should spike surprise
        let (_, novel_metrics) = agent.cycle(&vec![10.0; 64]);
        eprintln!("Stable surprise: {}, Novel surprise: {}",
            stable_metrics.surprise, novel_metrics.surprise);
        assert!(novel_metrics.surprise > stable_metrics.surprise,
            "Novel input should increase surprise");
    }

    #[test]
    fn test_expert_usage() {
        let mut agent = OODALoop::new(OODAConfig::default());
        for i in 0..50 {
            let input: Vec<f32> = (0..64).map(|j| ((i * 64 + j) as f32 * 0.1).sin()).collect();
            agent.cycle(&input);
        }
        let usage = agent.expert_usage();
        assert_eq!(usage.len(), 8);
        assert!(usage.iter().any(|&u| u > 0.0), "Some experts should be used");
    }

    #[test]
    fn test_batch_run() {
        let mut agent = OODALoop::new(OODAConfig::default());
        let inputs: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..64).map(|j| (i * 64 + j) as f32 * 0.01).collect())
            .collect();
        let results = agent.run_batch(&inputs);
        assert_eq!(results.len(), 10);
        assert_eq!(agent.cycle_count(), 10);
    }

    #[test]
    fn test_reset() {
        let mut agent = OODALoop::new(OODAConfig::default());
        for _ in 0..10 {
            agent.cycle(&vec![1.0; 64]);
        }
        assert_eq!(agent.cycle_count(), 10);
        agent.reset();
        assert_eq!(agent.cycle_count(), 0);
    }
}
