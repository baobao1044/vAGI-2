//! Predictive coding gate — surprise-driven information filtering.
//!
//! Predicts next state from current state, computes surprise (prediction error),
//! and gates how much new information passes. High surprise → gate opens → learn more.
//! Low surprise → gate closes → rely on prediction.

/// Configuration for predictive coding gate.
#[derive(Clone, Debug)]
pub struct PredictiveGateConfig {
    /// Model dimension.
    pub d_model: usize,
    /// Surprise threshold: below this → gate mostly closed.
    pub surprise_threshold: f32,
    /// Gate temperature: higher → sharper gating.
    pub temperature: f32,
}

impl Default for PredictiveGateConfig {
    fn default() -> Self {
        Self {
            d_model: 64,
            surprise_threshold: 0.1,
            temperature: 5.0,
        }
    }
}

/// Predictive coding gate.
///
/// Maintains a prediction of the next input. Uses prediction error (surprise)
/// to gate how much new information to accept vs rely on prediction.
pub struct PredictiveGate {
    /// Previous state (used for prediction).
    prev_state: Vec<f32>,
    /// Prediction weights [d_model × d_model] — simple linear predictor.
    pred_weights: Vec<f32>,
    /// Bias [d_model].
    pred_bias: Vec<f32>,
    /// Running average of surprise for normalization.
    surprise_ema: f32,
    /// Total calls.
    total_calls: u64,
    /// Configuration.
    pub config: PredictiveGateConfig,
}

impl PredictiveGate {
    /// Create with identity-like prediction (initially predicts same state).
    pub fn new(config: PredictiveGateConfig) -> Self {
        let d = config.d_model;
        // Initialize prediction weights near identity
        let mut pred_weights = vec![0.0f32; d * d];
        for i in 0..d {
            pred_weights[i * d + i] = 0.9; // ~identity
        }
        Self {
            prev_state: vec![0.0; d],
            pred_weights,
            pred_bias: vec![0.0; d],
            surprise_ema: 0.1,
            total_calls: 0,
            config,
        }
    }

    /// Predict next state from previous state.
    fn predict(&self) -> Vec<f32> {
        let d = self.config.d_model;
        let mut prediction = self.pred_bias.clone();
        for i in 0..d {
            for j in 0..d {
                prediction[i] += self.pred_weights[i * d + j] * self.prev_state[j];
            }
        }
        prediction
    }

    /// Compute surprise: normalized L2 prediction error.
    fn compute_surprise(&self, actual: &[f32], predicted: &[f32]) -> f32 {
        let d = actual.len() as f32;
        let mse: f32 = actual.iter().zip(predicted.iter())
            .map(|(a, p)| (a - p) * (a - p))
            .sum::<f32>() / d;
        mse.sqrt() // RMSE
    }

    /// Compute gate value from surprise.
    ///
    /// gate = σ(temperature × (surprise - threshold))
    /// High surprise → gate → 1 (pass new info)
    /// Low surprise → gate → 0 (rely on prediction)
    fn gate_value(&self, surprise: f32) -> f32 {
        let x = self.config.temperature * (surprise - self.config.surprise_threshold);
        1.0 / (1.0 + (-x).exp()) // sigmoid
    }

    /// Forward pass: gate actual input vs prediction.
    ///
    /// output = gate × actual + (1 - gate) × prediction
    ///
    /// Returns (output, surprise, gate_value).
    pub fn forward(&mut self, actual: &[f32]) -> (Vec<f32>, f32, f32) {
        let predicted = self.predict();
        let surprise = self.compute_surprise(actual, &predicted);
        let gate = self.gate_value(surprise);

        // Update surprise EMA
        let alpha = 0.1;
        self.surprise_ema = (1.0 - alpha) * self.surprise_ema + alpha * surprise;

        // Gated output
        let output: Vec<f32> = actual.iter().zip(predicted.iter())
            .map(|(&a, &p)| gate * a + (1.0 - gate) * p)
            .collect();

        // Update state for next prediction
        self.prev_state = actual.to_vec();
        self.total_calls += 1;

        (output, surprise, gate)
    }

    /// Get running average of surprise.
    pub fn average_surprise(&self) -> f32 { self.surprise_ema }

    /// Total forward calls.
    pub fn total_calls(&self) -> u64 { self.total_calls }

    /// Reset state.
    pub fn reset(&mut self) {
        self.prev_state.fill(0.0);
        self.surprise_ema = 0.1;
        self.total_calls = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_gate_basic() {
        let config = PredictiveGateConfig { d_model: 4, ..Default::default() };
        let mut gate = PredictiveGate::new(config);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let (output, surprise, gate_val) = gate.forward(&input);

        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(surprise >= 0.0);
        assert!(gate_val >= 0.0 && gate_val <= 1.0);
    }

    #[test]
    fn test_constant_input_low_surprise() {
        let config = PredictiveGateConfig { d_model: 4, ..Default::default() };
        let mut gate = PredictiveGate::new(config);
        let input = vec![1.0; 4];

        // Feed same input many times → surprise should decrease
        let mut last_surprise = f32::MAX;
        for _ in 0..20 {
            let (_, surprise, _) = gate.forward(&input);
            last_surprise = surprise;
        }
        // After convergence, surprise should be low (prediction ≈ actual)
        assert!(last_surprise < 0.5,
            "Constant input should have low surprise, got {last_surprise}");
    }

    #[test]
    fn test_novel_input_high_surprise() {
        let config = PredictiveGateConfig { d_model: 4, ..Default::default() };
        let mut gate = PredictiveGate::new(config);

        // Feed constant input to establish prediction
        for _ in 0..20 {
            gate.forward(&vec![1.0; 4]);
        }

        // Then feed very different input → high surprise
        let (_, surprise, gate_val) = gate.forward(&vec![10.0; 4]);
        assert!(surprise > 0.5, "Novel input should have high surprise, got {surprise}");
        assert!(gate_val > 0.5, "High surprise → gate should open, got {gate_val}");
    }

    #[test]
    fn test_gate_passes_novel_info() {
        let config = PredictiveGateConfig {
            d_model: 2,
            surprise_threshold: 0.1,
            temperature: 10.0,
        };
        let mut gate = PredictiveGate::new(config);

        // Establish prediction at [1, 1]
        for _ in 0..50 {
            gate.forward(&vec![1.0, 1.0]);
        }

        // Novel input [10, -10] → gate should open, output ≈ actual
        let (output, _, gate_val) = gate.forward(&vec![10.0, -10.0]);
        assert!(gate_val > 0.8, "Gate should be mostly open for surprise");
        // Output should lean toward actual values
        assert!(output[0] > 5.0, "Should pass novel info: {}", output[0]);
    }

    #[test]
    fn test_reset() {
        let config = PredictiveGateConfig { d_model: 4, ..Default::default() };
        let mut gate = PredictiveGate::new(config);
        for _ in 0..10 {
            gate.forward(&vec![1.0; 4]);
        }
        assert!(gate.total_calls() > 0);
        gate.reset();
        assert_eq!(gate.total_calls(), 0);
    }

    #[test]
    fn test_surprise_ema() {
        let config = PredictiveGateConfig { d_model: 4, ..Default::default() };
        let mut gate = PredictiveGate::new(config);
        for i in 0..50 {
            gate.forward(&vec![i as f32 * 0.1; 4]);
        }
        let avg = gate.average_surprise();
        assert!(avg > 0.0 && avg.is_finite(), "EMA should track surprise: {avg}");
    }
}
