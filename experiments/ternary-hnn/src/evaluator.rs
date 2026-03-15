//! Evaluator — compute metrics (energy drift, trajectory MSE, inference speed).

use crate::models::HNNModel;
use crate::ground_truth;

/// Energy conservation metric.
///
/// Integrates using learned Hamiltonian for `steps` steps,
/// reports max|ΔE|/|E_0| over the trajectory.
pub fn energy_drift(
    model: &dyn HNNModel,
    initial_state: &[f32],
    dt: f32,
    steps: usize,
    energy_fn: &dyn Fn(&[f32]) -> f32,
) -> f32 {
    let traj = model.integrate(initial_state, dt, steps);
    let e0 = energy_fn(initial_state);
    let e0_abs = e0.abs().max(1e-10);

    traj.iter()
        .map(|s| (energy_fn(s) - e0).abs() / e0_abs)
        .fold(0.0f32, f32::max)
}

/// Trajectory MSE vs ground truth at specific step indices.
pub fn trajectory_mse(
    predicted: &[Vec<f32>],
    ground_truth: &[Vec<f32>],
    eval_steps: &[usize],
) -> Vec<(usize, f32)> {
    eval_steps.iter().map(|&step| {
        let step = step.min(predicted.len() - 1).min(ground_truth.len() - 1);
        let mse: f32 = predicted[step].iter().zip(ground_truth[step].iter())
            .map(|(p, g)| (p - g) * (p - g))
            .sum::<f32>() / predicted[step].len() as f32;
        (step, mse)
    }).collect()
}

/// Inference speed: microseconds per forward pass.
pub fn inference_speed(model: &dyn HNNModel, state: &[f32], n_iters: usize) -> f64 {
    // Warmup
    for _ in 0..100 {
        let _ = model.hamiltonian(state);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n_iters {
        let _ = model.hamiltonian(state);
    }
    let elapsed = t0.elapsed();
    elapsed.as_micros() as f64 / n_iters as f64
}

/// Inference speed for MLP model.
pub fn inference_speed_mlp(model: &crate::models::MLPFP32, state: &[f32], n_iters: usize) -> f64 {
    for _ in 0..100 {
        let _ = model.predict_derivatives(state);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n_iters {
        let _ = model.predict_derivatives(state);
    }
    let elapsed = t0.elapsed();
    elapsed.as_micros() as f64 / n_iters as f64
}

/// Run full evaluation for an HNN model on a system.
pub struct EvalResult {
    pub model_name: String,
    pub system_name: String,
    pub seed: u64,
    /// max|ΔE|/|E₀| at various step counts
    pub energy_drifts: Vec<(usize, f32)>,
    /// Trajectory MSE at various steps
    pub trajectory_mses: Vec<(usize, f32)>,
    /// µs per forward pass
    pub inference_us: f64,
    /// Memory bytes
    pub memory_bytes: usize,
    /// Param count
    pub param_count: usize,
}

/// Evaluate a model on a system.
pub fn evaluate_hnn(
    model: &dyn HNNModel,
    system_name: &str,
    seed: u64,
    energy_fn: &dyn Fn(&[f32]) -> f32,
    deriv_fn: &dyn Fn(&[f32]) -> Vec<f32>,
    initial_states: &[Vec<f32>],
    dt: f32,
    eval_steps: &[usize],
) -> EvalResult {
    let max_steps = *eval_steps.iter().max().unwrap_or(&100);

    // Energy drift: average over all initial states
    let mut drift_sums = vec![0.0f32; eval_steps.len()];
    for init in initial_states {
        let traj = model.integrate(init, dt, max_steps);
        let e0 = energy_fn(init);
        let e0_abs = e0.abs().max(1e-10);
        for (i, &step) in eval_steps.iter().enumerate() {
            let s = step.min(traj.len() - 1);
            let drift = (energy_fn(&traj[s]) - e0).abs() / e0_abs;
            drift_sums[i] += drift;
        }
    }
    let n = initial_states.len() as f32;
    let energy_drifts: Vec<(usize, f32)> = eval_steps.iter()
        .zip(drift_sums.iter())
        .map(|(&step, &sum)| (step, sum / n))
        .collect();

    // Trajectory MSE: average over initial states
    let mut mse_sums = vec![0.0f32; eval_steps.len()];
    for init in initial_states {
        let pred_traj = model.integrate(init, dt, max_steps);
        let gt_traj = ground_truth::integrate_rk4(init, dt, max_steps, deriv_fn);
        for (i, &step) in eval_steps.iter().enumerate() {
            let s = step.min(pred_traj.len() - 1).min(gt_traj.len() - 1);
            let mse: f32 = pred_traj[s].iter().zip(gt_traj[s].iter())
                .map(|(p, g)| (p - g) * (p - g))
                .sum::<f32>() / pred_traj[s].len() as f32;
            mse_sums[i] += mse;
        }
    }
    let trajectory_mses: Vec<(usize, f32)> = eval_steps.iter()
        .zip(mse_sums.iter())
        .map(|(&step, &sum)| (step, sum / n))
        .collect();

    // Inference speed
    let test_state = initial_states.first().unwrap();
    let inference_us = inference_speed(model, test_state, 1000);

    EvalResult {
        model_name: model.name().to_string(),
        system_name: system_name.to_string(),
        seed,
        energy_drifts,
        trajectory_mses,
        inference_us,
        memory_bytes: model.memory_bytes(),
        param_count: model.param_count(),
    }
}

/// Format results as CSV lines.
impl EvalResult {
    pub fn to_csv_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for (step, drift) in &self.energy_drifts {
            lines.push(format!("{},{},{},{},energy_drift,{},{:.6e}",
                self.model_name, self.system_name, self.seed, step, self.param_count, drift));
        }
        for (step, mse) in &self.trajectory_mses {
            lines.push(format!("{},{},{},{},trajectory_mse,{},{:.6e}",
                self.model_name, self.system_name, self.seed, step, self.param_count, mse));
        }
        lines.push(format!("{},{},{},0,inference_us,{},{:.2}",
            self.model_name, self.system_name, self.seed, self.param_count, self.inference_us));
        lines.push(format!("{},{},{},0,memory_bytes,{},{}",
            self.model_name, self.system_name, self.seed, self.param_count, self.memory_bytes));
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::HNNFP32;

    #[test]
    fn test_energy_drift_finite() {
        let model = HNNFP32::new(1, 16, 2, 42);
        let drift = energy_drift(
            &model, &[1.0, 0.0], 0.01, 10,
            &ground_truth::harmonic::energy,
        );
        assert!(drift.is_finite(), "Drift should be finite: {drift}");
    }

    #[test]
    fn test_evaluation_runs() {
        let model = HNNFP32::new(1, 16, 2, 42);
        let inits = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let result = evaluate_hnn(
            &model, "harmonic", 42,
            &ground_truth::harmonic::energy,
            &ground_truth::harmonic::derivatives,
            &inits, 0.01, &[1, 10],
        );
        assert!(!result.energy_drifts.is_empty());
        assert!(result.inference_us > 0.0);
    }

    #[test]
    fn test_csv_output() {
        let model = HNNFP32::new(1, 16, 2, 42);
        let inits = vec![vec![1.0f32, 0.0]];
        let result = evaluate_hnn(
            &model, "harmonic", 42,
            &ground_truth::harmonic::energy,
            &ground_truth::harmonic::derivatives,
            &inits, 0.01, &[1, 10],
        );
        let lines = result.to_csv_lines();
        assert!(!lines.is_empty());
        assert!(lines[0].contains("HNN-FP32"));
    }
}
