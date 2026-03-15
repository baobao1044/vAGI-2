//! Optimizer — Sophia + STE for ternary weight training.

/// Sophia optimizer state for one parameter.
#[derive(Clone, Debug)]
pub struct SophiaState {
    pub momentum: f32,
    pub hessian_estimate: f32,
}

/// Sophia optimizer with Straight-Through Estimator (STE).
pub struct SophiaOptimizer {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    pub rho: f32,
    states: Vec<SophiaState>,
}

impl SophiaOptimizer {
    pub fn new(n_params: usize) -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
            rho: 0.04,
            states: vec![SophiaState { momentum: 0.0, hessian_estimate: 1.0 }; n_params],
        }
    }

    /// One optimization step.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        for i in 0..params.len().min(grads.len()).min(self.states.len()) {
            let s = &mut self.states[i];
            s.momentum = self.beta1 * s.momentum + (1.0 - self.beta1) * grads[i];
            let update = s.momentum / s.hessian_estimate.max(self.rho);
            params[i] -= self.lr * (update + self.weight_decay * params[i]);
        }
    }

    /// STE quantization: round to nearest ternary but pass gradient through.
    pub fn ste_quantize(params: &mut [f32]) {
        for p in params.iter_mut() {
            if *p > 0.5 { *p = 1.0; }
            else if *p < -0.5 { *p = -1.0; }
            else { *p = 0.0; }
        }
    }
}
