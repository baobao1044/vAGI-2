//! Ground truth generation: RK45 integrator + physical systems.

/// RK4 integrator (fixed step, sufficient for reference trajectories).
///
/// For ground truth we use RK4 with small dt for high accuracy.
pub fn rk4_step(
    state: &[f32],
    dt: f32,
    deriv: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    let n = state.len();
    let k1 = deriv(state);

    let s2: Vec<f32> = (0..n).map(|i| state[i] + 0.5 * dt * k1[i]).collect();
    let k2 = deriv(&s2);

    let s3: Vec<f32> = (0..n).map(|i| state[i] + 0.5 * dt * k2[i]).collect();
    let k3 = deriv(&s3);

    let s4: Vec<f32> = (0..n).map(|i| state[i] + dt * k3[i]).collect();
    let k4 = deriv(&s4);

    (0..n)
        .map(|i| state[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

/// Generate trajectory using RK4.
pub fn integrate_rk4(
    initial: &[f32],
    dt: f32,
    steps: usize,
    deriv: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Vec<Vec<f32>> {
    let mut trajectory = Vec::with_capacity(steps + 1);
    let mut state = initial.to_vec();
    trajectory.push(state.clone());
    for _ in 0..steps {
        state = rk4_step(&state, dt, deriv);
        trajectory.push(state.clone());
    }
    trajectory
}

// ── Physical Systems ──────────────────────────────────────────

/// Harmonic oscillator: H = ½p² + ½q²
pub mod harmonic {
    /// Hamiltonian.
    pub fn energy(state: &[f32]) -> f32 {
        let q = state[0];
        let p = state[1];
        0.5 * p * p + 0.5 * q * q
    }

    /// Derivatives: dq/dt = ∂H/∂p = p, dp/dt = -∂H/∂q = -q
    pub fn derivatives(state: &[f32]) -> Vec<f32> {
        let q = state[0];
        let p = state[1];
        vec![p, -q]
    }
}

/// Simple pendulum: H = ½p² - cos(q)
pub mod pendulum {
    /// Hamiltonian.
    pub fn energy(state: &[f32]) -> f32 {
        let q = state[0];
        let p = state[1];
        0.5 * p * p - q.cos()
    }

    /// Derivatives: dq/dt = p, dp/dt = -sin(q)
    pub fn derivatives(state: &[f32]) -> Vec<f32> {
        let q = state[0];
        let p = state[1];
        vec![p, -q.sin()]
    }
}

/// Double pendulum (simplified, unit mass/length).
/// State: [q1, q2, p1, p2]
pub mod double_pendulum {
    /// Hamiltonian (simplified, equal masses m=1, lengths l=1, g=1).
    pub fn energy(state: &[f32]) -> f32 {
        let (q1, q2, p1, p2) = (state[0], state[1], state[2], state[3]);
        let delta = q1 - q2;
        let denom = 1.0 + delta.sin().powi(2);
        // Kinetic energy (in terms of momenta)
        let ke = (p1 * p1 + 2.0 * p2 * p2 - 2.0 * p1 * p2 * delta.cos()) / (2.0 * denom);
        // Potential energy
        let pe = -(2.0 * q1.cos() + q2.cos());
        ke + pe
    }

    /// Derivatives via Hamilton's equations (numerical for simplicity).
    pub fn derivatives(state: &[f32]) -> Vec<f32> {
        let eps = 1e-5f32;
        let mut derivs = vec![0.0f32; 4];
        let h0 = energy(state);

        // dq_i/dt = ∂H/∂p_i
        for i in 2..4 {
            let mut s_plus = state.to_vec();
            s_plus[i] += eps;
            derivs[i - 2] = (energy(&s_plus) - h0) / eps;
        }
        // dp_i/dt = -∂H/∂q_i
        for i in 0..2 {
            let mut s_plus = state.to_vec();
            s_plus[i] += eps;
            derivs[i + 2] = -(energy(&s_plus) - h0) / eps;
        }
        derivs
    }
}

/// Dataset: collection of trajectories for a system.
#[derive(Clone)]
pub struct Dataset {
    /// Each trajectory is a sequence of state vectors.
    pub trajectories: Vec<Vec<Vec<f32>>>,
    /// State dimension (2 for 1D systems, 4 for double pendulum).
    pub state_dim: usize,
}

impl Dataset {
    /// Generate dataset for a system.
    pub fn generate(
        n_trajectories: usize,
        traj_length: usize,
        dt: f32,
        state_dim: usize,
        initial_range: &[(f32, f32)],
        deriv: &dyn Fn(&[f32]) -> Vec<f32>,
        seed: u64,
    ) -> Self {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);

        let trajectories: Vec<Vec<Vec<f32>>> = (0..n_trajectories)
            .map(|_| {
                let initial: Vec<f32> = initial_range
                    .iter()
                    .map(|(lo, hi)| rng.gen_range(*lo..*hi))
                    .collect();
                integrate_rk4(&initial, dt, traj_length, deriv)
            })
            .collect();

        Dataset { trajectories, state_dim }
    }

    /// Split into train/validation.
    pub fn split(&self, val_fraction: f32) -> (Dataset, Dataset) {
        let n = self.trajectories.len();
        let n_val = ((n as f32) * val_fraction).ceil() as usize;
        let n_train = n - n_val;
        let train = Dataset {
            trajectories: self.trajectories[..n_train].to_vec(),
            state_dim: self.state_dim,
        };
        let val = Dataset {
            trajectories: self.trajectories[n_train..].to_vec(),
            state_dim: self.state_dim,
        };
        (train, val)
    }

    /// Get training pairs: (state_t, derivatives_t) from all trajectories.
    pub fn training_pairs(
        &self, dt: f32,
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut pairs = Vec::new();
        for traj in &self.trajectories {
            for t in 0..traj.len() - 1 {
                let state = &traj[t];
                // Finite difference derivative estimate
                let deriv: Vec<f32> = state.iter().zip(traj[t + 1].iter())
                    .map(|(s0, s1)| (s1 - s0) / dt)
                    .collect();
                pairs.push((state.clone(), deriv));
            }
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_energy_conservation() {
        let state0 = vec![1.0f32, 0.0];
        let traj = integrate_rk4(&state0, 0.001, 10000, &harmonic::derivatives);
        let e0 = harmonic::energy(&state0);
        let e_final = harmonic::energy(traj.last().unwrap());
        let drift = (e_final - e0).abs() / e0.abs();
        assert!(drift < 1e-4, "RK4 harmonic drift: {drift}");
    }

    #[test]
    fn test_pendulum_energy_conservation() {
        let state0 = vec![1.0f32, 0.5];
        let traj = integrate_rk4(&state0, 0.001, 10000, &pendulum::derivatives);
        let e0 = pendulum::energy(&state0);
        let e_final = pendulum::energy(traj.last().unwrap());
        let drift = (e_final - e0).abs() / e0.abs().max(1e-10);
        assert!(drift < 1e-3, "RK4 pendulum drift: {drift}");
    }

    #[test]
    fn test_double_pendulum_energy_conservation() {
        let state0 = vec![0.5f32, 0.3, 0.1, -0.1];
        let traj = integrate_rk4(&state0, 0.001, 5000, &double_pendulum::derivatives);
        let e0 = double_pendulum::energy(&state0);
        let e_final = double_pendulum::energy(traj.last().unwrap());
        let drift = (e_final - e0).abs() / e0.abs().max(1e-10);
        assert!(drift < 1e-4, "RK4 double pendulum drift: {drift}");
    }

    #[test]
    fn test_harmonic_analytical() {
        // q(t) = cos(t), p(t) = -sin(t) for q0=1, p0=0
        let state0 = vec![1.0f32, 0.0];
        let dt = 0.001;
        let steps = 1000; // t = 1.0
        let traj = integrate_rk4(&state0, dt, steps, &harmonic::derivatives);
        let final_state = traj.last().unwrap();
        let t = dt * steps as f32;
        let q_exact = t.cos();
        let p_exact = -t.sin();
        assert!((final_state[0] - q_exact).abs() < 1e-5,
            "q: got {}, expected {}", final_state[0], q_exact);
        assert!((final_state[1] - p_exact).abs() < 1e-5,
            "p: got {}, expected {}", final_state[1], p_exact);
    }

    #[test]
    fn test_dataset_generation() {
        let ds = Dataset::generate(
            100, 50, 0.01, 2,
            &[(-2.0, 2.0), (-2.0, 2.0)],
            &harmonic::derivatives, 42,
        );
        assert_eq!(ds.trajectories.len(), 100);
        assert_eq!(ds.trajectories[0].len(), 51); // 50 steps + initial
        assert_eq!(ds.trajectories[0][0].len(), 2);
    }

    #[test]
    fn test_dataset_split() {
        let ds = Dataset::generate(
            100, 10, 0.01, 2,
            &[(-1.0, 1.0), (-1.0, 1.0)],
            &harmonic::derivatives, 42,
        );
        let (train, val) = ds.split(0.2);
        assert_eq!(train.trajectories.len(), 80);
        assert_eq!(val.trajectories.len(), 20);
    }

    #[test]
    fn test_training_pairs() {
        let ds = Dataset::generate(
            10, 5, 0.01, 2,
            &[(-1.0, 1.0), (-1.0, 1.0)],
            &harmonic::derivatives, 42,
        );
        let pairs = ds.training_pairs(0.01);
        assert_eq!(pairs.len(), 10 * 5); // 10 traj × 5 steps
        assert_eq!(pairs[0].0.len(), 2); // state dim
        assert_eq!(pairs[0].1.len(), 2); // derivative dim
    }
}
