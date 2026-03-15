//! Hamiltonian Neural Network (S3.1).
//!
//! Learns energy H(q,p), derives dynamics via Hamilton's equations.
//! Energy conservation is structural (by construction).

use vagi_core::{BitNetBlock, BitNetLinear, VagiError};

/// Hamiltonian Neural Network.
pub struct HamiltonianNN {
    /// H(q,p) → scalar energy. Input: [q; p], output: scalar.
    energy_net: BitNetBlock,
    energy_head: BitNetLinear,
    d_state: usize,
}

impl HamiltonianNN {
    pub fn new(d_state: usize) -> Self {
        let input_dim = 2 * d_state;
        let hidden_dim = 4 * d_state;
        Self {
            energy_net: BitNetBlock::new(input_dim, hidden_dim),
            energy_head: BitNetLinear::new(input_dim, 1, false),
            d_state,
        }
    }

    /// Compute Hamiltonian (total energy) for state (q, p).
    pub fn energy(&self, q: &[f32], p: &[f32]) -> Result<f32, VagiError> {
        let mut input = Vec::with_capacity(2 * self.d_state);
        input.extend_from_slice(q);
        input.extend_from_slice(p);
        let hidden = self.energy_net.forward_vec(&input)?;
        let mut out = vec![0.0f32; 1];
        self.energy_head.forward(&hidden, &mut out)?;
        Ok(out[0])
    }

    /// Compute dynamics via Hamilton's equations (finite differences).
    ///
    /// dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
    pub fn dynamics(&self, q: &[f32], p: &[f32]) -> Result<(Vec<f32>, Vec<f32>), VagiError> {
        let eps = 1e-4f32;
        let mut dq_dt = vec![0.0f32; self.d_state];
        let mut dp_dt = vec![0.0f32; self.d_state];

        let h0 = self.energy(q, p)?;

        // ∂H/∂p_i ≈ (H(q, p+ε_i) - H(q, p)) / ε
        for i in 0..self.d_state {
            let mut p_plus = p.to_vec();
            p_plus[i] += eps;
            let h_plus = self.energy(q, &p_plus)?;
            dq_dt[i] = (h_plus - h0) / eps;
        }

        // -∂H/∂q_i ≈ -(H(q+ε_i, p) - H(q, p)) / ε
        for i in 0..self.d_state {
            let mut q_plus = q.to_vec();
            q_plus[i] += eps;
            let h_plus = self.energy(&q_plus, p)?;
            dp_dt[i] = -(h_plus - h0) / eps;
        }

        Ok((dq_dt, dp_dt))
    }

    /// Symplectic (leapfrog) integration.
    pub fn integrate(
        &self, q: &[f32], p: &[f32], dt: f32, steps: usize,
    ) -> Result<Vec<(Vec<f32>, Vec<f32>)>, VagiError> {
        let mut trajectory = Vec::with_capacity(steps + 1);
        let mut q_cur = q.to_vec();
        let mut p_cur = p.to_vec();
        trajectory.push((q_cur.clone(), p_cur.clone()));

        for _ in 0..steps {
            // Leapfrog: half-step p, full-step q, half-step p
            let (_, dp1) = self.dynamics(&q_cur, &p_cur)?;
            let p_half: Vec<f32> = p_cur.iter().zip(dp1.iter())
                .map(|(pi, di)| pi + 0.5 * dt * di).collect();

            let (dq, _) = self.dynamics(&q_cur, &p_half)?;
            q_cur = q_cur.iter().zip(dq.iter())
                .map(|(qi, di)| qi + dt * di).collect();

            let (_, dp2) = self.dynamics(&q_cur, &p_half)?;
            p_cur = p_half.iter().zip(dp2.iter())
                .map(|(pi, di)| pi + 0.5 * dt * di).collect();

            trajectory.push((q_cur.clone(), p_cur.clone()));
        }

        Ok(trajectory)
    }

    /// State dimension.
    pub fn d_state(&self) -> usize {
        self.d_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_computes() {
        let hnn = HamiltonianNN::new(4);
        let q = vec![1.0f32; 4];
        let p = vec![0.5f32; 4];
        let e = hnn.energy(&q, &p);
        assert!(e.is_ok());
    }

    #[test]
    fn test_dynamics_shape() {
        let hnn = HamiltonianNN::new(4);
        let q = vec![1.0f32; 4];
        let p = vec![0.5f32; 4];
        let (dq, dp) = hnn.dynamics(&q, &p).unwrap();
        assert_eq!(dq.len(), 4);
        assert_eq!(dp.len(), 4);
    }

    #[test]
    fn test_integration() {
        let hnn = HamiltonianNN::new(2);
        let q = vec![1.0, 0.0];
        let p = vec![0.0, 1.0];
        let traj = hnn.integrate(&q, &p, 0.01, 10).unwrap();
        assert_eq!(traj.len(), 11);
    }
}
