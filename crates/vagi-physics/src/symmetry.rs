//! Symmetry discovery module (S3.2).

use vagi_core::{BitNetLinear, VagiError};

/// Learnable symmetry group for Hamiltonian systems.
pub struct SymmetryModule {
    /// Generators of continuous symmetries.
    generators: Vec<BitNetLinear>,
    /// Per-symmetry activation logit.
    symmetry_active: Vec<f32>,
    state_dim: usize,
}

impl SymmetryModule {
    pub fn new(state_dim: usize, n_symmetries: usize) -> Self {
        let generators = (0..n_symmetries)
            .map(|_| BitNetLinear::new(state_dim, state_dim, false))
            .collect();
        let symmetry_active = vec![0.0f32; n_symmetries];
        Self { generators, symmetry_active, state_dim }
    }

    /// Apply symmetry transformation: state + tau * generator(state).
    pub fn transform(
        &self, state: &[f32], generator_idx: usize, tau: f32,
    ) -> Result<Vec<f32>, VagiError> {
        if generator_idx >= self.generators.len() {
            return Err(VagiError::Model("Generator index out of range".into()));
        }
        let mut delta = vec![0.0f32; self.state_dim];
        self.generators[generator_idx].forward(state, &mut delta)?;
        Ok(state.iter().zip(delta.iter()).map(|(s, d)| s + tau * d).collect())
    }

    /// Compute conserved quantity for a generator (Noether's theorem).
    /// Q_i = p · G_i(q), where G_i is the i-th generator.
    pub fn conserved_quantity(
        &self, q: &[f32], p: &[f32], generator_idx: usize,
    ) -> Result<f32, VagiError> {
        if generator_idx >= self.generators.len() {
            return Err(VagiError::Model("Generator index out of range".into()));
        }
        let mut gen_q = vec![0.0f32; self.state_dim];
        self.generators[generator_idx].forward(q, &mut gen_q)?;
        let quantity: f32 = p.iter().zip(gen_q.iter()).map(|(pi, gi)| pi * gi).sum();
        Ok(quantity)
    }

    /// Number of symmetry generators.
    pub fn n_symmetries(&self) -> usize {
        self.generators.len()
    }

    /// Get activation logits.
    pub fn active_logits(&self) -> &[f32] {
        &self.symmetry_active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform() {
        let sym = SymmetryModule::new(4, 3);
        let state = vec![1.0f32; 4];
        let result = sym.transform(&state, 0, 0.01);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 4);
    }

    #[test]
    fn test_conserved_quantity() {
        let sym = SymmetryModule::new(4, 3);
        let q = vec![1.0f32; 4];
        let p = vec![0.5f32; 4];
        let cq = sym.conserved_quantity(&q, &p, 0);
        assert!(cq.is_ok());
    }
}
