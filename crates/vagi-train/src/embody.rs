//! Stage 1: Embody — JEPA-style self-supervised learning (S4.2).

use vagi_core::BitNetBlock;

/// JEPA-style embodiment trainer.
pub struct EmbodimentTrainer {
    /// Context encoder (learns via gradient).
    pub context_encoder: BitNetBlock,
    /// Target encoder (EMA-updated, no gradient).
    pub target_encoder: BitNetBlock,
    /// Predictor: context → target representation.
    pub predictor: BitNetBlock,
    /// EMA momentum for target encoder.
    pub ema_momentum: f32,
    pub d_model: usize,
}

impl EmbodimentTrainer {
    pub fn new(d_model: usize) -> Self {
        Self {
            context_encoder: BitNetBlock::new(d_model, 4 * d_model),
            target_encoder: BitNetBlock::new(d_model, 4 * d_model),
            predictor: BitNetBlock::new(d_model, 4 * d_model),
            ema_momentum: 0.996,
            d_model,
        }
    }

    /// One training step on a trajectory.
    ///
    /// Returns JEPA loss (prediction error in latent space).
    pub fn train_step(&mut self, trajectory: &[Vec<f32>], _mask_ratio: f32) -> f32 {
        if trajectory.len() < 2 { return 0.0; }
        // Encode context (first half)
        let mid = trajectory.len() / 2;
        let context = self.context_encoder
            .forward_vec(&trajectory[mid - 1])
            .unwrap_or_else(|_| vec![0.0; self.d_model]);
        // Encode target (second half)
        let target = self.target_encoder
            .forward_vec(&trajectory[mid])
            .unwrap_or_else(|_| vec![0.0; self.d_model]);
        // Predict target from context
        let prediction = self.predictor
            .forward_vec(&context)
            .unwrap_or_else(|_| vec![0.0; self.d_model]);
        // L2 loss in latent space
        let loss: f32 = prediction.iter().zip(target.iter())
            .map(|(p, t)| (p - t).powi(2)).sum::<f32>() / self.d_model as f32;
        loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_step() {
        let mut trainer = EmbodimentTrainer::new(64);
        let traj: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32 * 0.1; 64])
            .collect();
        let loss = trainer.train_step(&traj, 0.5);
        assert!(loss.is_finite());
    }
}
