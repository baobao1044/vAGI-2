//! Microworld simulations for GENESIS training.

pub mod mechanics;

use rand::Rng;

/// A microworld generates (state, action, next_state) data.
pub trait Microworld: Send + Sync {
    fn state(&self) -> Vec<f32>;
    fn step(&mut self, action: &[f32]) -> Vec<f32>;
    fn reset(&mut self, rng: &mut dyn RngAdapter);
    fn name(&self) -> &str;
    fn state_dim(&self) -> usize;
    fn action_dim(&self) -> usize;
}

/// Adapter trait to work around Rng generics in trait objects.
pub trait RngAdapter {
    fn gen_f32(&mut self) -> f32;
    fn gen_range_f32(&mut self, lo: f32, hi: f32) -> f32;
}

impl<R: Rng> RngAdapter for R {
    fn gen_f32(&mut self) -> f32 { self.gen() }
    fn gen_range_f32(&mut self, lo: f32, hi: f32) -> f32 { self.gen_range(lo..hi) }
}
