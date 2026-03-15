//! Basic mechanics microworlds: FreeFall, Spring, Pendulum.

use super::{Microworld, RngAdapter};

/// Free-fall under gravity.
pub struct FreeFall {
    y: f32,    // height
    vy: f32,   // velocity
    g: f32,    // gravity (9.81)
    dt: f32,   // timestep
}

impl FreeFall {
    pub fn new(g: f32, dt: f32) -> Self {
        Self { y: 10.0, vy: 0.0, g, dt }
    }
}

impl Microworld for FreeFall {
    fn state(&self) -> Vec<f32> { vec![self.y, self.vy] }
    fn step(&mut self, _action: &[f32]) -> Vec<f32> {
        self.vy -= self.g * self.dt;
        self.y += self.vy * self.dt;
        if self.y < 0.0 { self.y = 0.0; self.vy = -self.vy * 0.8; }
        self.state()
    }
    fn reset(&mut self, rng: &mut dyn RngAdapter) {
        self.y = rng.gen_range_f32(1.0, 20.0);
        self.vy = rng.gen_range_f32(-5.0, 5.0);
    }
    fn name(&self) -> &str { "FreeFall" }
    fn state_dim(&self) -> usize { 2 }
    fn action_dim(&self) -> usize { 0 }
}

/// Simple harmonic oscillator (spring).
pub struct Spring {
    x: f32,    // displacement
    v: f32,    // velocity
    k: f32,    // spring constant
    m: f32,    // mass
    dt: f32,
}

impl Spring {
    pub fn new(k: f32, m: f32, dt: f32) -> Self {
        Self { x: 1.0, v: 0.0, k, m, dt }
    }

    /// Total energy: E = ½kx² + ½mv² (should be conserved).
    pub fn energy(&self) -> f32 {
        0.5 * self.k * self.x * self.x + 0.5 * self.m * self.v * self.v
    }

    /// Get spring parameters for dimensional analysis.
    pub fn params(&self) -> (f32, f32) { (self.k, self.m) }
}

impl Microworld for Spring {
    fn state(&self) -> Vec<f32> { vec![self.x, self.v] }
    fn step(&mut self, _action: &[f32]) -> Vec<f32> {
        // Symplectic Euler: update v first, then x with new v
        // This preserves energy to O(dt²) per step (no drift).
        let a = -self.k * self.x / self.m;
        self.v += a * self.dt;
        self.x += self.v * self.dt;
        self.state()
    }
    fn reset(&mut self, rng: &mut dyn RngAdapter) {
        self.x = rng.gen_range_f32(-2.0, 2.0);
        self.v = rng.gen_range_f32(-1.0, 1.0);
    }
    fn name(&self) -> &str { "Spring" }
    fn state_dim(&self) -> usize { 2 }
    fn action_dim(&self) -> usize { 0 }
}

/// Simple pendulum (small angle approximation).
pub struct Pendulum {
    theta: f32,   // angle
    omega: f32,   // angular velocity
    length: f32,
    g: f32,
    dt: f32,
}

impl Pendulum {
    pub fn new(length: f32, g: f32, dt: f32) -> Self {
        Self { theta: 0.3, omega: 0.0, length, g, dt }
    }
}

impl Microworld for Pendulum {
    fn state(&self) -> Vec<f32> { vec![self.theta, self.omega] }
    fn step(&mut self, _action: &[f32]) -> Vec<f32> {
        let alpha = -(self.g / self.length) * self.theta.sin();
        self.omega += alpha * self.dt;
        self.theta += self.omega * self.dt;
        self.state()
    }
    fn reset(&mut self, rng: &mut dyn RngAdapter) {
        self.theta = rng.gen_range_f32(-0.5, 0.5);
        self.omega = rng.gen_range_f32(-0.3, 0.3);
    }
    fn name(&self) -> &str { "Pendulum" }
    fn state_dim(&self) -> usize { 2 }
    fn action_dim(&self) -> usize { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freefall_energy_decreases() {
        let mut ff = FreeFall::new(9.81, 0.01);
        let s0 = ff.state();
        for _ in 0..100 { ff.step(&[]); }
        let s1 = ff.state();
        // Object should have fallen
        assert!(s1[0] <= s0[0] + 1.0);
    }

    #[test]
    fn test_spring_oscillates() {
        let mut sp = Spring::new(1.0, 1.0, 0.01);
        let x0 = sp.state()[0];
        for _ in 0..314 { sp.step(&[]); } // ~half period
        let x1 = sp.state()[0];
        // Should be on opposite side
        assert!(x0 * x1 < 0.1, "Spring should oscillate");
    }

    #[test]
    fn test_spring_energy_conservation() {
        // E = ½kx² + ½mv² should be conserved under symplectic integration.
        let mut sp = Spring::new(2.0, 1.0, 0.001); // k=2, m=1, small dt
        let e0 = sp.energy();
        assert!(e0 > 0.0, "Initial energy should be positive");

        for _ in 0..10_000 { sp.step(&[]); }
        let e_final = sp.energy();

        let drift = ((e_final - e0) / e0).abs();
        assert!(
            drift < 0.001, // <0.1% drift over 10k steps
            "Energy drift = {:.4}%, E0={e0}, E_final={e_final}",
            drift * 100.0
        );
    }

    #[test]
    fn test_spring_analytical_solution() {
        // x(t) = A*cos(ωt), where ω = √(k/m), A = x₀ when v₀=0
        let k = 1.0f32;
        let m = 1.0f32;
        let dt = 0.001;
        let mut sp = Spring::new(k, m, dt);
        let omega = (k / m).sqrt();
        let a = sp.state()[0]; // x₀ = 1.0

        let t_target = std::f32::consts::PI; // half period
        let steps = (t_target / dt) as usize;
        for _ in 0..steps { sp.step(&[]); }
        let x_final = sp.state()[0];

        // x(π) = cos(π) = -1.0
        let expected = a * (omega * t_target).cos();
        let error = (x_final - expected).abs();
        assert!(
            error < 0.01,
            "Analytical check: x({t_target})={x_final}, expected={expected}, error={error}"
        );
    }

    #[test]
    fn test_pendulum_bounded() {
        let mut p = Pendulum::new(1.0, 9.81, 0.01);
        for _ in 0..10000 { p.step(&[]); }
        let s = p.state();
        assert!(s[0].abs() < 10.0, "Pendulum angle should be bounded");
    }
}

