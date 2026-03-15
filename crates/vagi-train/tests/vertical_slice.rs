//! Vertical Slice MVP: Spring → Linear Predictor → Predict → Evaluate
//!
//! This integration test demonstrates the full data flow:
//! 1. Spring microworld generates trajectory data
//! 2. Math engine verifies energy formula (symbolic + dimensional)
//! 3. Simple linear predictor fits state_t → state_{t+1}
//! 4. Model predicts next state
//! 5. Evaluate: prediction error + energy conservation check

use vagi_physics::microworlds::mechanics::Spring;
use vagi_physics::microworlds::Microworld;
use vagi_physics::units::{DimensionalAnalyzer, Unit};
use vagi_math::{Expr, simplify};
use vagi_math::calculus::differentiate;

/// Collect trajectory data from Spring microworld.
fn collect_trajectories(n_episodes: usize, steps_per_episode: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::with_capacity(n_episodes * steps_per_episode);
    let mut rng = rand::thread_rng();

    for _ in 0..n_episodes {
        let mut spring = Spring::new(1.0, 1.0, 0.01);
        spring.reset(&mut rng);

        for _ in 0..steps_per_episode {
            let state_t = spring.state();
            spring.step(&[]);
            let state_t1 = spring.state();
            data.push((state_t, state_t1));
        }
    }
    data
}

/// Simple linear model: s_{t+1} = A * s_t
/// Learns a 2x2 matrix A via least-squares gradient descent.
struct LinearPredictor {
    /// 2x2 weight matrix stored as [a00, a01, a10, a11]
    weights: [f32; 4],
    lr: f32,
}

impl LinearPredictor {
    fn new() -> Self {
        // Initialize close to identity (good prior for dynamics)
        Self {
            weights: [1.0, 0.0, 0.0, 1.0],
            lr: 1e-4,
        }
    }

    fn predict(&self, state: &[f32]) -> Vec<f32> {
        vec![
            self.weights[0] * state[0] + self.weights[1] * state[1],
            self.weights[2] * state[0] + self.weights[3] * state[1],
        ]
    }

    /// One gradient descent step on MSE loss.
    fn train_step(&mut self, state: &[f32], target: &[f32]) -> f32 {
        let pred = self.predict(state);
        let err = [pred[0] - target[0], pred[1] - target[1]];
        let loss = err[0] * err[0] + err[1] * err[1];

        // Gradient of MSE w.r.t. weights
        // d(loss)/d(w[0]) = 2 * err[0] * state[0]
        self.weights[0] -= self.lr * 2.0 * err[0] * state[0];
        self.weights[1] -= self.lr * 2.0 * err[0] * state[1];
        self.weights[2] -= self.lr * 2.0 * err[1] * state[0];
        self.weights[3] -= self.lr * 2.0 * err[1] * state[1];

        loss
    }

    /// Train on dataset for multiple epochs.
    fn fit(&mut self, data: &[(Vec<f32>, Vec<f32>)], epochs: usize) -> Vec<f32> {
        let mut loss_history = Vec::new();
        for _epoch in 0..epochs {
            let mut total_loss = 0.0f32;
            for (state, target) in data {
                total_loss += self.train_step(state, target);
            }
            loss_history.push(total_loss / data.len() as f32);
        }
        loss_history
    }
}

/// Evaluate predictions: MSE + energy conservation violation.
fn evaluate(
    model: &LinearPredictor,
    test_data: &[(Vec<f32>, Vec<f32>)],
    k: f32,
    m: f32,
) -> (f32, f32) {
    let mut total_mse = 0.0f32;
    let mut total_energy_violation = 0.0f32;
    let count = test_data.len() as f32;

    for (state, target) in test_data {
        let pred = model.predict(state);

        // Prediction MSE
        let mse = (pred[0] - target[0]).powi(2) + (pred[1] - target[1]).powi(2);
        total_mse += mse;

        // Energy conservation check:
        // E_pred = ½k*x_pred² + ½m*v_pred²
        // E_actual = ½k*x_actual² + ½m*v_actual²
        let e_pred = 0.5 * k * pred[0] * pred[0] + 0.5 * m * pred[1] * pred[1];
        let e_actual = 0.5 * k * target[0] * target[0] + 0.5 * m * target[1] * target[1];
        total_energy_violation += (e_pred - e_actual).abs();
    }

    (total_mse / count, total_energy_violation / count)
}

#[test]
fn test_vertical_slice_spring_to_prediction() {
    // ═══════════════════════════════════════════
    // Stage 1: Collect training data from Spring
    // ═══════════════════════════════════════════
    let train_data = collect_trajectories(100, 100);  // 10,000 samples
    let test_data = collect_trajectories(10, 100);     // 1,000 for test
    assert!(train_data.len() >= 10_000, "Need at least 10k samples");

    // ═══════════════════════════════════════════
    // Stage 2: Math engine verifies energy
    // ═══════════════════════════════════════════
    // 2a: Symbolic: verify d/dx[½kx²] = kx (Hooke's law ≈ restoring force)
    let energy_expr = Expr::num(0.5).mul(
        Expr::var("k").mul(Expr::var("x").pow(Expr::num(2.0)))
    );
    let force = differentiate(&energy_expr, "x");
    let force_simplified = simplify::simplify(&force);
    // d/dx[½kx²] should evaluate to k*x at specific points
    let mut bindings = std::collections::HashMap::new();
    bindings.insert("k".to_string(), 2.0);
    bindings.insert("x".to_string(), 3.0);
    let force_val = force_simplified.eval(&bindings);
    assert!(force_val.is_some(), "Force expression should be evaluable");
    let fv = force_val.unwrap();
    // d/dx[½*2*x²] = 2x, at x=3: 6.0
    assert!(
        (fv - 6.0).abs() < 1e-6,
        "d/dx[½kx²] at k=2,x=3 should be 6.0, got {fv}"
    );

    // 2b: Dimensional analysis: ½kx² + ½mv² should have Joule units
    let mut da = DimensionalAnalyzer::new();
    let spring_k_unit = Unit { kg: 1, m: 0, s: -2, a: 0, k: 0, mol: 0, cd: 0 };
    da.set_unit("k", spring_k_unit);
    da.set_unit("x", Unit::meter());
    da.set_unit("m_mass", Unit::kilogram());
    da.set_unit("v", Unit::velocity());
    let total_energy = Expr::num(0.5).mul(
        Expr::var("k").mul(Expr::var("x").pow(Expr::num(2.0)))
    ).add(
        Expr::num(0.5).mul(
            Expr::var("m_mass").mul(Expr::var("v").pow(Expr::num(2.0)))
        )
    );
    let unit = da.check(&total_energy).unwrap();
    assert_eq!(unit, Unit::joule(), "Total spring energy must be in Joules");

    // ═══════════════════════════════════════════
    // Stage 3: Fit linear predictor
    // ═══════════════════════════════════════════
    let mut model = LinearPredictor::new();
    let losses = model.fit(&train_data, 50);

    // Training loss should decrease
    let first_loss = losses[0];
    let last_loss = losses[losses.len() - 1];
    assert!(
        last_loss < first_loss,
        "Training should decrease loss: first={first_loss}, last={last_loss}"
    );

    // ═══════════════════════════════════════════
    // Stage 4 & 5: Predict + Evaluate
    // ═══════════════════════════════════════════
    let k = 1.0f32;
    let m = 1.0f32;
    let (prediction_mse, energy_violation) = evaluate(&model, &test_data, k, m);

    // Must have learned the dynamics
    assert!(
        prediction_mse < 0.01,
        "Prediction MSE = {prediction_mse} (should be < 0.01)"
    );

    // Print results for debugging
    eprintln!("═══ Vertical Slice Results ═══");
    eprintln!("Training samples:    {}", train_data.len());
    eprintln!("Test samples:        {}", test_data.len());
    eprintln!("Final training loss: {last_loss:.6}");
    eprintln!("Test prediction MSE: {prediction_mse:.6}");
    eprintln!("Energy violation:    {energy_violation:.6}");
    eprintln!("Learned weights:     {:?}", model.weights);
    eprintln!("Force d/dx[½kx²]:   {fv}");
    eprintln!("Energy units:        {unit}");
    eprintln!("═════════════════════════════");
}
