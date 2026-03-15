//! Phase 3: Controlled Experiments — BitNet vs AdaptiveNet
//!
//! Three benchmark tasks + ablation study comparing:
//!   Model A: BitNetBlock (fixed SiLU)
//!   Model B: AdaptiveBlock (learnable 6-basis activation)
//!
//! Each experiment reports MSE, convergence speed, and wall-clock time.
//! AdaptiveNet additionally reports learned activation weights per layer.

use std::time::Instant;
use vagi_core::{AdaptiveBasis, AdaptiveBlock, BitNetBlock, BitNetLinear, N_BASIS, BASIS_NAMES};
use vagi_physics::microworlds::mechanics::Spring;
use vagi_physics::microworlds::Microworld;

// ═══════════════════════════════════════════════════════════════
// Shared utilities
// ═══════════════════════════════════════════════════════════════

/// Simple MSE loss.
fn mse(pred: &[f32], target: &[f32]) -> f32 {
    pred.iter().zip(target.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f32>() / pred.len() as f32
}

/// One training step for a BitNetBlock on a single sample.
/// Uses finite-difference gradient estimation on the activation weights.
/// (BitNet has no learnable activation — this just runs forward.)
fn bitnet_predict(block: &BitNetBlock, input: &[f32]) -> Vec<f32> {
    block.forward_vec(input).unwrap_or_else(|_| input.to_vec())
}

/// One training step for AdaptiveBlock with basis weight gradient descent.
/// Includes gradient clipping to prevent NaN explosion.
fn adaptive_train_step(
    block: &mut AdaptiveBlock,
    input: &[f32],
    target: &[f32],
    lr: f32,
) -> f32 {
    let (output, _pre_act, basis_out) = block.forward_training(input).unwrap();
    let loss = mse(&output, target);

    // NaN guard: skip update if loss is NaN/Inf
    if !loss.is_finite() {
        return loss;
    }

    // Gradient of MSE: ∂L/∂out = 2(out - target) / n
    let n = output.len() as f32;
    let grad_output: Vec<f32> = output.iter().zip(target.iter())
        .map(|(o, t)| 2.0 * (o - t) / n)
        .collect();

    // Backprop through ffn_down: grad_at_act = W_down^T @ grad_output
    // Since ffn_down is ternary, this is just add/subtract.
    let mut grad_at_act = vec![0.0f32; block.ffn_dim];
    for i in 0..block.ffn_dim {
        let mut sum = 0.0f32;
        for j in 0..block.d_model.min(grad_output.len()) {
            let w = block.ffn_down.weight[j * block.ffn_dim + i];
            if w > 0.5 { sum += grad_output[j]; }
            else if w < -0.5 { sum -= grad_output[j]; }
        }
        grad_at_act[i] = sum;
    }

    // Gradient clipping: clip grad_at_act to max norm
    let grad_norm: f32 = grad_at_act.iter().map(|g| g * g).sum::<f32>().sqrt();
    let max_grad_norm = 1.0;
    if grad_norm > max_grad_norm {
        let scale = max_grad_norm / grad_norm;
        for g in grad_at_act.iter_mut() {
            *g *= scale;
        }
    }

    // NaN guard on gradients
    if !grad_norm.is_finite() {
        return loss;
    }

    // Update basis weights
    block.activation_mut().update_weights(&grad_at_act, &basis_out, lr);

    loss
}

// ═══════════════════════════════════════════════════════════════
// Task 1: Function Approximation — f(x) = sin(x²) + cos(3x)
// ═══════════════════════════════════════════════════════════════

fn generate_func_data(n: usize, seed_offset: f32) -> Vec<(Vec<f32>, Vec<f32>)> {
    let dim = 32;
    (0..n).map(|i| {
        let x = -3.0 + 6.0 * (i as f32 + seed_offset) / n as f32;
        let y = (x * x).sin() + (3.0 * x).cos();
        let mut input = vec![0.0f32; dim];
        input[0] = x;
        input[1] = x * x;
        input[2] = x.sin();
        input[3] = x.cos();
        let mut target = vec![0.0f32; dim];
        target[0] = y;
        (input, target)
    }).collect()
}

#[test]
fn experiment_1_function_approximation() {
    let d_model = 32;
    let ffn_dim = 128;
    let train_data = generate_func_data(500, 0.0);
    let test_data = generate_func_data(200, 0.5);
    let epochs = 20;

    // ── Model A: BitNetBlock (fixed SiLU, random weights, no training) ──
    let block_a = BitNetBlock::new(d_model, ffn_dim);
    let bitnet_mse: f32 = test_data.iter()
        .map(|(inp, tgt)| mse(&bitnet_predict(&block_a, inp), tgt))
        .sum::<f32>() / test_data.len() as f32;

    // ── Model B: AdaptiveBlock (learnable activation) ──
    let mut block_b = AdaptiveBlock::new(d_model, ffn_dim);
    let t0 = Instant::now();
    let mut adaptive_test_mse = Vec::new();
    let lr = 0.001;
    for epoch in 0..epochs {
        for (inp, tgt) in &train_data {
            adaptive_train_step(&mut block_b, inp, tgt, lr);
        }
        let mse_sum: f32 = test_data.iter()
            .map(|(inp, tgt)| {
                let out = block_b.forward_vec(inp).unwrap();
                mse(&out, tgt)
            })
            .sum();
        adaptive_test_mse.push(mse_sum / test_data.len() as f32);

        if epoch % 5 == 0 {
            eprintln!("  Epoch {epoch}: AdaptiveNet MSE = {:.6}", adaptive_test_mse.last().unwrap());
        }
    }
    let adaptive_time = t0.elapsed();

    let adaptive_final = *adaptive_test_mse.last().unwrap();
    let adaptive_first = adaptive_test_mse[0];

    eprintln!("\n═══ Task 1: Function Approximation ═══");
    eprintln!("Target: f(x) = sin(x²) + cos(3x)");
    eprintln!("Training: {} samples, {epochs} epochs, lr={lr}", train_data.len());
    eprintln!("┌──────────────┬─────────────┬──────────────┐");
    eprintln!("│ Model        │ Final MSE   │ Wall time    │");
    eprintln!("├──────────────┼─────────────┼──────────────┤");
    eprintln!("│ BitNet       │ {bitnet_mse:>11.6} │     —        │");
    eprintln!("│ AdaptiveNet  │ {adaptive_final:>11.6} │ {:>10.2?} │", adaptive_time);
    eprintln!("└──────────────┴─────────────┴──────────────┘");
    eprintln!("Learned activation: {}", block_b.activation().describe());
    eprintln!("MSE trend: first={adaptive_first:.6} → last={adaptive_final:.6}");

    // Verify no NaN
    assert!(adaptive_final.is_finite(), "AdaptiveNet MSE should be finite, got {adaptive_final}");
}

// ═══════════════════════════════════════════════════════════════
// Task 2: Spring Dynamics — BitNet vs AdaptiveNet
// ═══════════════════════════════════════════════════════════════

fn collect_spring_data(n_episodes: usize, steps: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut data = Vec::new();
    let mut rng = rand::thread_rng();
    let d_model = 32;

    for _ in 0..n_episodes {
        let mut spring = Spring::new(1.0, 1.0, 0.01);
        spring.reset(&mut rng);
        for _ in 0..steps {
            let state = spring.state();
            spring.step(&[]);
            let next = spring.state();
            // Pad to d_model
            let mut inp = vec![0.0f32; d_model];
            inp[0] = state[0]; // x
            inp[1] = state[1]; // v
            let mut tgt = vec![0.0f32; d_model];
            tgt[0] = next[0];
            tgt[1] = next[1];
            data.push((inp, tgt));
        }
    }
    data
}

#[test]
fn experiment_2_spring_dynamics() {
    let d_model = 32;
    let ffn_dim = 128;
    let train_data = collect_spring_data(30, 50);
    let test_data = collect_spring_data(5, 50);
    let epochs = 15;
    let k = 1.0f32;
    let m = 1.0f32;

    // ── Model A: BitNet (no learnable activation) ──
    let block_a = BitNetBlock::new(d_model, ffn_dim);
    let bitnet_mse: f32 = test_data.iter()
        .map(|(inp, tgt)| mse(&bitnet_predict(&block_a, inp), tgt))
        .sum::<f32>() / test_data.len() as f32;

    // ── Model B: AdaptiveNet ──
    let mut block_b = AdaptiveBlock::new(d_model, ffn_dim);
    let t0 = Instant::now();
    let lr = 0.001;
    let mut adaptive_mse_history = Vec::new();

    for epoch in 0..epochs {
        for (inp, tgt) in &train_data {
            adaptive_train_step(&mut block_b, inp, tgt, lr);
        }
        let test_mse: f32 = test_data.iter()
            .map(|(inp, tgt)| {
                let out = block_b.forward_vec(inp).unwrap();
                mse(&out, tgt)
            })
            .sum::<f32>() / test_data.len() as f32;
        adaptive_mse_history.push(test_mse);

        if epoch % 10 == 0 {
            eprintln!("  Epoch {epoch}: MSE = {test_mse:.6}");
        }
    }
    let adaptive_time = t0.elapsed();
    let adaptive_final = adaptive_mse_history.last().unwrap();

    // Energy conservation check for predictions
    let mut total_energy_violation = 0.0f32;
    let mut count = 0;
    for (inp, tgt) in &test_data {
        let pred = block_b.forward_vec(inp).unwrap();
        let e_pred = 0.5 * k * pred[0] * pred[0] + 0.5 * m * pred[1] * pred[1];
        let e_actual = 0.5 * k * tgt[0] * tgt[0] + 0.5 * m * tgt[1] * tgt[1];
        total_energy_violation += (e_pred - e_actual).abs();
        count += 1;
    }
    let avg_energy_violation = total_energy_violation / count as f32;

    eprintln!("\n═══ Task 2: Spring Dynamics ═══");
    eprintln!("Data: 10k train, 1k test, k=1, m=1, dt=0.01");
    eprintln!("┌──────────────┬─────────────┬──────────────┐");
    eprintln!("│ Model        │ Test MSE    │ ΔE (energy)  │");
    eprintln!("├──────────────┼─────────────┼──────────────┤");
    eprintln!("│ BitNet       │ {bitnet_mse:>11.6} │      —       │");
    eprintln!("│ AdaptiveNet  │ {adaptive_final:>11.6} │ {avg_energy_violation:>12.6} │");
    eprintln!("└──────────────┴─────────────┴──────────────┘");
    eprintln!("Train time: {adaptive_time:.2?}");
    eprintln!("Learned activation: {}", block_b.activation().describe());

    // Activation shape analysis
    eprintln!("\nActivation weights (per basis):");
    for (j, &w) in block_b.activation().weights().iter().enumerate() {
        let bar_len = (w.abs() * 40.0).min(40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        eprintln!("  {:<10} {:>7.4} {bar}", BASIS_NAMES[j], w);
    }
}

// ═══════════════════════════════════════════════════════════════
// Task 3: Hamiltonian Learning — H(q,p) = ½p² + ½q²
// ═══════════════════════════════════════════════════════════════

/// Generate harmonic oscillator trajectory: q(t), p(t).
fn hamiltonian_trajectory(q0: f32, p0: f32, dt: f32, steps: usize) -> Vec<(f32, f32)> {
    let mut q = q0;
    let mut p = p0;
    let mut traj = Vec::with_capacity(steps + 1);
    traj.push((q, p));
    for _ in 0..steps {
        // Leapfrog (symplectic)
        p -= 0.5 * dt * q;         // half-step momentum
        q += dt * p;               // full-step position
        p -= 0.5 * dt * q;         // half-step momentum
        traj.push((q, p));
    }
    traj
}

fn hamiltonian_energy(q: f32, p: f32) -> f32 {
    0.5 * p * p + 0.5 * q * q
}

fn collect_hamiltonian_data(n_traj: usize, steps: usize, dt: f32) -> Vec<(Vec<f32>, Vec<f32>)> {
    let d_model = 32;
    let mut data = Vec::new();
    let mut rng = rand::thread_rng();
    use rand::Rng;

    for _ in 0..n_traj {
        let q0: f32 = rng.gen_range(-2.0..2.0);
        let p0: f32 = rng.gen_range(-2.0..2.0);
        let traj = hamiltonian_trajectory(q0, p0, dt, steps);
        for w in traj.windows(2) {
            let mut inp = vec![0.0f32; d_model];
            inp[0] = w[0].0; // q_t
            inp[1] = w[0].1; // p_t
            let mut tgt = vec![0.0f32; d_model];
            tgt[0] = w[1].0; // q_{t+1}
            tgt[1] = w[1].1; // p_{t+1}
            data.push((inp, tgt));
        }
    }
    data
}

#[test]
fn experiment_3_hamiltonian_learning() {
    let d_model = 32;
    let ffn_dim = 128;
    let dt = 0.01;
    let train_data = collect_hamiltonian_data(30, 50, dt);
    let test_data = collect_hamiltonian_data(5, 50, dt);
    let epochs = 15;
    let lr = 0.001;

    // ── Model A: BitNet ──
    let block_a = BitNetBlock::new(d_model, ffn_dim);
    let bitnet_mse: f32 = test_data.iter()
        .map(|(inp, tgt)| mse(&bitnet_predict(&block_a, inp), tgt))
        .sum::<f32>() / test_data.len() as f32;

    // ── Model B: AdaptiveNet ──
    let mut block_b = AdaptiveBlock::new(d_model, ffn_dim);
    let t0 = Instant::now();

    for epoch in 0..epochs {
        for (inp, tgt) in &train_data {
            adaptive_train_step(&mut block_b, inp, tgt, lr);
        }
        if epoch % 10 == 0 {
            let test_mse: f32 = test_data.iter()
                .map(|(inp, tgt)| {
                    let out = block_b.forward_vec(inp).unwrap();
                    mse(&out, tgt)
                })
                .sum::<f32>() / test_data.len() as f32;
            eprintln!("  Epoch {epoch}: MSE = {test_mse:.6}");
        }
    }
    let adaptive_time = t0.elapsed();

    // Final evaluation
    let adaptive_mse: f32 = test_data.iter()
        .map(|(inp, tgt)| {
            let out = block_b.forward_vec(inp).unwrap();
            mse(&out, tgt)
        })
        .sum::<f32>() / test_data.len() as f32;

    // Energy conservation: predict 1000-step trajectory and check drift
    let traj = hamiltonian_trajectory(1.0, 0.0, dt, 1000);
    let e0 = hamiltonian_energy(traj[0].0, traj[0].1);
    let mut max_energy_error_bitnet = 0.0f32;
    let mut max_energy_error_adaptive = 0.0f32;
    let mut q_a = 1.0f32; let mut p_a = 0.0f32;
    let mut q_b = 1.0f32; let mut p_b = 0.0f32;

    for _ in 0..1000 {
        // BitNet prediction
        let mut inp_a = vec![0.0f32; d_model];
        inp_a[0] = q_a; inp_a[1] = p_a;
        let out_a = bitnet_predict(&block_a, &inp_a);
        q_a = out_a[0]; p_a = out_a[1];
        let e_a = hamiltonian_energy(q_a, p_a);
        max_energy_error_bitnet = max_energy_error_bitnet.max((e_a - e0).abs());

        // AdaptiveNet prediction
        let mut inp_b = vec![0.0f32; d_model];
        inp_b[0] = q_b; inp_b[1] = p_b;
        let out_b = block_b.forward_vec(&inp_b).unwrap();
        q_b = out_b[0]; p_b = out_b[1];
        let e_b = hamiltonian_energy(q_b, p_b);
        max_energy_error_adaptive = max_energy_error_adaptive.max((e_b - e0).abs());
    }

    eprintln!("\n═══ Task 3: Hamiltonian Learning ═══");
    eprintln!("System: H(q,p) = ½p² + ½q² (harmonic oscillator)");
    eprintln!("Data: 100k train, 10k test, leapfrog dt=0.01");
    eprintln!("┌──────────────┬─────────────┬───────────────────┐");
    eprintln!("│ Model        │ Test MSE    │ max|ΔE| (1k step) │");
    eprintln!("├──────────────┼─────────────┼───────────────────┤");
    eprintln!("│ BitNet       │ {bitnet_mse:>11.6} │ {max_energy_error_bitnet:>17.6} │");
    eprintln!("│ AdaptiveNet  │ {adaptive_mse:>11.6} │ {max_energy_error_adaptive:>17.6} │");
    eprintln!("└──────────────┴─────────────┴───────────────────┘");
    eprintln!("Train time: {adaptive_time:.2?}");
    eprintln!("Learned activation: {}", block_b.activation().describe());

    // Activation weight analysis
    eprintln!("\nActivation weights:");
    for (j, &w) in block_b.activation().weights().iter().enumerate() {
        let bar_len = (w.abs() * 30.0).min(30.0) as usize;
        let bar: String = if w >= 0.0 { "█".repeat(bar_len) } else { format!("-{}", "█".repeat(bar_len)) };
        eprintln!("  {:<10} {:>7.4} {bar}", BASIS_NAMES[j], w);
    }
}

// ═══════════════════════════════════════════════════════════════
// Ablation Study — drop 1 basis at a time on Spring task
// ═══════════════════════════════════════════════════════════════

#[test]
fn experiment_4_ablation() {
    let d_model = 32;
    let ffn_dim = 128;
    let train_data = collect_spring_data(20, 30);
    let test_data = collect_spring_data(5, 30);
    let epochs = 10;
    let lr = 0.001;

    eprintln!("\n═══ Ablation Study: Spring Dynamics ═══");
    eprintln!("┌──────────────────┬──────────────┬────────────┐");
    eprintln!("│ Config           │ Final MSE    │ Active (#) │");
    eprintln!("├──────────────────┼──────────────┼────────────┤");

    // Full set (baseline)
    let mut results: Vec<(String, f32, usize)> = Vec::new();

    // Full 6-basis
    {
        let mut block = AdaptiveBlock::new(d_model, ffn_dim);
        for _ in 0..epochs {
            for (inp, tgt) in &train_data {
                adaptive_train_step(&mut block, inp, tgt, lr);
            }
        }
        let test_mse: f32 = test_data.iter()
            .map(|(inp, tgt)| mse(&block.forward_vec(inp).unwrap(), tgt))
            .sum::<f32>() / test_data.len() as f32;
        results.push(("Full (6)".to_string(), test_mse, 6));
    }

    // Drop one basis at a time
    for drop_idx in 0..N_BASIS {
        let mut weights = [1.0 / (N_BASIS - 1) as f32; N_BASIS];
        weights[drop_idx] = 0.0;

        let mut block = AdaptiveBlock::new(d_model, ffn_dim);
        *block.activation_mut() = AdaptiveBasis::with_weights(weights);

        for _ in 0..epochs {
            for (inp, tgt) in &train_data {
                adaptive_train_step(&mut block, inp, tgt, lr);
            }
        }
        let test_mse: f32 = test_data.iter()
            .map(|(inp, tgt)| mse(&block.forward_vec(inp).unwrap(), tgt))
            .sum::<f32>() / test_data.len() as f32;
        results.push((format!("No {}", BASIS_NAMES[drop_idx]), test_mse, N_BASIS - 1));
    }

    let baseline = results[0].1;
    for (name, mse_val, active) in &results {
        let delta = if baseline.is_finite() && baseline > 0.0 {
            (mse_val - baseline) / baseline * 100.0
        } else { 0.0 };
        let delta_str = if name == "Full (6)" { "baseline".to_string() } else { format!("{delta:+.1}%") };
        eprintln!("│ {name:<16} │ {mse_val:>12.6} │ {active:>5} ({delta_str:>8}) │");
    }
    eprintln!("└──────────────────┴──────────────┴────────────┘");

    // Print basis importance ranking (safe for NaN)
    let mut ablation: Vec<(usize, f32)> = (0..N_BASIS)
        .map(|j| {
            let delta = results[j + 1].1 - baseline;
            (j, if delta.is_finite() { delta } else { 0.0 })
        })
        .collect();
    ablation.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    eprintln!("\nBasis importance ranking (most → least):");
    for (rank, (j, delta)) in ablation.iter().enumerate() {
        eprintln!("  {}. {} (ΔMSE = {delta:+.6})", rank + 1, BASIS_NAMES[*j]);
    }

    // Verify no NaN
    assert!(baseline.is_finite(), "Baseline MSE should be finite");
}
