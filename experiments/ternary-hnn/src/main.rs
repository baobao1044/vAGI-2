//! Ternary HNN Experiment — Entry Point
//!
//! Usage:
//!   cargo run --release -- run-all     # Full experiment
//!   cargo run --release -- quick       # Quick smoke test

mod evaluator;
mod ground_truth;
mod models;
mod trainer;

use models::{HNNFP32, HNNTernary, HNNAdaptive, MLPFP32, HNNModel};
use ground_truth::Dataset;
use trainer::{TrainConfig, train_hnn_fp32, train_hnn_ternary, train_hnn_adaptive, train_mlp_fp32};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("quick");

    match mode {
        "quick" => run_quick(),
        "run-all" => run_all(),
        "train" => run_training(),
        _ => {
            eprintln!("Usage: ternary-hnn [quick|run-all|train]");
            std::process::exit(1);
        }
    }
}

/// Quick smoke test — verify everything works.
fn run_quick() {
    println!("=== Ternary HNN — Quick Smoke Test ===\n");

    let seeds = [42u64];
    let d_state = 1;
    let hidden = 16;
    let n_layers = 2;

    for &seed in &seeds {
        println!("--- Seed {seed} ---");

        // Create models
        let fp32 = HNNFP32::new(d_state, hidden, n_layers, seed);
        let ternary = HNNTernary::new(d_state, hidden, n_layers, seed);
        let adaptive = HNNAdaptive::new(d_state, hidden, n_layers, seed);
        let mlp = MLPFP32::new(d_state, hidden, n_layers, seed);

        // Quick forward pass check
        let state = vec![1.0f32, 0.0];
        println!("HNN-FP32     H={:.6}, params={}, mem={}B",
            fp32.hamiltonian(&state), fp32.param_count(), fp32.memory_bytes());
        println!("HNN-Ternary  H={:.6}, params={}, mem={}B",
            ternary.hamiltonian(&state), ternary.param_count(), ternary.memory_bytes());
        println!("HNN-Adaptive H={:.6}, params={}, mem={}B",
            adaptive.hamiltonian(&state), adaptive.param_count(), adaptive.memory_bytes());
        println!("MLP-FP32     pred={:?}, params={}, mem={}B",
            &mlp.predict_derivatives(&state)[..2], mlp.param_count(), mlp.memory_bytes());

        // Quick energy drift on harmonic
        let inits = vec![vec![1.0f32, 0.0]];
        let result = evaluator::evaluate_hnn(
            &fp32, "harmonic", seed,
            &ground_truth::harmonic::energy,
            &ground_truth::harmonic::derivatives,
            &inits, 0.01, &[1, 10, 100],
        );
        println!("\n  Energy drift (HNN-FP32, harmonic):");
        for (step, drift) in &result.energy_drifts {
            println!("    step {step:>5}: {drift:.6e}");
        }
        println!("  Inference: {:.1} µs/call", result.inference_us);
    }

    println!("\n=== Data generation test ===");
    let ds = Dataset::generate(
        100, 50, 0.01, 2,
        &[(-2.0, 2.0), (-2.0, 2.0)],
        &ground_truth::harmonic::derivatives, 42,
    );
    println!("Generated {} trajectories, {} steps each", ds.trajectories.len(), ds.trajectories[0].len() - 1);
    let pairs = ds.training_pairs(0.01);
    println!("Training pairs: {}", pairs.len());

    println!("\n✅ Quick test passed!");
}

/// Training smoke test — trains all 4 models on harmonic oscillator.
fn run_training() {
    println!("=== Training All 4 Models on Harmonic ===\n");

    let seed = 42u64;
    let d_state = 1;
    let hidden = 16;
    let n_layers = 2;
    let dt = 0.01;

    // Generate dataset
    let ds = Dataset::generate(
        50, 20, dt, 2,
        &[(-1.0, 1.0), (-1.0, 1.0)],
        &ground_truth::harmonic::derivatives, seed,
    );
    let (train_ds, val_ds) = ds.split(0.2);

    let config = TrainConfig {
        max_epochs: 20,
        batch_size: 16,
        print_every: 5,
        patience: 50,
        basis_warmup_epochs: 5,
        seed,
        ..Default::default()
    };

    println!("--- HNN-FP32 ---");
    let mut fp32 = HNNFP32::new(d_state, hidden, n_layers, seed);
    let r1 = train_hnn_fp32(&mut fp32, &train_ds, &val_ds, &config, "harmonic", dt);
    println!("  Final: train={:.6e} val={:.6e} best_val={:.6e}@{} ({:.1}s)\n",
        r1.final_train_loss, r1.final_val_loss, r1.best_val_loss, r1.best_epoch, r1.train_seconds);

    println!("--- HNN-Ternary ---");
    let mut ternary = HNNTernary::new(d_state, hidden, n_layers, seed);
    let r2 = train_hnn_ternary(&mut ternary, &train_ds, &val_ds, &config, "harmonic", dt);
    println!("  Final: train={:.6e} val={:.6e} best_val={:.6e}@{} ({:.1}s)\n",
        r2.final_train_loss, r2.final_val_loss, r2.best_val_loss, r2.best_epoch, r2.train_seconds);

    println!("--- HNN-Adaptive ---");
    let mut adaptive = HNNAdaptive::new(d_state, hidden, n_layers, seed);
    let r3 = train_hnn_adaptive(&mut adaptive, &train_ds, &val_ds, &config, "harmonic", dt);
    println!("  Final: train={:.6e} val={:.6e} best_val={:.6e}@{} ({:.1}s)\n",
        r3.final_train_loss, r3.final_val_loss, r3.best_val_loss, r3.best_epoch, r3.train_seconds);

    println!("--- MLP-FP32 ---");
    let mut mlp = MLPFP32::new(d_state, hidden, n_layers, seed);
    let r4 = train_mlp_fp32(&mut mlp, &train_ds, &val_ds, &config, "harmonic", dt);
    println!("  Final: train={:.6e} val={:.6e} best_val={:.6e}@{} ({:.1}s)\n",
        r4.final_train_loss, r4.final_val_loss, r4.best_val_loss, r4.best_epoch, r4.train_seconds);

    // Save CSV
    trainer::save_training_csv(&[r1, r2, r3, r4], "results/training_smoke.csv");
    println!("Results saved to results/training_smoke.csv");
    println!("\n✅ Training test done!");
}

/// Full experiment — train all models × all systems × all seeds, then evaluate.
fn run_all() {
    println!("=== Ternary HNN — Full Experiment (Train → Evaluate) ===\n");

    let seeds = [42u64, 123, 456];
    let hidden = 16;      // small model for feasible numerical gradient
    let n_layers = 2;
    let eval_steps = [1, 10, 100, 1000];
    let n_train = 200;    // training trajectories
    let n_eval = 50;      // eval initial conditions
    let traj_len = 50;    // steps per trajectory
    let dt = 0.01;

    let train_config = TrainConfig {
        max_epochs: 200,
        batch_size: 32,
        learning_rate: 1e-3,
        basis_lr: 1e-2,
        basis_warmup_epochs: 30,
        patience: 100,
        grad_eps: 1e-4,
        seed: 0, // will be overwritten per run
        print_every: 50,
    };

    // System configs: (name, state_dim, energy_fn, deriv_fn, train_init_range, eval_init_range)
    let systems: Vec<(&str, usize,
        Box<dyn Fn(&[f32])->f32>,
        Box<dyn Fn(&[f32])->Vec<f32>>,
        Vec<(f32,f32)>)> = vec![
        ("harmonic", 1,
         Box::new(ground_truth::harmonic::energy),
         Box::new(ground_truth::harmonic::derivatives),
         vec![(-2.0, 2.0), (-2.0, 2.0)]),
        ("pendulum", 1,
         Box::new(ground_truth::pendulum::energy),
         Box::new(ground_truth::pendulum::derivatives),
         vec![(-std::f32::consts::PI * 0.8, std::f32::consts::PI * 0.8), (-1.5, 1.5)]),
        ("double_pendulum", 2,
         Box::new(ground_truth::double_pendulum::energy),
         Box::new(ground_truth::double_pendulum::derivatives),
         vec![(-0.8, 0.8), (-0.8, 0.8), (-0.5, 0.5), (-0.5, 0.5)]),
    ];

    std::fs::create_dir_all("results").ok();

    // ── Experiment A: Energy Conservation ──────────────────────
    println!("╔══════════════════════════════════════╗");
    println!("║  EXPERIMENT A: Energy Conservation   ║");
    println!("╚══════════════════════════════════════╝\n");

    let mut eval_csv = vec![
        "model,system,seed,step,metric,param_count,value".to_string()
    ];
    let mut train_results = Vec::new();

    for (sys_name, d_state, energy_fn, deriv_fn, init_range) in &systems {
        println!("\n━━━ System: {sys_name} (d_state={d_state}) ━━━");

        // Generate training data
        let train_ds = Dataset::generate(
            n_train, traj_len, dt, 2 * d_state,
            init_range, deriv_fn.as_ref(), 777,
        );
        let (train_split, val_split) = train_ds.split(0.2);
        println!("  Data: {} train, {} val trajectories",
            train_split.trajectories.len(), val_split.trajectories.len());

        // Generate eval initial conditions (different seed from training)
        let eval_ds = Dataset::generate(
            n_eval, 1, dt, 2 * d_state,
            init_range, deriv_fn.as_ref(), 999,
        );
        let eval_inits: Vec<Vec<f32>> = eval_ds.trajectories.iter()
            .map(|t| t[0].clone()).collect();

        for &seed in &seeds {
            println!("\n  ── Seed {seed} ──");
            let mut config = train_config.clone();
            config.seed = seed;

            // ── Train + Evaluate HNN-FP32 ──
            println!("    Training HNN-FP32...");
            let mut fp32 = HNNFP32::new(*d_state, hidden, n_layers, seed);
            let r1 = train_hnn_fp32(&mut fp32, &train_split, &val_split, &config, sys_name, dt);
            println!("    → val_loss={:.4e} best@{} ({:.1}s)", r1.best_val_loss, r1.best_epoch, r1.train_seconds);
            train_results.push(r1);

            let result = evaluator::evaluate_hnn(
                &fp32, sys_name, seed,
                energy_fn.as_ref(), deriv_fn.as_ref(),
                &eval_inits, dt, &eval_steps,
            );
            print!("    Eval: ");
            for (step, drift) in &result.energy_drifts {
                print!("ΔE@{step}={drift:.2e} ");
            }
            println!("| {:.1}µs", result.inference_us);
            eval_csv.extend(result.to_csv_lines());

            // ── Train + Evaluate HNN-Ternary ──
            println!("    Training HNN-Ternary...");
            let mut ternary = HNNTernary::new(*d_state, hidden, n_layers, seed);
            let r2 = train_hnn_ternary(&mut ternary, &train_split, &val_split, &config, sys_name, dt);
            println!("    → val_loss={:.4e} best@{} ({:.1}s)", r2.best_val_loss, r2.best_epoch, r2.train_seconds);
            train_results.push(r2);

            let result = evaluator::evaluate_hnn(
                &ternary, sys_name, seed,
                energy_fn.as_ref(), deriv_fn.as_ref(),
                &eval_inits, dt, &eval_steps,
            );
            print!("    Eval: ");
            for (step, drift) in &result.energy_drifts {
                print!("ΔE@{step}={drift:.2e} ");
            }
            println!("| {:.1}µs", result.inference_us);
            eval_csv.extend(result.to_csv_lines());

            // ── Train + Evaluate HNN-Adaptive ──
            println!("    Training HNN-Adaptive...");
            let mut adaptive = HNNAdaptive::new(*d_state, hidden, n_layers, seed);
            let r3 = train_hnn_adaptive(&mut adaptive, &train_split, &val_split, &config, sys_name, dt);
            println!("    → val_loss={:.4e} best@{} ({:.1}s)", r3.best_val_loss, r3.best_epoch, r3.train_seconds);
            println!("    Basis weights: {:?}", adaptive.basis_weights());
            train_results.push(r3);

            let result = evaluator::evaluate_hnn(
                &adaptive, sys_name, seed,
                energy_fn.as_ref(), deriv_fn.as_ref(),
                &eval_inits, dt, &eval_steps,
            );
            print!("    Eval: ");
            for (step, drift) in &result.energy_drifts {
                print!("ΔE@{step}={drift:.2e} ");
            }
            println!("| {:.1}µs", result.inference_us);
            eval_csv.extend(result.to_csv_lines());

            // ── Train + Evaluate MLP-FP32 ──
            println!("    Training MLP-FP32...");
            let mut mlp = MLPFP32::new(*d_state, hidden, n_layers, seed);
            let r4 = train_mlp_fp32(&mut mlp, &train_split, &val_split, &config, sys_name, dt);
            println!("    → val_loss={:.4e} best@{} ({:.1}s)", r4.best_val_loss, r4.best_epoch, r4.train_seconds);
            train_results.push(r4);

            // MLP energy evaluation via Euler integration
            let mut mlp_drift_total = 0.0f32;
            for init in &eval_inits {
                let e0 = energy_fn(init);
                let mut state = init.clone();
                let max_step = *eval_steps.iter().max().unwrap_or(&100);
                for _ in 0..max_step {
                    let deriv = mlp.predict_derivatives(&state);
                    for j in 0..state.len() {
                        state[j] += dt * deriv[j];
                    }
                }
                let ef = energy_fn(&state);
                mlp_drift_total += (ef - e0).abs() / e0.abs().max(1e-10);
            }
            let mlp_avg_drift = mlp_drift_total / eval_inits.len() as f32;
            let mlp_speed = evaluator::inference_speed_mlp(&mlp, &eval_inits[0], 1000);
            println!("    Eval: ΔE@{}={:.2e} | {:.1}µs (Euler, no Hamiltonian)",
                eval_steps.last().unwrap(), mlp_avg_drift, mlp_speed);

            eval_csv.push(format!("MLP-FP32,{sys_name},{seed},{},energy_drift,{},{:.6e}",
                eval_steps.last().unwrap(), mlp.param_count(), mlp_avg_drift));
            eval_csv.push(format!("MLP-FP32,{sys_name},{seed},0,inference_us,{},{:.2}",
                mlp.param_count(), mlp_speed));
            eval_csv.push(format!("MLP-FP32,{sys_name},{seed},0,memory_bytes,{},{}",
                mlp.param_count(), mlp.memory_bytes()));
        }
    }

    // Save results
    let eval_content = eval_csv.join("\n");
    std::fs::write("results/experiment_A.csv", &eval_content)
        .expect("Failed to write experiment_A.csv");
    println!("\n\n═══ Results saved to results/experiment_A.csv ═══");

    trainer::save_training_csv(&train_results, "results/training_history.csv");
    println!("Training history saved to results/training_history.csv");

    // ── Summary Table ──────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║                    SUMMARY TABLE                     ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║ {:<15} {:>8} {:>8} {:>12}   ║", "Model", "Params", "Memory", "Best Val Loss");
    println!("╠══════════════════════════════════════════════════════╣");

    let fp32 = HNNFP32::new(1, hidden, n_layers, 42);
    let ternary = HNNTernary::new(1, hidden, n_layers, 42);
    let adaptive = HNNAdaptive::new(1, hidden, n_layers, 42);
    let mlp = MLPFP32::new(1, hidden, n_layers, 42);

    println!("║ {:<15} {:>8} {:>6}B {:>12}   ║", "HNN-FP32", fp32.param_count(), fp32.memory_bytes(), "see CSV");
    println!("║ {:<15} {:>8} {:>6}B {:>12}   ║", "HNN-Ternary", ternary.param_count(), ternary.memory_bytes(), "see CSV");
    println!("║ {:<15} {:>8} {:>6}B {:>12}   ║", "HNN-Adaptive", adaptive.param_count(), adaptive.memory_bytes(), "see CSV");
    println!("║ {:<15} {:>8} {:>6}B {:>12}   ║", "MLP-FP32", mlp.param_count(), mlp.memory_bytes(), "see CSV");
    println!("╚══════════════════════════════════════════════════════╝");

    let total_runs = train_results.len();
    let total_time: f64 = train_results.iter().map(|r| r.train_seconds).sum();
    println!("\nTotal: {} training runs in {:.0}s ({:.1} min)",
        total_runs, total_time, total_time / 60.0);
    println!("\n✅ Full experiment complete!");
}
