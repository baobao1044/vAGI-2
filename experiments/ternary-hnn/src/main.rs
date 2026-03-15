//! Ternary HNN Experiment — Entry Point
//!
//! Usage:
//!   cargo run --release -- run-all     # Full experiment
//!   cargo run --release -- quick       # Quick smoke test

mod evaluator;
mod ground_truth;
mod models;

use models::{HNNFP32, HNNTernary, HNNAdaptive, MLPFP32, HNNModel};
use ground_truth::Dataset;

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

/// Training smoke test.
fn run_training() {
    println!("=== Training Test — HNN-FP32 on Harmonic ===\n");

    let seed = 42u64;
    let d_state = 1;
    let hidden = 16;
    let n_layers = 2;

    let mut model = HNNFP32::new(d_state, hidden, n_layers, seed);

    // Generate small dataset
    let ds = Dataset::generate(
        20, 20, 0.01, 2,
        &[(-1.0, 1.0), (-1.0, 1.0)],
        &ground_truth::harmonic::derivatives, seed,
    );
    let pairs = ds.training_pairs(0.01);
    // Use only a small subset for speed
    let mini_batch: Vec<_> = pairs.iter().take(10).cloned().collect();

    println!("Training with {} pairs, {} batch...\n", pairs.len(), mini_batch.len());

    for epoch in 0..5 {
        let loss = model.compute_loss(&mini_batch);
        println!("Epoch {epoch}: loss = {loss:.6}");
        model.update_weights(&mini_batch, 0.001);
    }

    println!("\n✅ Training test done!");
}

/// Full experiment — all models × all systems × all seeds.
fn run_all() {
    println!("=== Ternary HNN — Full Experiment ===\n");

    let seeds = [42u64, 123, 456];
    let d_state_1d = 1;
    let d_state_2d = 2;
    let hidden = 64;
    let n_layers = 3;
    let eval_steps = [1, 10, 100, 1000];
    let n_eval = 20; // initial conditions for evaluation
    let dt = 0.01;

    println!("Generating datasets...");

    // System configs: (name, state_dim, energy_fn, deriv_fn, initial_range)
    let systems: Vec<(&str, usize, Box<dyn Fn(&[f32])->f32>, Box<dyn Fn(&[f32])->Vec<f32>>, Vec<(f32,f32)>)> = vec![
        ("harmonic", 1,
         Box::new(ground_truth::harmonic::energy),
         Box::new(ground_truth::harmonic::derivatives),
         vec![(-2.0, 2.0), (-2.0, 2.0)]),
        ("pendulum", 1,
         Box::new(ground_truth::pendulum::energy),
         Box::new(ground_truth::pendulum::derivatives),
         vec![(-std::f32::consts::PI, std::f32::consts::PI), (-2.0, 2.0)]),
        ("double_pendulum", 2,
         Box::new(ground_truth::double_pendulum::energy),
         Box::new(ground_truth::double_pendulum::derivatives),
         vec![(-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)]),
    ];

    // CSV header
    let mut csv_lines = vec![
        "model,system,seed,step,metric,param_count,value".to_string()
    ];

    for (sys_name, d_state, energy_fn, deriv_fn, init_range) in &systems {
        println!("\n--- System: {sys_name} ---");

        // Generate evaluation initial conditions
        let eval_ds = Dataset::generate(
            n_eval, 1, dt, 2 * d_state,
            init_range, deriv_fn.as_ref(), 999,
        );
        let eval_inits: Vec<Vec<f32>> = eval_ds.trajectories.iter()
            .map(|t| t[0].clone()).collect();

        for &seed in &seeds {
            println!("  Seed {seed}:");

            // Create all 4 models
            let fp32 = HNNFP32::new(*d_state, hidden, n_layers, seed);
            let ternary = HNNTernary::new(*d_state, hidden, n_layers, seed);
            let adaptive = HNNAdaptive::new(*d_state, hidden, n_layers, seed);

            // Evaluate HNN models
            let hnn_models: Vec<(&str, &dyn HNNModel)> = vec![
                ("HNN-FP32", &fp32),
                ("HNN-Ternary", &ternary),
                ("HNN-Adaptive", &adaptive),
            ];

            for (name, model) in &hnn_models {
                let result = evaluator::evaluate_hnn(
                    *model, sys_name, seed,
                    energy_fn.as_ref(), deriv_fn.as_ref(),
                    &eval_inits, dt, &eval_steps,
                );
                print!("    {name}: ");
                for (step, drift) in &result.energy_drifts {
                    print!("ΔE@{step}={drift:.2e} ");
                }
                println!("| {:.0}µs | {}B", result.inference_us, result.memory_bytes);
                csv_lines.extend(result.to_csv_lines());
            }

            // MLP-FP32: evaluate differently (no Hamiltonian)
            let mlp = MLPFP32::new(*d_state, hidden, n_layers, seed);
            // For MLP, evaluate trajectory via direct integration
            let test_state = &eval_inits[0];
            let mlp_speed = evaluator::inference_speed_mlp(&mlp, test_state, 1000);
            let mem = mlp.memory_bytes();
            let params = mlp.param_count();
            println!("    MLP-FP32: {:.0}µs | {mem}B (no energy structure)", mlp_speed);
            csv_lines.push(format!("MLP-FP32,{sys_name},{seed},0,inference_us,{params},{mlp_speed:.2}"));
            csv_lines.push(format!("MLP-FP32,{sys_name},{seed},0,memory_bytes,{params},{mem}"));
        }
    }

    // Save CSV
    let csv_content = csv_lines.join("\n");
    std::fs::create_dir_all("results").ok();
    std::fs::write("results/experiment_A.csv", &csv_content)
        .expect("Failed to write CSV");
    println!("\n\nResults saved to results/experiment_A.csv");
    println!("Total CSV lines: {}", csv_lines.len());

    // Print summary table
    println!("\n=== SUMMARY ===");
    println!("{:<15} {:>10} {:>10}", "Model", "Params", "Memory");
    let fp32 = HNNFP32::new(1, hidden, n_layers, 42);
    let ternary = HNNTernary::new(1, hidden, n_layers, 42);
    let adaptive = HNNAdaptive::new(1, hidden, n_layers, 42);
    let mlp = MLPFP32::new(1, hidden, n_layers, 42);
    println!("{:<15} {:>10} {:>8}B", "HNN-FP32", fp32.param_count(), fp32.memory_bytes());
    println!("{:<15} {:>10} {:>8}B", "HNN-Ternary", ternary.param_count(), ternary.memory_bytes());
    println!("{:<15} {:>10} {:>8}B", "HNN-Adaptive", adaptive.param_count(), adaptive.memory_bytes());
    println!("{:<15} {:>10} {:>8}B", "MLP-FP32", mlp.param_count(), mlp.memory_bytes());
    println!("\n✅ Full experiment complete!");
}
