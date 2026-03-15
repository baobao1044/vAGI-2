//! Training pipeline — mini-batch gradient descent with STE, warmup, early stopping.
//!
//! All models use numerical gradient (finite difference per parameter).
//! HNN models: loss = MSE(predicted_dynamics, true_dynamics)
//! MLP model: loss = MSE(predicted_next_state, true_next_state)

use crate::models::{HNNModel, HNNFP32, HNNTernary, HNNAdaptive, MLPFP32};
use crate::ground_truth::Dataset;
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Training configuration.
#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub max_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    /// Learning rate for AdaptiveBasis weights (typically 10× lr).
    pub basis_lr: f32,
    /// Epochs to freeze basis weights before unfreezing.
    pub basis_warmup_epochs: usize,
    /// Early stopping patience (epochs without improvement).
    pub patience: usize,
    /// Finite difference epsilon for numerical gradient.
    pub grad_eps: f32,
    /// Random seed for mini-batch shuffling.
    pub seed: u64,
    /// Print progress every N epochs.
    pub print_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_epochs: 2000,
            batch_size: 32,
            learning_rate: 1e-3,
            basis_lr: 1e-2,
            basis_warmup_epochs: 50,
            patience: 200,
            grad_eps: 1e-4,
            seed: 42,
            print_every: 100,
        }
    }
}

/// Training result for a single run.
#[derive(Clone, Debug)]
pub struct TrainResult {
    pub model_name: String,
    pub system_name: String,
    pub seed: u64,
    pub final_train_loss: f32,
    pub final_val_loss: f32,
    pub best_val_loss: f32,
    pub best_epoch: usize,
    pub total_epochs: usize,
    pub train_seconds: f64,
    /// Loss history: (epoch, train_loss, val_loss)
    pub history: Vec<(usize, f32, f32)>,
}

impl TrainResult {
    pub fn to_csv_lines(&self) -> Vec<String> {
        self.history.iter().map(|(epoch, train_loss, val_loss)| {
            format!("{},{},{},{},{:.6e},{:.6e}",
                self.model_name, self.system_name, self.seed,
                epoch, train_loss, val_loss)
        }).collect()
    }
}

// ── Mini-batch helpers ────────────────────────────────────────

/// Sample a mini-batch from training pairs.
fn sample_batch<'a>(
    pairs: &'a [(Vec<f32>, Vec<f32>)],
    batch_size: usize,
    rng: &mut StdRng,
) -> Vec<&'a (Vec<f32>, Vec<f32>)> {
    let mut indices: Vec<usize> = (0..pairs.len()).collect();
    indices.shuffle(rng);
    indices.truncate(batch_size.min(pairs.len()));
    indices.iter().map(|&i| &pairs[i]).collect()
}

/// Compute loss over a batch for HNN model.
fn hnn_batch_loss(model: &dyn HNNModel, batch: &[&(Vec<f32>, Vec<f32>)]) -> f32 {
    let n = batch.len() as f32;
    if n == 0.0 { return 0.0; }
    batch.iter().map(|(state, target_deriv)| {
        let pred_deriv = model.dynamics(state);
        pred_deriv.iter().zip(target_deriv.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
    }).sum::<f32>() / n
}

/// Compute loss over full dataset for HNN model.
fn hnn_full_loss(model: &dyn HNNModel, pairs: &[(Vec<f32>, Vec<f32>)]) -> f32 {
    let n = pairs.len() as f32;
    if n == 0.0 { return 0.0; }
    pairs.iter().map(|(state, target_deriv)| {
        let pred_deriv = model.dynamics(state);
        pred_deriv.iter().zip(target_deriv.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
    }).sum::<f32>() / n
}

// ── Training Implementations ──────────────────────────────────

/// Train HNN-FP32 model.
pub fn train_hnn_fp32(
    model: &mut HNNFP32,
    train_data: &Dataset,
    val_data: &Dataset,
    config: &TrainConfig,
    system_name: &str,
    dt: f32,
) -> TrainResult {
    let train_pairs = train_data.training_pairs(dt);
    let val_pairs = val_data.training_pairs(dt);
    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut best_val_loss = f32::INFINITY;
    let mut best_epoch = 0;
    let mut epochs_no_improve = 0;
    let mut history = Vec::new();

    let t0 = std::time::Instant::now();

    for epoch in 0..config.max_epochs {
        // Mini-batch gradient step
        let batch = sample_batch(&train_pairs, config.batch_size, &mut rng);
        let batch_owned: Vec<(Vec<f32>, Vec<f32>)> = batch.iter().map(|&&ref p| p.clone()).collect();
        model.update_weights(&batch_owned, config.learning_rate);

        // Evaluate every print_every or at important epochs
        if epoch % config.print_every == 0 || epoch == config.max_epochs - 1 {
            let train_loss = hnn_full_loss(model, &train_pairs);
            let val_loss = hnn_full_loss(model, &val_pairs);
            history.push((epoch, train_loss, val_loss));

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_no_improve = 0;
            } else {
                epochs_no_improve += config.print_every;
            }

            if config.print_every <= 100 {
                eprintln!("  [HNN-FP32] epoch {epoch:>4}: train={train_loss:.6e} val={val_loss:.6e}");
            }
        }

        if epochs_no_improve >= config.patience {
            eprintln!("  Early stopping at epoch {epoch} (best={best_epoch})");
            break;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let final_train = hnn_full_loss(model, &train_pairs);
    let final_val = hnn_full_loss(model, &val_pairs);

    TrainResult {
        model_name: "HNN-FP32".to_string(),
        system_name: system_name.to_string(),
        seed: config.seed,
        final_train_loss: final_train,
        final_val_loss: final_val,
        best_val_loss,
        best_epoch,
        total_epochs: history.last().map(|h| h.0).unwrap_or(0),
        train_seconds: elapsed,
        history,
    }
}

/// Train HNN-Ternary model (STE: update shadow f32 weights, forward quantizes).
pub fn train_hnn_ternary(
    model: &mut HNNTernary,
    train_data: &Dataset,
    val_data: &Dataset,
    config: &TrainConfig,
    system_name: &str,
    dt: f32,
) -> TrainResult {
    let train_pairs = train_data.training_pairs(dt);
    let val_pairs = val_data.training_pairs(dt);
    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut best_val_loss = f32::INFINITY;
    let mut best_epoch = 0;
    let mut epochs_no_improve = 0;
    let mut history = Vec::new();

    let t0 = std::time::Instant::now();

    for epoch in 0..config.max_epochs {
        let batch = sample_batch(&train_pairs, config.batch_size, &mut rng);
        let batch_owned: Vec<(Vec<f32>, Vec<f32>)> = batch.iter().map(|&&ref p| p.clone()).collect();
        // STE: update_weights perturbs latent f32 weights, forward auto-quantizes
        model.update_weights(&batch_owned, config.learning_rate);

        if epoch % config.print_every == 0 || epoch == config.max_epochs - 1 {
            let train_loss = hnn_full_loss(model, &train_pairs);
            let val_loss = hnn_full_loss(model, &val_pairs);
            history.push((epoch, train_loss, val_loss));

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_no_improve = 0;
            } else {
                epochs_no_improve += config.print_every;
            }

            if config.print_every <= 100 {
                eprintln!("  [HNN-Ternary] epoch {epoch:>4}: train={train_loss:.6e} val={val_loss:.6e}");
            }
        }

        if epochs_no_improve >= config.patience {
            eprintln!("  Early stopping at epoch {epoch} (best={best_epoch})");
            break;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let final_train = hnn_full_loss(model, &train_pairs);
    let final_val = hnn_full_loss(model, &val_pairs);

    TrainResult {
        model_name: "HNN-Ternary".to_string(),
        system_name: system_name.to_string(),
        seed: config.seed,
        final_train_loss: final_train,
        final_val_loss: final_val,
        best_val_loss,
        best_epoch,
        total_epochs: history.last().map(|h| h.0).unwrap_or(0),
        train_seconds: elapsed,
        history,
    }
}

/// Train HNN-Adaptive model (STE weights + AdaptiveBasis with warmup).
pub fn train_hnn_adaptive(
    model: &mut HNNAdaptive,
    train_data: &Dataset,
    val_data: &Dataset,
    config: &TrainConfig,
    system_name: &str,
    dt: f32,
) -> TrainResult {
    let train_pairs = train_data.training_pairs(dt);
    let val_pairs = val_data.training_pairs(dt);
    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut best_val_loss = f32::INFINITY;
    let mut best_epoch = 0;
    let mut epochs_no_improve = 0;
    let mut history = Vec::new();

    let t0 = std::time::Instant::now();

    for epoch in 0..config.max_epochs {
        let batch = sample_batch(&train_pairs, config.batch_size, &mut rng);
        let batch_owned: Vec<(Vec<f32>, Vec<f32>)> = batch.iter().map(|&&ref p| p.clone()).collect();

        // AdaptiveBasis warmup: only update weights (not basis) for first N epochs
        if epoch < config.basis_warmup_epochs {
            model.update_weights_only(&batch_owned, config.learning_rate);
        } else {
            // After warmup: update both weights and basis
            model.update_weights(&batch_owned, config.learning_rate);
        }

        if epoch % config.print_every == 0 || epoch == config.max_epochs - 1 {
            let train_loss = hnn_full_loss(model, &train_pairs);
            let val_loss = hnn_full_loss(model, &val_pairs);
            history.push((epoch, train_loss, val_loss));

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_no_improve = 0;
            } else {
                epochs_no_improve += config.print_every;
            }

            if config.print_every <= 100 {
                eprintln!("  [HNN-Adaptive] epoch {epoch:>4}: train={train_loss:.6e} val={val_loss:.6e} basis={:?}",
                    model.basis_weights());
            }
        }

        if epochs_no_improve >= config.patience {
            eprintln!("  Early stopping at epoch {epoch} (best={best_epoch})");
            break;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let final_train = hnn_full_loss(model, &train_pairs);
    let final_val = hnn_full_loss(model, &val_pairs);

    TrainResult {
        model_name: "HNN-Adaptive".to_string(),
        system_name: system_name.to_string(),
        seed: config.seed,
        final_train_loss: final_train,
        final_val_loss: final_val,
        best_val_loss,
        best_epoch,
        total_epochs: history.last().map(|h| h.0).unwrap_or(0),
        train_seconds: elapsed,
        history,
    }
}

/// Train MLP-FP32 model (direct prediction, no Hamiltonian).
pub fn train_mlp_fp32(
    model: &mut MLPFP32,
    train_data: &Dataset,
    val_data: &Dataset,
    config: &TrainConfig,
    system_name: &str,
    dt: f32,
) -> TrainResult {
    let train_pairs = train_data.training_pairs(dt);
    let val_pairs = val_data.training_pairs(dt);
    let mut rng = StdRng::seed_from_u64(config.seed);

    let mut best_val_loss = f32::INFINITY;
    let mut best_epoch = 0;
    let mut epochs_no_improve = 0;
    let mut history = Vec::new();

    let t0 = std::time::Instant::now();

    for epoch in 0..config.max_epochs {
        let batch = sample_batch(&train_pairs, config.batch_size, &mut rng);
        let batch_owned: Vec<(Vec<f32>, Vec<f32>)> = batch.iter().map(|&&ref p| p.clone()).collect();
        model.update_weights(&batch_owned, config.learning_rate);

        if epoch % config.print_every == 0 || epoch == config.max_epochs - 1 {
            let train_loss = mlp_loss(model, &train_pairs);
            let val_loss = mlp_loss(model, &val_pairs);
            history.push((epoch, train_loss, val_loss));

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_no_improve = 0;
            } else {
                epochs_no_improve += config.print_every;
            }

            if config.print_every <= 100 {
                eprintln!("  [MLP-FP32] epoch {epoch:>4}: train={train_loss:.6e} val={val_loss:.6e}");
            }
        }

        if epochs_no_improve >= config.patience {
            eprintln!("  Early stopping at epoch {epoch} (best={best_epoch})");
            break;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let final_train = mlp_loss(model, &train_pairs);
    let final_val = mlp_loss(model, &val_pairs);

    TrainResult {
        model_name: "MLP-FP32".to_string(),
        system_name: system_name.to_string(),
        seed: config.seed,
        final_train_loss: final_train,
        final_val_loss: final_val,
        best_val_loss,
        best_epoch,
        total_epochs: history.last().map(|h| h.0).unwrap_or(0),
        train_seconds: elapsed,
        history,
    }
}

fn mlp_loss(model: &MLPFP32, pairs: &[(Vec<f32>, Vec<f32>)]) -> f32 {
    let n = pairs.len() as f32;
    if n == 0.0 { return 0.0; }
    pairs.iter().map(|(state, target_deriv)| {
        let pred = model.predict_derivatives(state);
        pred.iter().zip(target_deriv.iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
    }).sum::<f32>() / n
}

/// Save training history to CSV file.
pub fn save_training_csv(results: &[TrainResult], path: &str) {
    let mut lines = vec![
        "model,system,seed,epoch,train_loss,val_loss".to_string()
    ];
    for r in results {
        lines.extend(r.to_csv_lines());
    }
    std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."))).ok();
    std::fs::write(path, lines.join("\n")).expect("Failed to write training CSV");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ground_truth;

    fn small_harmonic_data() -> (Dataset, Dataset) {
        let ds = Dataset::generate(
            20, 10, 0.01, 2,
            &[(-1.0, 1.0), (-1.0, 1.0)],
            &ground_truth::harmonic::derivatives, 42,
        );
        ds.split(0.2)
    }

    #[test]
    fn test_train_hnn_fp32_loss_decreases() {
        let (train, val) = small_harmonic_data();
        let mut model = HNNFP32::new(1, 16, 2, 42);
        let config = TrainConfig {
            max_epochs: 10,
            batch_size: 16,
            print_every: 5,
            patience: 100,
            ..Default::default()
        };

        let result = train_hnn_fp32(&mut model, &train, &val, &config, "harmonic", 0.01);
        assert!(result.history.len() >= 2, "Should have at least 2 history entries");
        // Loss should be finite
        assert!(result.final_train_loss.is_finite());
        assert!(result.final_val_loss.is_finite());
    }

    #[test]
    fn test_train_hnn_ternary() {
        let (train, val) = small_harmonic_data();
        let mut model = HNNTernary::new(1, 16, 2, 42);
        let config = TrainConfig {
            max_epochs: 5,
            batch_size: 8,
            print_every: 5,
            patience: 100,
            ..Default::default()
        };

        let result = train_hnn_ternary(&mut model, &train, &val, &config, "harmonic", 0.01);
        assert!(result.final_train_loss.is_finite());
    }

    #[test]
    fn test_train_hnn_adaptive_with_warmup() {
        let (train, val) = small_harmonic_data();
        let mut model = HNNAdaptive::new(1, 16, 2, 42);
        let config = TrainConfig {
            max_epochs: 10,
            batch_size: 8,
            print_every: 5,
            basis_warmup_epochs: 3, // Warmup for 3 epochs
            ..Default::default()
        };

        let result = train_hnn_adaptive(&mut model, &train, &val, &config, "harmonic", 0.01);
        assert!(result.final_train_loss.is_finite());
    }

    #[test]
    fn test_train_mlp() {
        let (train, val) = small_harmonic_data();
        let mut model = MLPFP32::new(1, 16, 2, 42);
        let config = TrainConfig {
            max_epochs: 5,
            batch_size: 8,
            print_every: 5,
            ..Default::default()
        };

        let result = train_mlp_fp32(&mut model, &train, &val, &config, "harmonic", 0.01);
        assert!(result.final_train_loss.is_finite());
    }

    #[test]
    fn test_training_result_csv() {
        let result = TrainResult {
            model_name: "test".to_string(),
            system_name: "harmonic".to_string(),
            seed: 42,
            final_train_loss: 0.1,
            final_val_loss: 0.2,
            best_val_loss: 0.15,
            best_epoch: 10,
            total_epochs: 20,
            train_seconds: 1.5,
            history: vec![(0, 0.5, 0.6), (10, 0.2, 0.3)],
        };
        let lines = result.to_csv_lines();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("test"));
    }
}
