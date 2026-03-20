//! Train vagi-lm on Vietnamese text — Breakthrough Edition.
//!
//! Features:
//! - --fast: batch-parallel f32 training (no ternary overhead)
//! - --curriculum: progressive seq_len (16→32→64→128)
//! - Configurable model size: --tiny, --small, --base
//!
//! Usage:
//!   cargo run --example train_vietnamese -p vagi-lm --release -- --fast
//!   cargo run --example train_vietnamese -p vagi-lm --release -- --small --fast
//!   cargo run --example train_vietnamese -p vagi-lm --release -- --small --curriculum

use std::io::Write;
use std::time::Instant;
use vagi_lm::{VagiLM, LMConfig, LMTrainer, AdvancedConfig, TextDataset};
use vagi_lm::tokenizer::ByteTokenizer;
use vagi_lm::checkpoint;
use vagi_lm::fast_train;

const DATA_PATH: &str = "data/vi_sentences.txt";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_base = args.iter().any(|a| a == "--base");
    let use_small = args.iter().any(|a| a == "--small") && !use_base;
    let use_curriculum = args.iter().any(|a| a == "--curriculum");
    let use_fast = args.iter().any(|a| a == "--fast");
    let batch_size: usize = args.windows(2)
        .find(|w| w[0] == "--batch")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(8);
    let config_name = if use_base { "BASE" } else if use_small { "SMALL" } else { "TINY" };

    let n_epochs: usize = args.windows(2)
        .find(|w| w[0] == "--epochs")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(if use_curriculum { 30 } else if use_base { 15 } else if use_small { 20 } else { 10 });

    let start = Instant::now();
    println!("═══════════════════════════════════════════");
    println!("  vAGI-2 Vietnamese Training (Breakthrough)");
    println!("═══════════════════════════════════════════");
    flush();

    // ── Load data ──
    print!("[1/4] Loading data... ");
    flush();
    let text = match std::fs::read_to_string(DATA_PATH) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: {e}\nRun: powershell -File scripts/download_vi_large.ps1");
            std::process::exit(1);
        }
    };
    let sentences: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| l.len() >= 5).collect();
    println!("{} sentences", sentences.len());
    for (i, s) in sentences.iter().take(3).enumerate() {
        let preview: String = s.chars().take(60).collect();
        println!("  [{i}] {preview}...");
    }
    flush();

    // ── Model config ──
    let config = if use_base {
        LMConfig::base()
    } else if use_small {
        LMConfig::small()
    } else {
        LMConfig::tiny()
    };
    let mut model = VagiLM::new(config.clone());
    let tok = ByteTokenizer::new();

    println!("[2/4] Model: d={} L={} H={} (~{}K params) [{}]",
        config.d_model, config.n_layers, config.n_heads,
        config.param_count() / 1000, config_name);
    flush();

    // ── Training ──
    if use_fast {
        let seq_len = if use_base { 128 } else if use_small { 64 } else { 32 };
        println!("[3/4] FAST Training (f32, batch={batch_size}, seq_len={seq_len})");
        println!("       No ternary overhead | Batch-parallel via rayon");
        flush();
        train_fast(&mut model, &sentences, n_epochs, seq_len, batch_size, &tok, &config);
    } else if use_curriculum {
        println!("[3/4] Curriculum Training (progressive seq_len)");
        println!("       Total budget: {n_epochs} epochs");
        flush();
        train_curriculum(&mut model, &sentences, n_epochs, &tok, &config);
    } else {
        let seq_len = if use_base { 256 } else if use_small { 128 } else { 64 };
        println!("[3/4] Standard Training (seq_len={seq_len})");
        flush();
        train_standard(&mut model, &sentences, n_epochs, seq_len, &tok, &config);
    }

    // ── Save final ──
    println!("[4/4] Saving final model...");
    match checkpoint::save_model(&model, "data/vi_model.bin") {
        Ok(sz) => println!("  data/vi_model.bin ({:.1} KB)", sz as f64 / 1024.0),
        Err(e) => eprintln!("  Save error: {e}"),
    }

    let total_time = start.elapsed().as_secs_f32();
    println!("\n═══ Done in {:.0}s ({:.1} min) ═══", total_time, total_time / 60.0);
    println!("\nTo chat: cargo run --example chat_vi -p vagi-lm --release");
}

/// Curriculum training: progressive seq_len for massive speedup.
///
/// Phase 1: seq_len=16  (64x faster attention) — learn character patterns
/// Phase 2: seq_len=32  (16x faster) — learn word patterns  
/// Phase 3: seq_len=64  (4x faster) — learn phrases
/// Phase 4: seq_len=128 (baseline) — learn sentence structure
fn train_curriculum(
    model: &mut VagiLM,
    sentences: &[&str],
    total_epochs: usize,
    tok: &ByteTokenizer,
    config: &LMConfig,
) {
    // Distribute epochs across phases (weighted toward longer seqs)
    let phases: Vec<(usize, usize)> = if total_epochs >= 20 {
        vec![
            (16,  total_epochs * 15 / 100),  // 15% of epochs
            (32,  total_epochs * 20 / 100),  // 20%
            (64,  total_epochs * 30 / 100),  // 30%
            (128, total_epochs * 35 / 100),  // 35%
        ]
    } else {
        vec![
            (16,  (total_epochs / 4).max(1)),
            (32,  (total_epochs / 4).max(1)),
            (64,  (total_epochs / 4).max(1)),
            (128, (total_epochs - 3 * (total_epochs / 4).max(1))),
        ]
    };

    println!("  Phases:");
    for (i, (sl, ep)) in phases.iter().enumerate() {
        let speedup = (128 * 128) / (sl * sl);
        println!("    Phase {}: seq_len={:3}  epochs={:2}  (~{}x faster)", i+1, sl, ep, speedup);
    }
    println!();
    flush();

    let mut _global_step = 0usize;
    let mut rng = rand::thread_rng();

    for (phase_idx, (seq_len, phase_epochs)) in phases.iter().enumerate() {
        let seq_len = *seq_len;
        let phase_epochs = *phase_epochs;
        if phase_epochs == 0 { continue; }

        println!("  ┌─ Phase {} ─ seq_len={} ─ {} epochs ────────────────", phase_idx+1, seq_len, phase_epochs);
        flush();

        // Create dataset for this phase's seq_len
        let mut dataset = TextDataset::from_samples(sentences, seq_len);
        let n_seqs = dataset.len();
        println!("  │ {} sequences", n_seqs);
        flush();

        // Compute total steps for LR scheduling
        let phase_total_steps = phase_epochs * n_seqs;
        let lr_base = if config.d_model >= 512 { 0.001 }
            else if config.d_model >= 256 { 0.003 }
            else { 0.01 };

        // Scale LR: shorter seqs can use higher LR
        let lr = lr_base * (128.0 / seq_len as f32).sqrt();

        let train_cfg = AdvancedConfig {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            grad_clip: 1.0,
            warmup_steps: (phase_total_steps / 20).max(50),
            total_steps: phase_total_steps,
            label_smoothing: 0.05,
        };
        let mut trainer = LMTrainer::new(model, train_cfg);

        for epoch in 0..phase_epochs {
            let epoch_start = Instant::now();
            dataset.shuffle(&mut rng);
            let seqs = dataset.sequences();
            let n = seqs.len();

            let mut epoch_loss = 0.0f32;
            let mut epoch_acc = 0.0f32;
            let mut count = 0;

            for (i, seq) in seqs.iter().enumerate() {
                if seq.len() < 3 { continue; }
                let m = trainer.train_step(model, seq);
                epoch_loss += m.loss;
                epoch_acc += m.accuracy;
                count += 1;
                _global_step += 1;

                if (i + 1) % 500 == 0 || i + 1 == n {
                    let avg_loss = epoch_loss / count as f32;
                    let avg_acc = epoch_acc / count as f32 * 100.0;
                    let elapsed = epoch_start.elapsed().as_secs_f32();
                    let sps = (i + 1) as f32 / elapsed;
                    print!(
                        "\r  │ P{} E{}/{} [{:5}/{:5}] loss={:.3} ppl={:.1} acc={:.1}% lr={:.5} ({:.1} sps)",
                        phase_idx+1, epoch+1, phase_epochs, i+1, n,
                        avg_loss, avg_loss.exp(), avg_acc, m.lr, sps,
                    );
                    flush();
                }
            }

            let avg_loss = epoch_loss / count.max(1) as f32;
            let avg_acc = epoch_acc / count.max(1) as f32 * 100.0;
            let t = epoch_start.elapsed().as_secs_f32();
            let sps = count as f32 / t;
            println!(
                "\n  │ ✓ Epoch {} — loss={:.4} ppl={:.2} acc={:.1}% ({:.1}s, {:.1} sps)",
                epoch+1, avg_loss, avg_loss.exp(), avg_acc, t, sps,
            );

            // Generate sample at end of each phase
            if epoch + 1 == phase_epochs {
                generate_samples(model, tok);
            }
            flush();
        }

        // Save checkpoint after phase
        let ckpt = format!("data/vi_ckpt_p{}_s{}.bin", phase_idx+1, seq_len);
        match checkpoint::save_model(model, &ckpt) {
            Ok(sz) => println!("  │ Saved {} ({:.1} KB)", ckpt, sz as f64 / 1024.0),
            Err(e) => eprintln!("  │ Save error: {e}"),
        }
        println!("  └────────────────────────────────────────────");
        println!();
        flush();
    }
}

/// Standard training (fixed seq_len).
fn train_standard(
    model: &mut VagiLM,
    sentences: &[&str],
    n_epochs: usize,
    seq_len: usize,
    tok: &ByteTokenizer,
    config: &LMConfig,
) {
    let mut dataset = TextDataset::from_samples(sentences, seq_len);
    println!("  {} sequences (seq_len={seq_len})", dataset.len());
    flush();

    let total_steps = n_epochs * dataset.len();
    let train_cfg = AdvancedConfig {
        lr: if config.d_model >= 512 { 0.001 } else if config.d_model >= 256 { 0.003 } else { 0.01 },
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
        grad_clip: 1.0,
        warmup_steps: if config.d_model >= 512 { 500 } else if config.d_model >= 256 { 200 } else { 100 },
        total_steps,
        label_smoothing: 0.05,
    };
    let mut trainer = LMTrainer::new(model, train_cfg);
    let mut rng = rand::thread_rng();

    for epoch in 0..n_epochs {
        let epoch_start = Instant::now();
        dataset.shuffle(&mut rng);
        let seqs = dataset.sequences();
        let n = seqs.len();

        let mut epoch_loss = 0.0f32;
        let mut epoch_acc = 0.0f32;
        let mut count = 0;

        for (i, seq) in seqs.iter().enumerate() {
            if seq.len() < 3 { continue; }
            let m = trainer.train_step(model, seq);
            epoch_loss += m.loss;
            epoch_acc += m.accuracy;
            count += 1;

            if (i + 1) % 500 == 0 || i + 1 == n {
                let avg_loss = epoch_loss / count as f32;
                let avg_acc = epoch_acc / count as f32 * 100.0;
                let elapsed = epoch_start.elapsed().as_secs_f32();
                let sps = (i + 1) as f32 / elapsed;
                print!(
                    "\r  E{}/{} [{:5}/{:5}] loss={:.3} ppl={:.1} acc={:.1}% lr={:.5} ({:.1} sps)",
                    epoch+1, n_epochs, i+1, n, avg_loss, avg_loss.exp(), avg_acc, m.lr, sps,
                );
                flush();
            }
        }

        let avg_loss = epoch_loss / count.max(1) as f32;
        let avg_acc = epoch_acc / count.max(1) as f32 * 100.0;
        let t = epoch_start.elapsed().as_secs_f32();
        println!(
            "\n  ✓ Epoch {} — loss={:.4} ppl={:.2} acc={:.1}% ({:.1}s)",
            epoch+1, avg_loss, avg_loss.exp(), avg_acc, t,
        );

        if (epoch + 1) % 5 == 0 || epoch + 1 == n_epochs {
            generate_samples(model, tok);
        }
        flush();

        let ckpt = format!("data/vi_ckpt_e{}.bin", epoch + 1);
        match checkpoint::save_model(model, &ckpt) {
            Ok(sz) => println!("  Saved {} ({:.1} KB)", ckpt, sz as f64 / 1024.0),
            Err(e) => eprintln!("  Save error: {e}"),
        }
        flush();
    }
}

fn generate_samples(model: &VagiLM, tok: &ByteTokenizer) {
    println!("  Samples:");
    for prompt in ["thầy giảng bài", "sinh viên", "trường"] {
        let tokens = tok.encode(prompt);
        let gen = model.generate_fast(&tokens, 50, 0.7);
        let raw = tok.decode(&gen);
        let text: String = raw.chars()
            .filter(|c| !c.is_control() && *c != '\u{FFFD}')
            .take(60)
            .collect();
        println!("    \"{prompt}\" → {text}");
    }
}

/// FAST training: f32 matmul + batch-parallel + AdamW.
///
/// No ternary pack/unpack overhead. Multiple sequences processed
/// in parallel via rayon. Single weight update per batch.
fn train_fast(
    model: &mut VagiLM,
    sentences: &[&str],
    n_epochs: usize,
    seq_len: usize,
    batch_size: usize,
    tok: &ByteTokenizer,
    config: &LMConfig,
) {
    let mut dataset = TextDataset::from_samples(sentences, seq_len);
    let n_seqs = dataset.len();
    let batches_per_epoch = (n_seqs + batch_size - 1) / batch_size;
    let total_steps = n_epochs * batches_per_epoch;

    println!("  {} sequences, {} batches/epoch, {} total steps",
        n_seqs, batches_per_epoch, total_steps);
    flush();

    let lr_base = if config.d_model >= 512 { 0.001 }
        else if config.d_model >= 256 { 0.003 }
        else { 0.01 };

    let mut adam_m: Vec<f32> = Vec::new();
    let mut adam_v: Vec<f32> = Vec::new();
    let mut rng = rand::thread_rng();
    let mut global_step = 0usize;

    for epoch in 0..n_epochs {
        let epoch_start = Instant::now();
        dataset.shuffle(&mut rng);
        let seqs = dataset.sequences();

        let mut epoch_loss = 0.0f32;
        let mut epoch_acc = 0.0f32;
        let mut batch_count = 0;

        for batch_start in (0..seqs.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(seqs.len());
            let batch: Vec<&[u32]> = seqs[batch_start..batch_end].iter()
                .filter(|s| s.len() >= 3)
                .map(|s| s.as_slice())
                .collect();

            if batch.is_empty() { continue; }

            // Cosine LR schedule
            let progress = global_step as f32 / total_steps.max(1) as f32;
            let warmup_steps = (total_steps / 20).max(50);
            let lr = if global_step < warmup_steps {
                lr_base * (global_step + 1) as f32 / warmup_steps as f32
            } else {
                let min_lr = lr_base * 0.1;
                min_lr + 0.5 * (lr_base - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
            };

            let (loss, acc) = fast_train::batch_train_step(
                model, &batch, &mut adam_m, &mut adam_v, global_step, lr,
            );

            epoch_loss += loss;
            epoch_acc += acc;
            batch_count += 1;
            global_step += 1;

            if batch_count % 50 == 0 || batch_start + batch_size >= seqs.len() {
                let avg_loss = epoch_loss / batch_count as f32;
                let avg_acc = epoch_acc / batch_count as f32 * 100.0;
                let elapsed = epoch_start.elapsed().as_secs_f32();
                let bps = batch_count as f32 / elapsed;
                let sps = (batch_count * batch_size) as f32 / elapsed;
                print!(
                    "\r  E{}/{} [{:4}/{:4}] loss={:.3} ppl={:.1} acc={:.1}% lr={:.5} ({:.1} batch/s, {:.0} sps)",
                    epoch+1, n_epochs, batch_count, batches_per_epoch,
                    avg_loss, avg_loss.exp(), avg_acc, lr, bps, sps,
                );
                flush();
            }
        }

        let avg_loss = epoch_loss / batch_count.max(1) as f32;
        let avg_acc = epoch_acc / batch_count.max(1) as f32 * 100.0;
        let t = epoch_start.elapsed().as_secs_f32();
        let sps = (batch_count * batch_size) as f32 / t;
        println!(
            "\n  ✓ Epoch {} — loss={:.4} ppl={:.2} acc={:.1}% ({:.1}s, {:.0} sps)",
            epoch+1, avg_loss, avg_loss.exp(), avg_acc, t, sps,
        );

        if (epoch + 1) % 5 == 0 || epoch + 1 == n_epochs {
            generate_samples(model, tok);
        }

        let ckpt = format!("data/vi_fast_e{}.bin", epoch + 1);
        match checkpoint::save_model(model, &ckpt) {
            Ok(sz) => println!("  Saved {} ({:.1} KB)", ckpt, sz as f64 / 1024.0),
            Err(e) => eprintln!("  Save error: {e}"),
        }
        flush();
    }
}

fn flush() { std::io::stdout().flush().ok(); }
