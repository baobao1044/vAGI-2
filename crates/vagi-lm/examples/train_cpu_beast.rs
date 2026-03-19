//! CPU Beast Training — All three pillars combined.
//!
//! Pillar 1: BPE Tokenizer (3-5x token efficiency)
//! Pillar 2: f32 batch-parallel training (no ternary overhead)
//! Pillar 3: AdamW optimizer + exact attention gradients
//!
//! Target: beat NVIDIA T4 GPU for small ternary models.
//!
//! Usage:
//!   cargo run --example train_cpu_beast -p vagi-lm --release
//!   cargo run --example train_cpu_beast -p vagi-lm --release -- --small
//!   cargo run --example train_cpu_beast -p vagi-lm --release -- --small --epochs 30

use std::io::Write;
use std::time::Instant;
use vagi_lm::{VagiLM, LMConfig, batch_train_step};
use vagi_lm::tokenizer_bpe::BPETokenizer;
use vagi_lm::checkpoint;

const DATA_PATH: &str = "data/vi_sentences.txt";
const BPE_PATH: &str = "data/bpe_merges.txt";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_base = args.iter().any(|a| a == "--base");
    let use_small = args.iter().any(|a| a == "--small") && !use_base;
    let n_epochs: usize = args.windows(2)
        .find(|w| w[0] == "--epochs")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(20);
    let batch_size: usize = args.windows(2)
        .find(|w| w[0] == "--batch")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(16);
    let n_merges: usize = args.windows(2)
        .find(|w| w[0] == "--merges")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(1800);

    let start = Instant::now();
    println!("╔══════════════════════════════════════════╗");
    println!("║   vAGI-2 CPU Beast Training              ║");
    println!("║   BPE + f32 Batch-Parallel + AdamW       ║");
    println!("╚══════════════════════════════════════════╝");
    flush();

    // ── 1. Load data ──
    print!("[1/6] Loading data... ");
    flush();
    let text = std::fs::read_to_string(DATA_PATH).expect("Cannot read data file");
    let sentences: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| l.len() >= 5).collect();
    println!("{} sentences ({:.0} KB)", sentences.len(), text.len() as f64 / 1024.0);
    flush();

    // ── 2. Train or load BPE tokenizer ──
    let bpe = if let Ok(tok) = BPETokenizer::load(BPE_PATH) {
        println!("[2/6] BPE: loaded from {} (vocab={})", BPE_PATH, tok.vocab_size());
        tok
    } else {
        print!("[2/6] BPE: training {} merges... ", n_merges);
        flush();
        let t0 = Instant::now();
        let corpus = sentences.join(" ");
        let tok = BPETokenizer::train(&corpus, n_merges);
        tok.save(BPE_PATH).expect("Cannot save BPE");
        println!("done in {:.1}s (vocab={})", t0.elapsed().as_secs_f32(), tok.vocab_size());
        tok
    };

    // Show compression stats
    let test_text = "thầy giảng bài hay, sinh viên đi học đầy đủ";
    let ratio = bpe.compression_ratio(test_text);
    let tokens = bpe.encode(test_text);
    println!("  Compression: {:.1} bytes/token", ratio);
    println!("   \"{test_text}\"");
    println!("    → {} tokens (vs {} bytes)", tokens.len(), test_text.len());
    flush();

    // ── 3. Create model with BPE vocab ──
    let vocab_size = bpe.vocab_size();
    let config = if use_base {
        LMConfig { vocab_size, ..LMConfig::base() }
    } else if use_small {
        LMConfig { vocab_size, ..LMConfig::small() }
    } else {
        LMConfig { vocab_size, ..LMConfig::tiny() }
    };
    let config_name = if use_base { "BASE" } else if use_small { "SMALL" } else { "TINY" };
    let mut model = VagiLM::new(config.clone());
    println!("[3/6] Model: d={} L={} H={} vocab={} (~{}K params) [{}]",
        config.d_model, config.n_layers, config.n_heads, vocab_size,
        config.param_count() / 1000, config_name);
    println!("  AVX2: {}", if vagi_core::has_avx2() { "YES ✓" } else { "NO" });
    flush();

    // ── 4. Tokenize all sequences ──
    print!("[4/6] Tokenizing with BPE... ");
    flush();
    let t0 = Instant::now();
    let seq_len = if use_base { 64 } else if use_small { 48 } else { 32 };
    let all_tokens: Vec<Vec<u32>> = sentences.iter()
        .map(|s| bpe.encode(s))
        .filter(|t| t.len() >= 4)
        .collect();

    // Create fixed-length sequences
    let mut sequences: Vec<Vec<u32>> = Vec::new();
    for tokens in &all_tokens {
        if tokens.len() <= seq_len + 1 {
            sequences.push(tokens.clone());
        } else {
            // Chunk into overlapping windows
            for start in (0..tokens.len().saturating_sub(seq_len)).step_by(seq_len / 2) {
                let end = (start + seq_len + 1).min(tokens.len());
                if end - start >= 4 {
                    sequences.push(tokens[start..end].to_vec());
                }
            }
        }
    }
    println!("{} sequences (seq_len={}) in {:.1}s", sequences.len(), seq_len, t0.elapsed().as_secs_f32());
    flush();

    // ── 5. Train! ──
    let batches_per_epoch = (sequences.len() + batch_size - 1) / batch_size;
    let total_steps = n_epochs * batches_per_epoch;
    println!("[5/6] Training: {} epochs, batch={}, {} batches/epoch, {} total steps",
        n_epochs, batch_size, batches_per_epoch, total_steps);
    println!("  Optimizer: AdamW (exact attention gradients)");
    println!("  Forward: f32 matmul (no ternary overhead)");
    println!("  Parallel: rayon batch-parallel");
    flush();

    let lr_base = if config.d_model >= 512 { 0.0003 }
        else if config.d_model >= 256 { 0.001 }
        else { 0.003 };

    let mut rng_state = 42u64;
    let mut best_loss = f32::MAX;
    let mut adam_m: Vec<f32> = Vec::new();
    let mut adam_v: Vec<f32> = Vec::new();
    let mut global_step = 0usize;

    for epoch in 0..n_epochs {
        let epoch_start = Instant::now();

        // Shuffle sequences
        fisher_yates_shuffle(&mut sequences, &mut rng_state);

        let mut epoch_loss = 0.0f32;
        let mut epoch_acc = 0.0f32;
        let mut batch_count = 0;

        for batch_start in (0..sequences.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(sequences.len());
            let batch: Vec<&[u32]> = sequences[batch_start..batch_end].iter()
                .map(|s| s.as_slice())
                .collect();
            if batch.is_empty() { continue; }

            // Cosine LR schedule
            let progress = global_step as f32 / total_steps.max(1) as f32;
            let warmup_steps = (total_steps / 20).max(100);
            let lr = if global_step < warmup_steps {
                lr_base * (global_step + 1) as f32 / warmup_steps as f32
            } else {
                let min_lr = lr_base * 0.05;
                min_lr + 0.5 * (lr_base - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
            };

            let (loss, acc) = batch_train_step(
                &mut model, &batch, &mut adam_m, &mut adam_v, global_step, lr,
            );
            global_step += 1;

            epoch_loss += loss;
            epoch_acc += acc;
            batch_count += 1;

            if batch_count % 100 == 0 || batch_start + batch_size >= sequences.len() {
                let avg_loss = epoch_loss / batch_count as f32;
                let avg_acc = epoch_acc / batch_count as f32 * 100.0;
                let elapsed = epoch_start.elapsed().as_secs_f32();
                let bps = batch_count as f32 / elapsed;
                let sps = (batch_count * batch_size) as f32 / elapsed;
                print!(
                    "\r  E{}/{} [{:5}/{:5}] loss={:.3} ppl={:.1} acc={:.1}% lr={:.5} ({:.1}b/s {:.0}sps)  ",
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
            "\n  ✓ E{} — loss={:.4} ppl={:.2} acc={:.1}% ({:.1}s, {:.0} sps)",
            epoch+1, avg_loss, avg_loss.exp(), avg_acc, t, sps,
        );

        // Generate samples every 5 epochs
        if (epoch + 1) % 5 == 0 || epoch + 1 == n_epochs || epoch == 0 {
            generate_samples(&model, &bpe);
        }

        // Save best model
        if avg_loss < best_loss {
            best_loss = avg_loss;
            let ckpt = format!("data/vi_beast_best.bin");
            match checkpoint::save_model(&model, &ckpt) {
                Ok(sz) => println!("  ★ NEW BEST → {} ({:.1} KB)", ckpt, sz as f64 / 1024.0),
                Err(e) => eprintln!("  Save error: {e}"),
            }
        }

        // Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 {
            let ckpt = format!("data/vi_beast_e{}.bin", epoch + 1);
            match checkpoint::save_model(&model, &ckpt) {
                Ok(sz) => println!("  Saved {} ({:.1} KB)", ckpt, sz as f64 / 1024.0),
                Err(e) => eprintln!("  Save error: {e}"),
            }
        }
        flush();
    }

    // ── 6. Save final ──
    println!("[6/6] Saving final model...");
    let out_path = "data/vi_beast_final.bin";
    match checkpoint::save_model(&model, out_path) {
        Ok(sz) => println!("  {} ({:.1} KB)", out_path, sz as f64 / 1024.0),
        Err(e) => eprintln!("  Save error: {e}"),
    }

    let total = start.elapsed().as_secs_f32();
    println!("\n═══ Done in {:.0}s ({:.1} min) ═══", total, total / 60.0);
    println!("\nBest loss: {:.4}", best_loss);
}

fn generate_samples(model: &VagiLM, bpe: &BPETokenizer) {
    println!("  Samples:");
    for prompt in ["thầy giảng bài", "sinh viên", "trường đại học", "tốt nghiệp"] {
        let tokens = bpe.encode(prompt);
        let gen = model.generate_fast(&tokens, 40, 0.8);
        let text = bpe.decode(&gen);
        let clean: String = text.chars()
            .filter(|c| !c.is_control() && *c != '\u{FFFD}')
            .take(80)
            .collect();
        println!("    \"{prompt}\" → {clean}");
    }
}

/// Simple Fisher-Yates shuffle (no rand dependency).
fn fisher_yates_shuffle<T>(data: &mut [T], state: &mut u64) {
    for i in (1..data.len()).rev() {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (*state >> 33) as usize % (i + 1);
        data.swap(i, j);
    }
}

fn flush() { std::io::stdout().flush().ok(); }
