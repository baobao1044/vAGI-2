//! Fine-tune vAGI-2 from f32 pre-trained checkpoint using STE training.
//!
//! Loads weights from `data/vi_fast_e17.bin` (best checkpoint) and runs
//! exact-gradient STE training to fix generation quality.
//!
//! Usage:
//!   cargo run --example finetune_vietnamese -p vagi-lm --release
//!   cargo run --example finetune_vietnamese -p vagi-lm --release -- --epochs 10

use std::io::Write;
use std::time::Instant;
use vagi_lm::{VagiLM, LMTrainer, AdvancedConfig, TextDataset};
use vagi_lm::tokenizer::ByteTokenizer;
use vagi_lm::checkpoint;

const DATA_PATH: &str = "data/vi_sentences.txt";
const CKPT_PATH: &str = "data/vi_fast_e17.bin";
const OUT_PATH: &str = "data/vi_model.bin";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_epochs: usize = args.windows(2)
        .find(|w| w[0] == "--epochs")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(5);
    let seq_len: usize = args.windows(2)
        .find(|w| w[0] == "--seq")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(32);

    let start = Instant::now();
    println!("═══════════════════════════════════════════");
    println!("  vAGI-2 STE Fine-Tune (Exact Gradients)");
    println!("═══════════════════════════════════════════");
    flush();

    // 1. Load checkpoint
    print!("[1/5] Loading checkpoint: {CKPT_PATH}... ");
    flush();
    let mut model = match checkpoint::load_model(CKPT_PATH) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: {e}");
            eprintln!("Run f32 pre-training first:");
            eprintln!("  cargo run --example train_vietnamese -p vagi-lm --release -- --small --fast --epochs 20");
            std::process::exit(1);
        }
    };
    println!("OK (d={}, L={}, ~{}K params)",
        model.config.d_model, model.config.n_layers,
        model.config.param_count() / 1000);
    flush();

    // 2. Load data
    print!("[2/5] Loading data... ");
    flush();
    let text = match std::fs::read_to_string(DATA_PATH) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        }
    };
    let sentences: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| l.len() >= 5).collect();
    println!("{} sentences", sentences.len());
    flush();

    // 3. Test generation before fine-tune
    let tok = ByteTokenizer::new();
    println!("[3/5] Before fine-tune:");
    generate_samples(&model, &tok);
    flush();

    // 4. STE fine-tune
    println!("[4/5] STE Fine-Tuning (seq_len={seq_len}, {n_epochs} epochs)");
    println!("       Exact gradients — slow but accurate attention training");
    flush();

    let mut dataset = TextDataset::from_samples(&sentences, seq_len);
    let n_seqs = dataset.len();
    let total_steps = n_epochs * n_seqs;

    println!("  {} sequences, {} total steps", n_seqs, total_steps);
    flush();

    let train_cfg = AdvancedConfig {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
        grad_clip: 1.0,
        warmup_steps: (total_steps / 20).max(50),
        total_steps,
        label_smoothing: 0.05,
    };
    let mut trainer = LMTrainer::new(&model, train_cfg);
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
            let m = trainer.train_step(&mut model, seq);
            epoch_loss += m.loss;
            epoch_acc += m.accuracy;
            count += 1;

            if (i + 1) % 200 == 0 || i + 1 == n {
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
        let sps = count as f32 / t;
        println!(
            "\n  ✓ Epoch {} — loss={:.4} ppl={:.2} acc={:.1}% ({:.1}s, {:.1} sps)",
            epoch+1, avg_loss, avg_loss.exp(), avg_acc, t, sps,
        );

        generate_samples(&model, &tok);
        flush();

        // Save checkpoint every epoch
        let ckpt = format!("data/vi_ste_e{}.bin", epoch + 1);
        match checkpoint::save_model(&model, &ckpt) {
            Ok(sz) => println!("  Saved {} ({:.1} KB)", ckpt, sz as f64 / 1024.0),
            Err(e) => eprintln!("  Save error: {e}"),
        }
        flush();
    }

    // 5. Save final
    println!("[5/5] Saving final model...");
    match checkpoint::save_model(&model, OUT_PATH) {
        Ok(sz) => println!("  {OUT_PATH} ({:.1} KB)", sz as f64 / 1024.0),
        Err(e) => eprintln!("  Save error: {e}"),
    }

    println!("\n[After fine-tune]");
    generate_samples(&model, &tok);

    let total = start.elapsed().as_secs_f32();
    println!("\n═══ Done in {:.0}s ({:.1} min) ═══", total, total / 60.0);
    println!("\nTo chat: cargo run --example chat_vi -p vagi-lm --release");
}

fn generate_samples(model: &VagiLM, tok: &ByteTokenizer) {
    println!("  Samples:");
    for prompt in ["thầy giảng bài", "sinh viên", "trường đại học", "tốt nghiệp"] {
        let tokens = tok.encode(prompt);
        let gen = model.generate_fast(&tokens, 60, 0.8);
        let raw = tok.decode(&gen);
        let text: String = raw.chars()
            .filter(|c| !c.is_control() && *c != '\u{FFFD}')
            .take(80)
            .collect();
        println!("    \"{prompt}\" → {text}");
    }
}

fn flush() { std::io::stdout().flush().ok(); }
