//! Evaluate trained Vietnamese model quality.
//!
//! Loads model checkpoint, splits data into train/test,
//! compares trained vs random model.
//!
//! Usage:
//!   cargo run --example eval_vietnamese -p vagi-lm --release

use std::io::Write;
use vagi_lm::{VagiLM, TextDataset};
use vagi_lm::tokenizer::ByteTokenizer;
use vagi_lm::checkpoint;
use vagi_lm::eval;

const DATA_PATH: &str = "data/vi_sentences.txt";
const MODEL_PATH: &str = "data/vi_model.bin";

fn main() {
    println!("═══════════════════════════════════════════");
    println!("  vAGI-2 Vietnamese Model Evaluation");
    println!("═══════════════════════════════════════════");
    std::io::stdout().flush().ok();

    // Load data
    let text = match std::fs::read_to_string(DATA_PATH) {
        Ok(t) => t,
        Err(e) => { eprintln!("ERROR loading data: {e}"); return; }
    };
    let sentences: Vec<&str> = text.lines().map(|l| l.trim()).filter(|l| l.len() >= 5).collect();
    let n = sentences.len();
    let split = (n * 9) / 10; // 90/10 train/test split
    let test_sentences = &sentences[split..];
    println!("Data: {} total, {} test", n, test_sentences.len());

    let test_ds = TextDataset::from_samples(test_sentences, 64);
    println!("Test sequences: {}", test_ds.len());

    // Load trained model
    print!("Loading trained model... ");
    std::io::stdout().flush().ok();
    let trained = match checkpoint::load_model(MODEL_PATH) {
        Ok(m) => {
            println!("OK (d={}, {}K params)", m.config.d_model, m.config.param_count() / 1000);
            m
        }
        Err(e) => {
            println!("FAIL: {e}");
            println!("Run training first: cargo run --example train_vietnamese -p vagi-lm --release");
            return;
        }
    };

    // Random baseline
    let random = VagiLM::new(trained.config.clone());

    // Evaluate both
    println!("\n── Trained Model ──");
    let trained_report = eval::evaluate(&trained, &test_ds);
    trained_report.print();

    println!("\n── Random Baseline ──");
    let random_report = eval::evaluate(&random, &test_ds);
    random_report.print();

    // Comparison
    println!("\n── Improvement ──");
    let ppl_ratio = random_report.perplexity / trained_report.perplexity.max(0.001);
    let acc_diff = trained_report.accuracy - random_report.accuracy;
    println!("  Perplexity:  {:.1}x better", ppl_ratio);
    println!("  Accuracy:    +{:.1}%", acc_diff * 100.0);

    // Diversity
    let tok = ByteTokenizer::new();
    let prompts = vec!["thầy giảng", "sinh viên", "trường", "học"];
    let trained_div = eval::diversity_score(&trained, &prompts, &tok);
    let random_div = eval::diversity_score(&random, &prompts, &tok);
    println!("  Diversity:   trained={:.2} random={:.2}", trained_div, random_div);

    // Sample generations
    println!("\n── Sample Generations ──");
    for prompt in &prompts {
        let tokens = tok.encode(prompt);
        let gen = trained.generate_fast(&tokens, 40, 0.7);
        let all: Vec<u32> = tokens.iter().chain(gen.iter()).cloned().collect();
        let text: String = tok.decode(&all).chars().take(80).collect();
        println!("  \"{prompt}\" → {text}");
    }
}
