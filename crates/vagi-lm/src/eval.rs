//! Evaluation utilities for language model quality.
//!
//! Provides perplexity, accuracy, and diversity metrics.

use crate::model::VagiLM;
use crate::dataset::TextDataset;

/// Evaluation results.
pub struct EvalReport {
    /// Average cross-entropy loss.
    pub loss: f32,
    /// Perplexity (exp(loss)).
    pub perplexity: f32,
    /// Top-1 token prediction accuracy.
    pub accuracy: f32,
    /// Unique tokens generated in sample set.
    pub unique_tokens: usize,
    /// Number of evaluation samples.
    pub n_samples: usize,
}

impl EvalReport {
    pub fn print(&self) {
        println!("  Loss:          {:.4}", self.loss);
        println!("  Perplexity:    {:.2}", self.perplexity);
        println!("  Accuracy:      {:.1}%", self.accuracy * 100.0);
        println!("  Unique tokens: {}", self.unique_tokens);
        println!("  Samples:       {}", self.n_samples);
    }
}

/// Evaluate model on a dataset.
pub fn evaluate(model: &VagiLM, dataset: &TextDataset) -> EvalReport {
    let seqs = dataset.sequences();
    let v = model.config.vocab_size;
    let mut total_loss = 0.0f32;
    let mut total_acc = 0.0f32;
    let mut count = 0usize;
    let mut token_set = std::collections::HashSet::new();

    for seq in seqs {
        if seq.len() < 3 { continue; }
        let seq_len = seq.len() - 1;
        let logits = model.forward(&seq[..seq_len]);
        let targets = &seq[1..];

        let mut seq_loss = 0.0f32;
        let mut seq_correct = 0usize;

        for t in 0..seq_len {
            let tok_logits = &logits[t * v..(t + 1) * v];
            let target = targets[t] as usize;

            // Softmax + loss
            let max_l = tok_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = tok_logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let prob = exps[target] / sum;
            seq_loss += -prob.max(1e-10).ln();

            // Accuracy
            let pred = tok_logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            if pred == target { seq_correct += 1; }

            token_set.insert(target);
        }

        total_loss += seq_loss / seq_len as f32;
        total_acc += seq_correct as f32 / seq_len as f32;
        count += 1;
    }

    let avg_loss = total_loss / count.max(1) as f32;
    EvalReport {
        loss: avg_loss,
        perplexity: avg_loss.exp(),
        accuracy: total_acc / count.max(1) as f32,
        unique_tokens: token_set.len(),
        n_samples: count,
    }
}

/// Measure generation diversity: run prompts through model and count unique n-grams.
pub fn diversity_score(model: &VagiLM, prompts: &[&str], tok: &crate::tokenizer::ByteTokenizer) -> f32 {
    use std::collections::HashSet;
    let mut all_bigrams = HashSet::new();
    let mut total_bigrams = 0usize;

    for prompt in prompts {
        let tokens = tok.encode(prompt);
        let generated = model.generate_fast(&tokens, 50, 0.8);
        for w in generated.windows(2) {
            all_bigrams.insert((w[0], w[1]));
            total_bigrams += 1;
        }
    }

    if total_bigrams == 0 { return 0.0; }
    all_bigrams.len() as f32 / total_bigrams as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LMConfig, VagiLM, TextDataset};

    #[test]
    fn test_evaluate() {
        let model = VagiLM::new(LMConfig::tiny());
        let samples = vec!["Hello world", "test input"];
        let ds = TextDataset::from_samples(&samples, 16);
        let report = evaluate(&model, &ds);
        assert!(report.perplexity > 0.0);
        assert!(report.n_samples > 0);
    }
}
