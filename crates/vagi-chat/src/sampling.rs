//! Advanced sampling strategies for text generation.
//!
//! Provides top-k, top-p (nucleus), and repetition penalty.

use rand::Rng;

/// Top-K sampling: keep only the K tokens with highest logits, sample from them.
pub fn top_k_sample(logits: &[f32], k: usize, temperature: f32, rng: &mut impl Rng) -> u32 {
    let k = k.min(logits.len()).max(1);

    // Find top-K indices
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);

    // Temperature scaling + softmax over top-K
    if temperature < 1e-6 {
        return indexed[0].0 as u32; // greedy
    }

    let max_val = indexed[0].1;
    let exps: Vec<f32> = indexed.iter()
        .map(|(_, v)| ((v - max_val) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();

    // Sample
    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for (i, &e) in exps.iter().enumerate() {
        cumulative += e / sum;
        if r < cumulative {
            return indexed[i].0 as u32;
        }
    }
    indexed.last().unwrap().0 as u32
}

/// Top-P (nucleus) sampling: sample from the smallest set of tokens whose
/// cumulative probability exceeds P.
pub fn top_p_sample(logits: &[f32], p: f32, temperature: f32, rng: &mut impl Rng) -> u32 {
    if temperature < 1e-6 {
        return argmax(logits);
    }

    // Temperature scaling + softmax
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| ((l - max_val) / temperature).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Sort by probability (descending)
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find the nucleus (smallest set with cumulative prob >= p)
    let mut cumulative_prob = 0.0;
    let mut nucleus_size = 0;
    for (_, prob) in &indexed {
        cumulative_prob += prob;
        nucleus_size += 1;
        if cumulative_prob >= p {
            break;
        }
    }
    nucleus_size = nucleus_size.max(1);

    // Renormalize and sample from nucleus
    let nucleus = &indexed[..nucleus_size];
    let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();

    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for &(idx, prob) in nucleus {
        cum += prob / nucleus_sum;
        if r < cum {
            return idx as u32;
        }
    }
    nucleus.last().unwrap().0 as u32
}

/// Apply repetition penalty to logits.
///
/// Tokens that appear in `history` get their logits divided by `penalty` (if positive)
/// or multiplied by `penalty` (if negative).
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 { return; }
    for &token in history {
        let idx = token as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

fn argmax(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_top_k_greedy() {
        let logits = vec![0.1, 0.5, 0.2, 0.9, 0.3];
        let mut rng = StdRng::seed_from_u64(42);
        let token = top_k_sample(&logits, 1, 0.0, &mut rng);
        assert_eq!(token, 3, "Top-1 should select argmax");
    }

    #[test]
    fn test_top_k_restricts_vocab() {
        let mut logits = vec![0.0f32; 100];
        logits[50] = 10.0;
        logits[51] = 9.0;
        logits[52] = 8.0;
        let mut rng = StdRng::seed_from_u64(42);

        let mut selected = std::collections::HashSet::new();
        for _ in 0..100 {
            let tok = top_k_sample(&logits, 3, 0.5, &mut rng);
            selected.insert(tok);
        }
        assert!(selected.len() <= 3, "Top-3 should only select from 3 tokens, got {:?}", selected);
    }

    #[test]
    fn test_top_p_greedy() {
        let logits = vec![0.1, 0.9, 0.2];
        let mut rng = StdRng::seed_from_u64(42);
        let token = top_p_sample(&logits, 0.9, 0.0, &mut rng);
        assert_eq!(token, 1, "Greedy should select argmax");
    }

    #[test]
    fn test_top_p_nucleus() {
        let mut logits = vec![0.0f32; 10];
        logits[0] = 10.0; // dominant token
        let mut rng = StdRng::seed_from_u64(42);

        // With very low p, should mostly select the dominant token
        let mut count_0 = 0;
        for _ in 0..50 {
            let tok = top_p_sample(&logits, 0.5, 0.3, &mut rng);
            if tok == 0 { count_0 += 1; }
        }
        assert!(count_0 > 40, "Dominant token should be selected most of the time, got {count_0}/50");
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let original = logits.clone();
        apply_repetition_penalty(&mut logits, &[2, 4], 2.0);

        // Tokens 2 and 4 should be penalized
        assert!((logits[0] - original[0]).abs() < 1e-6, "Unpenalized token changed");
        assert!(logits[2] < original[2], "Token 2 should be penalized");
        assert!(logits[4] < original[4], "Token 4 should be penalized");
    }
}
