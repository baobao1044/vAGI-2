//! Streaming state machine — multi-scale running state for infinite context.
//!
//! Maintains 5 levels of state at different temporal scales:
//! L0: word     — update every token
//! L1: sentence — update every ~10 tokens
//! L2: paragraph — update every ~50 tokens
//! L3: topic    — update every ~200 tokens
//! L4: episode  — update every ~1000 tokens
//!
//! Key property: O(1) compute per token, constant memory regardless of sequence length.

/// Configuration for a single state level.
#[derive(Clone, Debug)]
pub struct LevelConfig {
    /// Update every N tokens.
    pub update_interval: usize,
    /// EMA decay factor (0..1). Higher = more weight on new data.
    pub ema_alpha: f32,
    /// Human-readable label.
    pub label: &'static str,
}

/// A single level in the streaming state hierarchy.
#[derive(Clone, Debug)]
pub struct StateLevel {
    /// Current running state [d_model].
    state: Vec<f32>,
    /// Accumulation buffer for tokens since last update.
    buffer: Vec<Vec<f32>>,
    /// Configuration.
    config: LevelConfig,
    /// Tokens since last state update.
    tokens_since_update: usize,
}

impl StateLevel {
    fn new(d_model: usize, config: LevelConfig) -> Self {
        Self {
            state: vec![0.0; d_model],
            buffer: Vec::new(),
            config,
            tokens_since_update: 0,
        }
    }

    /// Ingest a token embedding, possibly triggering a state update.
    /// Returns true if state was updated.
    fn ingest(&mut self, token: &[f32]) -> bool {
        self.buffer.push(token.to_vec());
        self.tokens_since_update += 1;

        if self.tokens_since_update >= self.config.update_interval {
            self.update_state();
            true
        } else {
            false
        }
    }

    /// Compress buffer via mean and EMA-update the state.
    fn update_state(&mut self) {
        if self.buffer.is_empty() { return; }

        let d = self.state.len();
        let n = self.buffer.len() as f32;
        let alpha = self.config.ema_alpha;

        // Compress buffer: simple mean
        let mut compressed = vec![0.0f32; d];
        for tok in &self.buffer {
            for (c, t) in compressed.iter_mut().zip(tok.iter()) {
                *c += t;
            }
        }
        for c in compressed.iter_mut() {
            *c /= n;
        }

        // EMA update: state = (1 - alpha) * state + alpha * compressed
        for (s, c) in self.state.iter_mut().zip(compressed.iter()) {
            *s = (1.0 - alpha) * *s + alpha * c;
        }

        self.buffer.clear();
        self.tokens_since_update = 0;
    }

    /// Get current state (read-only).
    fn state(&self) -> &[f32] {
        &self.state
    }
}

/// Default 5-level configuration.
fn default_levels() -> Vec<LevelConfig> {
    vec![
        LevelConfig { update_interval: 1,    ema_alpha: 0.3,  label: "word" },
        LevelConfig { update_interval: 10,   ema_alpha: 0.2,  label: "sentence" },
        LevelConfig { update_interval: 50,   ema_alpha: 0.15, label: "paragraph" },
        LevelConfig { update_interval: 200,  ema_alpha: 0.1,  label: "topic" },
        LevelConfig { update_interval: 1000, ema_alpha: 0.05, label: "episode" },
    ]
}

/// Multi-scale streaming state machine.
///
/// O(1) compute per token, constant memory regardless of sequence length.
/// Each level captures a different temporal scale via EMA.
#[derive(Clone, Debug)]
pub struct StreamingState {
    /// State levels ordered from finest (word) to coarsest (episode).
    levels: Vec<StateLevel>,
    /// Total tokens ingested.
    pub total_tokens: usize,
    /// Model dimension.
    pub d_model: usize,
}

impl StreamingState {
    /// Create with default 5-level configuration.
    pub fn new(d_model: usize) -> Self {
        let levels = default_levels().into_iter()
            .map(|cfg| StateLevel::new(d_model, cfg))
            .collect();
        Self { levels, total_tokens: 0, d_model }
    }

    /// Create with custom level configurations.
    pub fn with_levels(d_model: usize, configs: Vec<LevelConfig>) -> Self {
        let levels = configs.into_iter()
            .map(|cfg| StateLevel::new(d_model, cfg))
            .collect();
        Self { levels, total_tokens: 0, d_model }
    }

    /// Ingest a single token embedding.
    pub fn update(&mut self, token: &[f32]) {
        debug_assert_eq!(token.len(), self.d_model);
        for level in &mut self.levels {
            level.ingest(token);
        }
        self.total_tokens += 1;
    }

    /// Ingest a batch of token embeddings.
    pub fn update_batch(&mut self, tokens: &[Vec<f32>]) {
        for tok in tokens {
            self.update(tok);
        }
    }

    /// Get the state at a specific level index.
    pub fn level_state(&self, level: usize) -> Option<&[f32]> {
        self.levels.get(level).map(|l| l.state())
    }

    /// Get all level states concatenated: [L0 | L1 | L2 | L3 | L4].
    /// Total length = 5 * d_model.
    pub fn concat_states(&self) -> Vec<f32> {
        self.levels.iter()
            .flat_map(|l| l.state().iter().copied())
            .collect()
    }

    /// Number of levels.
    pub fn n_levels(&self) -> usize { self.levels.len() }

    /// Memory usage in bytes (state vectors only).
    pub fn memory_bytes(&self) -> usize {
        self.levels.iter()
            .map(|l| l.state.len() * 4 + l.buffer.len() * l.state.len() * 4)
            .sum()
    }

    /// Get level labels.
    pub fn level_labels(&self) -> Vec<&'static str> {
        self.levels.iter().map(|l| l.config.label).collect()
    }

    /// Reset all levels to zero state.
    pub fn reset(&mut self) {
        for level in &mut self.levels {
            level.state.fill(0.0);
            level.buffer.clear();
            level.tokens_since_update = 0;
        }
        self.total_tokens = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_default() {
        let ss = StreamingState::new(64);
        assert_eq!(ss.n_levels(), 5);
        assert_eq!(ss.total_tokens, 0);
        assert_eq!(ss.d_model, 64);
    }

    #[test]
    fn test_single_update() {
        let mut ss = StreamingState::new(4);
        let token = vec![1.0, 2.0, 3.0, 4.0];
        ss.update(&token);
        assert_eq!(ss.total_tokens, 1);
        // L0 (update_interval=1) should have updated
        let l0 = ss.level_state(0).unwrap();
        assert!(l0.iter().any(|v| *v != 0.0), "L0 should update after 1 token");
    }

    #[test]
    fn test_level_update_frequencies() {
        let mut ss = StreamingState::new(4);
        // Feed 10 tokens
        for i in 0..10 {
            ss.update(&vec![i as f32; 4]);
        }
        assert_eq!(ss.total_tokens, 10);
        // L0 (interval=1) should have updated 10 times
        // L1 (interval=10) should have updated exactly 1 time
        let l1 = ss.level_state(1).unwrap();
        assert!(l1.iter().any(|v| *v != 0.0), "L1 should update after 10 tokens");
    }

    #[test]
    fn test_constant_memory_100k() {
        let mut ss = StreamingState::new(64);
        let mem_before = ss.memory_bytes();
        // Feed 100K tokens
        for i in 0..100_000 {
            ss.update(&vec![(i as f32 * 0.001).sin(); 64]);
        }
        let mem_after = ss.memory_bytes();
        assert_eq!(ss.total_tokens, 100_000);
        // Memory should stay roughly constant (buffers may have partial data)
        // Max buffer: L4 has up to 999 entries × 64 × 4 = ~256KB
        assert!(mem_after < 300_000, "Memory should stay bounded, got {}", mem_after);
        // State should still be finite
        for level in 0..5 {
            let s = ss.level_state(level).unwrap();
            assert!(s.iter().all(|v| v.is_finite()), "Level {level} should be finite");
        }
    }

    #[test]
    fn test_state_changes_with_content() {
        let mut ss = StreamingState::new(8);
        // Phase 1: constant input
        for _ in 0..50 {
            ss.update(&vec![1.0; 8]);
        }
        let state_phase1: Vec<f32> = ss.level_state(0).unwrap().to_vec();

        // Phase 2: different input
        for _ in 0..50 {
            ss.update(&vec![-1.0; 8]);
        }
        let state_phase2: Vec<f32> = ss.level_state(0).unwrap().to_vec();

        assert_ne!(state_phase1, state_phase2, "State should change when content shifts");
    }

    #[test]
    fn test_concat_states() {
        let ss = StreamingState::new(4);
        let concat = ss.concat_states();
        assert_eq!(concat.len(), 5 * 4, "Concat should be 5 levels × d_model");
    }

    #[test]
    fn test_reset() {
        let mut ss = StreamingState::new(4);
        for _ in 0..100 {
            ss.update(&vec![1.0; 4]);
        }
        assert_eq!(ss.total_tokens, 100);
        ss.reset();
        assert_eq!(ss.total_tokens, 0);
        let l0 = ss.level_state(0).unwrap();
        assert!(l0.iter().all(|v| *v == 0.0), "Reset should zero all states");
    }

    #[test]
    fn test_level_labels() {
        let ss = StreamingState::new(4);
        let labels = ss.level_labels();
        assert_eq!(labels, vec!["word", "sentence", "paragraph", "topic", "episode"]);
    }

    #[test]
    fn test_ema_convergence() {
        // Feed constant input — state should converge to that value
        let mut ss = StreamingState::with_levels(2, vec![
            LevelConfig { update_interval: 1, ema_alpha: 0.5, label: "test" },
        ]);
        for _ in 0..100 {
            ss.update(&vec![3.0, -2.0]);
        }
        let s = ss.level_state(0).unwrap();
        assert!((s[0] - 3.0).abs() < 0.01, "Should converge to 3.0, got {}", s[0]);
        assert!((s[1] - (-2.0)).abs() < 0.01, "Should converge to -2.0, got {}", s[1]);
    }
}
