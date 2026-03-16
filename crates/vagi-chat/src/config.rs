//! Configuration for chat sessions.

/// Chat session configuration.
#[derive(Clone, Debug)]
pub struct ChatConfig {
    /// Sampling temperature (0 = greedy, higher = more random).
    pub temperature: f32,
    /// Top-K: only consider the K most likely tokens.
    pub top_k: usize,
    /// Top-P (nucleus): sample from tokens covering P cumulative probability.
    pub top_p: f32,
    /// Maximum response tokens.
    pub max_response_tokens: usize,
    /// Repetition penalty (1.0 = no penalty, >1.0 = penalize repeats).
    pub repetition_penalty: f32,
    /// Optional system prompt to prefix all conversations.
    pub system_prompt: Option<String>,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            max_response_tokens: 256,
            repetition_penalty: 1.1,
            system_prompt: None,
        }
    }
}

impl ChatConfig {
    /// Greedy decoding (temperature=0, no sampling randomness).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            max_response_tokens: 256,
            repetition_penalty: 1.0,
            system_prompt: None,
        }
    }

    /// Creative mode (higher temperature, larger top_k).
    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_k: 80,
            top_p: 0.95,
            max_response_tokens: 512,
            repetition_penalty: 1.2,
            system_prompt: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let c = ChatConfig::default();
        assert!((c.temperature - 0.7).abs() < 1e-6);
        assert_eq!(c.top_k, 40);
        assert!(c.system_prompt.is_none());
    }

    #[test]
    fn test_greedy_config() {
        let c = ChatConfig::greedy();
        assert!(c.temperature < 1e-6);
        assert_eq!(c.top_k, 1);
    }
}
