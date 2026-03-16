//! vagi-chat — Conversational interface for vAGI v2.
//!
//! Provides chat sessions with multi-turn dialogue, advanced sampling
//! (top-k, top-p, repetition penalty), and conversation management.

#![allow(dead_code)]

pub mod config;
pub mod sampling;
pub mod session;

pub use config::ChatConfig;
pub use sampling::{top_k_sample, top_p_sample, apply_repetition_penalty};
pub use session::{ChatSession, ChatMessage, Role};
