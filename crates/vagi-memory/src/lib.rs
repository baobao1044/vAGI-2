//! vagi-memory — Hierarchical memory pyramid for vAGI v2.
//!
//! Implements streaming state (multi-scale EMA), two-phase attention
//! (HDC scout + dot-product focus), and connects to HDC memory.

pub mod attention;
pub mod streaming;

pub use attention::{TwoPhaseAttention, TwoPhaseConfig};
pub use streaming::{StreamingState, LevelConfig};
