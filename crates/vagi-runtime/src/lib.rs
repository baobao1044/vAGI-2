//! vagi-runtime — OODA loop + orchestrator for vAGI v2.
//!
//! Wires together all vAGI layers (core, HDC, memory, reasoning, world)
//! into a unified processing pipeline using the OODA loop pattern.

pub mod ooda;

pub use ooda::{OODALoop, OODAConfig, CycleMetrics};
