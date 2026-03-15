//! vagi-train — GENESIS Training Protocol for vAGI v2.
//!
//! GENESIS = Generative Epistemological Neural Evolution via
//! Symbolic-Inductive Synthesis
//!
//! 5-stage training: Embody → Abstract → Formalize → Compose → Consolidate

pub mod genesis;
pub mod embody;
pub mod abstract_;
pub mod formalize;
pub mod compose;
pub mod consolidate;
pub mod ewc;
pub mod curriculum;
pub mod optimizer;

pub use genesis::{GenesisScheduler, GenesisStage};
