//! vagi-hdc — Hyperdimensional Computing engine.
//!
//! Binary hypervector memory with sub-millisecond retrieval.
//! Features: HDC encoding, in-memory search, SQLite persistence,
//! forgetting with decay/pruning/merging.

pub mod encoder;
pub mod memory;
pub mod vector;

pub use encoder::HDCEncoder;
pub use memory::{Episode, ForgettingPolicy, HDCMemory, MaintenanceReport, MemoryConfig};
pub use vector::HyperVector;
