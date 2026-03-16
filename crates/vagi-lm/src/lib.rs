//! vagi-lm — Ternary language model for vAGI v2.
//!
//! Byte-level tokenizer → Embedding → Transformer (STELinear + AdaptiveBlock) → LM Head.
//! All linear projections use ternary weights with STE training.
//!
//! Training: cross-entropy loss + backpropagation through all layers via STE.
//! Dataset: text tokenization and sequence batching for training.

#![allow(dead_code)]

pub mod tokenizer;
pub mod config;
pub mod embedding;
pub mod attention;
pub mod transformer;
pub mod model;
pub mod training;
pub mod dataset;

pub use tokenizer::ByteTokenizer;
pub use config::LMConfig;
pub use embedding::Embedding;
pub use attention::CausalAttention;
pub use transformer::TransformerLayer;
pub use model::VagiLM;
pub use dataset::TextDataset;
pub use training::{LMTrainer, AdvancedConfig, TrainMetrics, TrainConfig};
