//! vagi-lm — Ternary language model for vAGI v2.
//!
//! Byte-level tokenizer → Embedding → Transformer (STELinear + AdaptiveBlock) → LM Head.
//! All linear projections use ternary weights with STE training.
//!
//! Training: cross-entropy loss + backpropagation through all layers via STE.
//! Dataset: text tokenization and sequence batching for training.
//! Checkpoint: save/load model weights and optimizer state.

#![allow(dead_code)]

pub mod tokenizer;
pub mod config;
pub mod embedding;
pub mod attention;
pub mod transformer;
pub mod model;
pub mod training;
pub mod dataset;
pub mod checkpoint;
pub mod eval;
pub mod tokenizer_bpe;
pub mod fast_train;
pub mod sign_sgd;

pub use tokenizer::ByteTokenizer;
pub use config::LMConfig;
pub use embedding::Embedding;
pub use attention::{CausalAttention, KVCache};
pub use transformer::TransformerLayer;
pub use model::VagiLM;
pub use dataset::TextDataset;
pub use training::{LMTrainer, AdvancedConfig, TrainMetrics, TrainConfig};
pub use checkpoint::{save_model, load_model, save_checkpoint, load_checkpoint};
pub use eval::{evaluate, EvalReport};
pub use fast_train::{batch_train_step, f32_forward_backward, apply_gradients, GradBuffer};
pub use sign_sgd::{SignSGDTrainer, sign_sgd_batch};
