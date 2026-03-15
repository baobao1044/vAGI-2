//! vagi-core — BitNet ops, tensor primitives, SIMD
//!
//! Foundation layer for vAGI v2. All neural computations are built on
//! ternary {-1, 0, +1} weights with f32 activations.
//! AdaptiveNet extends BitNet with learnable activation functions.

pub mod adaptive;
pub mod bitnet;
pub mod error;

pub use adaptive::{AdaptiveBasis, AdaptiveBlock, BasisConfig, BasisScheduler, N_BASIS, BASIS_NAMES};
pub use bitnet::{BitNetBlock, BitNetConfig, BitNetLinear, RMSNorm};
pub use error::VagiError;
