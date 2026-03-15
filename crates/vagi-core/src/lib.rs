//! vagi-core — BitNet ops, tensor primitives, SIMD
//!
//! Foundation layer for vAGI v2. All neural computations are built on
//! ternary {-1, 0, +1} weights with f32 activations.

pub mod bitnet;
pub mod error;

pub use bitnet::{BitNetBlock, BitNetConfig, BitNetLinear, RMSNorm};
pub use error::VagiError;
