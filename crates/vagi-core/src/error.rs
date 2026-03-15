//! Central error type for all vagi crates.

/// Central error type for all vagi crates.
#[derive(Debug, thiserror::Error)]
pub enum VagiError {
    #[error("Shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Memory capacity exceeded: {0}")]
    MemoryExceeded(String),

    #[error("Math error: {0}")]
    Math(String),

    #[error("Physics error: {0}")]
    Physics(String),

    #[error("Dimension error: {0}")]
    Dimension(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}
