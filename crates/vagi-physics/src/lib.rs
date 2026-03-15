//! vagi-physics — Physics-grounded world model for vAGI v2.
//!
//! Provides:
//! - Hamiltonian Neural Networks (energy-conserving dynamics)
//! - Symmetry discovery (Noether's theorem)
//! - Symbolic regression (law discovery from data)
//! - Dimensional analysis (unit checking)
//! - Microworld simulations

pub mod units;
pub mod hamiltonian;
pub mod symmetry;
pub mod symbolic_reg;
pub mod microworlds;
pub mod discovery;

pub use units::{DimensionalAnalyzer, Unit};
pub use hamiltonian::HamiltonianNN;
pub use symmetry::SymmetryModule;
pub use symbolic_reg::SymbolicRegressor;
pub use discovery::AbstractionEngine;
