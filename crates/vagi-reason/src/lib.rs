//! vagi-reason — Reasoning Engine (MoE + Predictive Coding).
//!
//! Energy-based sparse MoE with per-expert AdaptiveBasis activations
//! and predictive coding gate for surprise-driven information filtering.

pub mod expert;
pub mod gate;
pub mod router;

pub use expert::{ExpertPool, ExpertPoolConfig};
pub use gate::{PredictiveGate, PredictiveGateConfig};
pub use router::{EnergyRouter, RouterConfig, RoutingDecision};
