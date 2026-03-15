//! vagi-world — World model & causal engine for vAGI v2.
//!
//! Causal DAG for structured reasoning, planning via forward simulation,
//! and intervention analysis.

pub mod causal;
pub mod planner;

pub use causal::{CausalGraph, CausalNode, CausalEdge};
pub use planner::{Planner, Plan, PlannedAction};
