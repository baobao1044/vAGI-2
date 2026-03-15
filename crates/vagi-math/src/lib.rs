//! vagi-math — Mathematical foundation layer for vAGI v2.
//!
//! Provides a dual-track reasoning system:
//! - **Neural track** (BitNet): fast, approximate, pattern-based heuristics
//! - **Symbolic track** (rule engine): exact, guaranteed correct, axiom-based
//!
//! The neural track proposes candidate transformations and the symbolic
//! track verifies and executes them. Results are guaranteed correct but
//! guided efficiently.

pub mod expr;
pub mod rewrite;
pub mod simplify;
pub mod calculus;
pub mod linear_algebra;
pub mod solver;
pub mod proof;
pub mod embedding;
pub mod reasoner;

pub use expr::Expr;
pub use rewrite::{RewriteEngine, RewriteRule, RuleCategory};
pub use proof::Proof;
pub use reasoner::MathReasoner;
pub use embedding::ExprEncoder;
