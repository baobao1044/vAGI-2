//! Expression simplification strategies.
//!
//! Uses the rewrite engine to simplify expressions by repeatedly
//! applying algebraic rules until a fixed point is reached.

use crate::expr::Expr;
use crate::rewrite::RewriteEngine;

/// Simplify an expression using the standard rewrite rules.
///
/// Applies rules until no more transformations are possible, up to
/// `max_iterations` passes to prevent infinite loops.
pub fn simplify(expr: &Expr) -> Expr {
    let engine = RewriteEngine::new();
    engine.rewrite_fixpoint(expr, 100)
}

/// Simplify with a custom rewrite engine.
pub fn simplify_with(expr: &Expr, engine: &RewriteEngine, max_iter: usize) -> Expr {
    engine.rewrite_fixpoint(expr, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_zero_add() {
        // x + 0 → x
        let expr = Expr::var("x").add(Expr::num(0.0));
        assert_eq!(simplify(&expr), Expr::var("x"));
    }

    #[test]
    fn test_simplify_nested() {
        // (x * 1) + 0 → x
        let expr = Expr::var("x").mul(Expr::num(1.0)).add(Expr::num(0.0));
        assert_eq!(simplify(&expr), Expr::var("x"));
    }

    #[test]
    fn test_simplify_const_fold() {
        // 2 + 3 → 5
        let expr = Expr::num(2.0).add(Expr::num(3.0));
        assert_eq!(simplify(&expr), Expr::num(5.0));
    }
}
