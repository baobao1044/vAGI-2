//! Equation solving.
//!
//! Basic symbolic equation solver for simple cases:
//! linear equations, quadratics, and substitution-based solving.

use crate::expr::Expr;
use crate::simplify::simplify;

/// Attempt to solve `equation = 0` for the given variable.
///
/// Returns a list of possible solutions (there may be multiple roots).
/// Returns empty vec if the equation cannot be solved symbolically.
pub fn solve(equation: &Expr, var: &str) -> Vec<Expr> {
    // Try linear: ax + b = 0 → x = -b/a
    if let Some(solution) = solve_linear(equation, var) {
        return vec![solution];
    }
    // TODO: quadratic solver, polynomial solver
    vec![]
}

/// Solve a linear equation ax + b = 0 for var.
///
/// Identifies a and b by evaluating the expression at var=0 and var=1:
///   b = f(0)
///   a = f(1) - f(0)
/// Then x = -b/a.
fn solve_linear(expr: &Expr, var: &str) -> Option<Expr> {
    use std::collections::HashMap;

    // Check: is this actually linear? (appears at most once, not in powers/trig)
    let vars = expr.free_vars();
    if !vars.contains(&var.to_string()) {
        return None;
    }

    // Evaluate at var=0 to get b
    let mut bindings_0 = HashMap::new();
    bindings_0.insert(var.to_string(), 0.0);

    // Evaluate at var=1 to get a+b
    let mut bindings_1 = HashMap::new();
    bindings_1.insert(var.to_string(), 1.0);

    // If the expression contains other free variables, we can't solve numerically
    if vars.len() > 1 {
        return None;
    }

    let b = expr.eval(&bindings_0)?;
    let a_plus_b = expr.eval(&bindings_1)?;
    let a = a_plus_b - b;

    if a.abs() < 1e-15 {
        return None; // Not actually linear in var (or degenerate)
    }

    // Verify linearity: check at var=2
    let mut bindings_2 = HashMap::new();
    bindings_2.insert(var.to_string(), 2.0);
    let at_2 = expr.eval(&bindings_2)?;
    let expected_2 = 2.0 * a + b;
    if (at_2 - expected_2).abs() > 1e-10 {
        return None; // Not linear
    }

    // x = -b/a
    Some(simplify(&Expr::num(-b / a)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_linear() {
        // 2x + 6 = 0 → x = -3
        let eq = Expr::num(2.0).mul(Expr::var("x")).add(Expr::num(6.0));
        let solutions = solve(&eq, "x");
        assert_eq!(solutions.len(), 1);
        let val = solutions[0].as_f64().unwrap();
        assert!((val - (-3.0)).abs() < 1e-10, "Expected -3, got {val}");
    }

    #[test]
    fn test_solve_no_var() {
        // 5 = 0 → no solution for x
        let eq = Expr::num(5.0);
        let solutions = solve(&eq, "x");
        assert!(solutions.is_empty());
    }
}
