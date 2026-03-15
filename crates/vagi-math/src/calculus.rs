//! Symbolic differentiation and integration.
//!
//! Implements calculus rules from S2.2: power rule, chain rule,
//! product rule, trig derivatives, and basic integration.

use crate::expr::Expr;

/// Compute the symbolic derivative of `expr` with respect to `var`.
///
/// Implements standard differentiation rules:
/// - Constant rule: d/dx[c] = 0
/// - Variable rule: d/dx[x] = 1
/// - Power rule: d/dx[x^n] = n * x^(n-1)
/// - Sum rule: d/dx[f + g] = f' + g'
/// - Product rule: d/dx[f * g] = f'g + fg'
/// - Chain rule for trig, exp, ln
pub fn differentiate(expr: &Expr, var: &str) -> Expr {
    match expr {
        // d/dx[c] = 0
        Expr::Const(_) | Expr::Symbol(_) => Expr::num(0.0),

        // d/dx[x] = 1, d/dx[y] = 0
        Expr::Var(name) => {
            if name == var {
                Expr::num(1.0)
            } else {
                Expr::num(0.0)
            }
        }

        // d/dx[f + g] = f' + g'
        Expr::Add(a, b) => {
            let da = differentiate(a, var);
            let db = differentiate(b, var);
            Expr::Add(Box::new(da), Box::new(db))
        }

        // d/dx[f * g] = f'g + fg'
        Expr::Mul(a, b) => {
            let da = differentiate(a, var);
            let db = differentiate(b, var);
            // f'g + fg'
            Expr::Add(
                Box::new(Expr::Mul(Box::new(da), b.clone())),
                Box::new(Expr::Mul(a.clone(), Box::new(db))),
            )
        }

        // d/dx[-f] = -f'
        Expr::Neg(a) => Expr::Neg(Box::new(differentiate(a, var))),

        // d/dx[1/f] = -f' / f²
        Expr::Inv(a) => {
            let da = differentiate(a, var);
            Expr::Neg(Box::new(Expr::Mul(
                Box::new(da),
                Box::new(Expr::Inv(Box::new(Expr::Pow(
                    a.clone(),
                    Box::new(Expr::num(2.0)),
                )))),
            )))
        }

        // d/dx[f^g] — general case
        // If g is constant: d/dx[f^n] = n * f^(n-1) * f' (power rule + chain rule)
        // General: d/dx[f^g] = f^g * (g' * ln(f) + g * f'/f) (logarithmic differentiation)
        Expr::Pow(base, exp) => {
            if exp.as_f64().is_some() {
                // Power rule: n * f^(n-1) * f'
                let df = differentiate(base, var);
                let n = exp.clone();
                let n_minus_1 = Expr::Add(exp.clone(), Box::new(Expr::num(-1.0)));
                Expr::Mul(
                    Box::new(Expr::Mul(n, Box::new(Expr::Pow(base.clone(), Box::new(n_minus_1))))),
                    Box::new(df),
                )
            } else {
                // Logarithmic differentiation: f^g * (g' * ln(f) + g * f'/f)
                let df = differentiate(base, var);
                let dg = differentiate(exp, var);
                let f_to_g = Expr::Pow(base.clone(), exp.clone());
                let term1 = Expr::Mul(Box::new(dg), Box::new(Expr::Ln(base.clone())));
                let term2 = Expr::Mul(
                    exp.clone(),
                    Box::new(Expr::Mul(
                        Box::new(df),
                        Box::new(Expr::Inv(base.clone())),
                    )),
                );
                Expr::Mul(
                    Box::new(f_to_g),
                    Box::new(Expr::Add(Box::new(term1), Box::new(term2))),
                )
            }
        }

        // d/dx[sin(f)] = cos(f) * f'
        Expr::Sin(a) => {
            let da = differentiate(a, var);
            Expr::Mul(Box::new(Expr::Cos(a.clone())), Box::new(da))
        }

        // d/dx[cos(f)] = -sin(f) * f'
        Expr::Cos(a) => {
            let da = differentiate(a, var);
            Expr::Neg(Box::new(Expr::Mul(
                Box::new(Expr::Sin(a.clone())),
                Box::new(da),
            )))
        }

        // d/dx[e^f] = e^f * f'
        Expr::Exp(a) => {
            let da = differentiate(a, var);
            Expr::Mul(Box::new(Expr::Exp(a.clone())), Box::new(da))
        }

        // d/dx[ln(f)] = f' / f
        Expr::Ln(a) => {
            let da = differentiate(a, var);
            Expr::Mul(Box::new(da), Box::new(Expr::Inv(a.clone())))
        }

        // Already a derivative marker — just wrap it
        Expr::Derivative(inner, v) => {
            // d/dx[d/dy[f]] = d/dy[d/dx[f]]
            let d_inner = differentiate(inner, var);
            Expr::Derivative(Box::new(d_inner), v.clone())
        }

        // Other structures — not differentiable in general
        _ => Expr::Derivative(Box::new(expr.clone()), var.to_string()),
    }
}

/// Attempt basic symbolic integration of `expr` with respect to `var`.
///
/// Returns `None` for integrals that cannot be computed symbolically.
/// Implements:
/// - ∫ c dx = cx
/// - ∫ x dx = x²/2
/// - ∫ x^n dx = x^(n+1)/(n+1) for n ≠ -1
/// - ∫ sin(x) dx = -cos(x)
/// - ∫ cos(x) dx = sin(x)
/// - ∫ e^x dx = e^x
/// - ∫ 1/x dx = ln(x)
pub fn integrate(expr: &Expr, var: &str) -> Option<Expr> {
    match expr {
        // ∫ c dx = c*x
        Expr::Const(c) => Some(Expr::Mul(
            Box::new(Expr::num(*c)),
            Box::new(Expr::var(var)),
        )),

        // ∫ x dx = x²/2
        Expr::Var(name) if name == var => Some(Expr::Mul(
            Box::new(Expr::Pow(
                Box::new(Expr::var(var)),
                Box::new(Expr::num(2.0)),
            )),
            Box::new(Expr::Inv(Box::new(Expr::num(2.0)))),
        )),

        // ∫ y dx = y*x (y independent of x)
        Expr::Var(name) if name != var => Some(Expr::Mul(
            Box::new(expr.clone()),
            Box::new(Expr::var(var)),
        )),

        // ∫ (f + g) dx = ∫f dx + ∫g dx
        Expr::Add(a, b) => {
            let ia = integrate(a, var)?;
            let ib = integrate(b, var)?;
            Some(Expr::Add(Box::new(ia), Box::new(ib)))
        }

        // ∫ x^n dx = x^(n+1)/(n+1) when n is constant and n ≠ -1
        Expr::Pow(base, exp) => {
            if let Expr::Var(name) = base.as_ref() {
                if name == var {
                    if let Some(n) = exp.as_f64() {
                        if (n + 1.0).abs() < 1e-15 {
                            // ∫ x^(-1) dx = ln(x)
                            return Some(Expr::Ln(Box::new(Expr::var(var))));
                        }
                        let n1 = n + 1.0;
                        return Some(Expr::Mul(
                            Box::new(Expr::Pow(
                                Box::new(Expr::var(var)),
                                Box::new(Expr::num(n1)),
                            )),
                            Box::new(Expr::Inv(Box::new(Expr::num(n1)))),
                        ));
                    }
                }
            }
            None
        }

        // ∫ sin(x) dx = -cos(x)
        Expr::Sin(a) => {
            if let Expr::Var(name) = a.as_ref() {
                if name == var {
                    return Some(Expr::Neg(Box::new(Expr::Cos(Box::new(Expr::var(var))))));
                }
            }
            None
        }

        // ∫ cos(x) dx = sin(x)
        Expr::Cos(a) => {
            if let Expr::Var(name) = a.as_ref() {
                if name == var {
                    return Some(Expr::Sin(Box::new(Expr::var(var))));
                }
            }
            None
        }

        // ∫ e^x dx = e^x
        Expr::Exp(a) => {
            if let Expr::Var(name) = a.as_ref() {
                if name == var {
                    return Some(Expr::Exp(Box::new(Expr::var(var))));
                }
            }
            None
        }

        // ∫ c*f dx = c * ∫f dx (constant factor)
        Expr::Mul(a, b) => {
            // Check if a is independent of var
            if !a.free_vars().contains(&var.to_string()) {
                if let Some(ib) = integrate(b, var) {
                    return Some(Expr::Mul(a.clone(), Box::new(ib)));
                }
            }
            // Check if b is independent of var
            if !b.free_vars().contains(&var.to_string()) {
                if let Some(ia) = integrate(a, var) {
                    return Some(Expr::Mul(Box::new(ia), b.clone()));
                }
            }
            None
        }

        // ∫ -f dx = -(∫f dx)
        Expr::Neg(a) => {
            let ia = integrate(a, var)?;
            Some(Expr::Neg(Box::new(ia)))
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_diff_constant() {
        let expr = Expr::num(5.0);
        let result = differentiate(&expr, "x");
        assert_eq!(result, Expr::num(0.0));
    }

    #[test]
    fn test_diff_variable() {
        let result = differentiate(&Expr::var("x"), "x");
        assert_eq!(result, Expr::num(1.0));

        let result = differentiate(&Expr::var("y"), "x");
        assert_eq!(result, Expr::num(0.0));
    }

    #[test]
    fn test_diff_x_squared() {
        // d/dx[x²] = 2x
        let expr = Expr::var("x").pow(Expr::num(2.0));
        let deriv = differentiate(&expr, "x");

        // Evaluate at x=3: should be 6
        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), 3.0);

        let val = deriv.eval(&bindings).unwrap();
        assert!(
            (val - 6.0).abs() < 1e-10,
            "d/dx[x²] at x=3 should be 6.0, got {val}"
        );
    }

    #[test]
    fn test_diff_sin() {
        // d/dx[sin(x)] = cos(x)
        let expr = Expr::var("x").sin();
        let deriv = differentiate(&expr, "x");

        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), 0.0);
        let val = deriv.eval(&bindings).unwrap();
        assert!((val - 1.0).abs() < 1e-10, "cos(0) should be 1.0, got {val}");
    }

    #[test]
    fn test_diff_product() {
        // d/dx[x * sin(x)] = sin(x) + x*cos(x)
        let expr = Expr::var("x").mul(Expr::var("x").sin());
        let deriv = differentiate(&expr, "x");

        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), 1.0);

        let val = deriv.eval(&bindings).unwrap();
        let expected = 1.0f64.sin() + 1.0 * 1.0f64.cos();
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {expected}, got {val}"
        );
    }

    #[test]
    fn test_integrate_x_squared() {
        // ∫ x² dx = x³/3
        let expr = Expr::var("x").pow(Expr::num(2.0));
        let result = integrate(&expr, "x").expect("Should integrate x²");

        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), 3.0);

        let val = result.eval(&bindings).unwrap();
        assert!(
            (val - 9.0).abs() < 1e-10,
            "∫x² at x=3 should be 9.0 (3³/3), got {val}"
        );
    }

    #[test]
    fn test_integrate_sin() {
        // ∫ sin(x) dx = -cos(x)
        let expr = Expr::var("x").sin();
        let result = integrate(&expr, "x").expect("Should integrate sin(x)");

        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), 0.0);

        let val = result.eval(&bindings).unwrap();
        assert!(
            (val - (-1.0)).abs() < 1e-10,
            "-cos(0) should be -1.0, got {val}"
        );
    }
}
