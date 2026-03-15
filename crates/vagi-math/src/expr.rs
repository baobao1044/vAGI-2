//! Mathematical expression AST.
//!
//! Expressions are trees, not strings. This module defines the `Expr` enum
//! which represents mathematical expressions as an abstract syntax tree,
//! along with display, evaluation, and utility methods.

use std::collections::HashMap;
use std::fmt;

/// Mathematical expression as an abstract syntax tree.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    // ── Atoms ──────────────────────────────────────────────────
    /// Numeric constant.
    Const(f64),
    /// Named variable (e.g., "x", "y").
    Var(String),
    /// Symbolic constant (e.g., "pi", "e").
    Symbol(String),

    // ── Arithmetic ─────────────────────────────────────────────
    /// Addition: a + b.
    Add(Box<Expr>, Box<Expr>),
    /// Multiplication: a * b.
    Mul(Box<Expr>, Box<Expr>),
    /// Unary negation: -a.
    Neg(Box<Expr>),
    /// Multiplicative inverse: 1/a.
    Inv(Box<Expr>),
    /// Exponentiation: a^b.
    Pow(Box<Expr>, Box<Expr>),

    // ── Transcendental ─────────────────────────────────────────
    /// Sine.
    Sin(Box<Expr>),
    /// Cosine.
    Cos(Box<Expr>),
    /// Exponential: e^x.
    Exp(Box<Expr>),
    /// Natural logarithm: ln(x).
    Ln(Box<Expr>),

    // ── Calculus (symbolic) ────────────────────────────────────
    /// Symbolic derivative: d(expr)/d(var).
    Derivative(Box<Expr>, String),
    /// Symbolic integral: ∫ expr d(var).
    Integral(Box<Expr>, String),

    // ── Logic / Relations ──────────────────────────────────────
    /// Equality: a = b.
    Eq(Box<Expr>, Box<Expr>),
    /// Less than: a < b.
    Lt(Box<Expr>, Box<Expr>),

    // ── Structured ─────────────────────────────────────────────
    /// Vector: [e1, e2, ...].
    Vec(Vec<Expr>),
    /// Matrix: [[e11, e12, ...], [e21, e22, ...], ...].
    Matrix(Vec<Vec<Expr>>),
    /// Summation: Σ_{var=lo}^{hi} expr.
    Sum(Box<Expr>, String, Box<Expr>, Box<Expr>),
}

// ── Convenience constructors ───────────────────────────────────

#[allow(clippy::should_implement_trait)]
impl Expr {
    /// Create numeric constant.
    pub fn num(v: f64) -> Self {
        Expr::Const(v)
    }

    /// Create variable.
    pub fn var(name: &str) -> Self {
        Expr::Var(name.to_string())
    }

    /// Create symbolic constant.
    pub fn sym(name: &str) -> Self {
        Expr::Symbol(name.to_string())
    }

    /// Create addition: self + other.
    pub fn add(self, other: Expr) -> Self {
        Expr::Add(Box::new(self), Box::new(other))
    }

    /// Create subtraction: self - other = self + (-other).
    pub fn sub(self, other: Expr) -> Self {
        Expr::Add(Box::new(self), Box::new(Expr::Neg(Box::new(other))))
    }

    /// Create multiplication: self * other.
    pub fn mul(self, other: Expr) -> Self {
        Expr::Mul(Box::new(self), Box::new(other))
    }

    /// Create division: self / other = self * (1/other).
    pub fn div(self, other: Expr) -> Self {
        Expr::Mul(Box::new(self), Box::new(Expr::Inv(Box::new(other))))
    }

    /// Create exponentiation: self ^ exp.
    pub fn pow(self, exp: Expr) -> Self {
        Expr::Pow(Box::new(self), Box::new(exp))
    }

    /// Create negation: -self.
    pub fn neg(self) -> Self {
        Expr::Neg(Box::new(self))
    }

    /// Create inverse: 1/self.
    pub fn inv(self) -> Self {
        Expr::Inv(Box::new(self))
    }

    /// Create sin(self).
    pub fn sin(self) -> Self {
        Expr::Sin(Box::new(self))
    }

    /// Create cos(self).
    pub fn cos(self) -> Self {
        Expr::Cos(Box::new(self))
    }

    /// Create symbolic derivative d(self)/d(var).
    pub fn deriv(self, var: &str) -> Self {
        Expr::Derivative(Box::new(self), var.to_string())
    }

    /// Create symbolic integral ∫ self d(var).
    pub fn integral(self, var: &str) -> Self {
        Expr::Integral(Box::new(self), var.to_string())
    }

    /// Check if this expression is a constant.
    pub fn is_const(&self) -> bool {
        matches!(self, Expr::Const(_))
    }

    /// Check if this is a variable.
    pub fn is_var(&self) -> bool {
        matches!(self, Expr::Var(_))
    }

    /// Try to extract the numeric value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Expr::Const(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract the variable name.
    pub fn as_var(&self) -> Option<&str> {
        match self {
            Expr::Var(name) => Some(name),
            _ => None,
        }
    }

    /// Check if expression is zero.
    pub fn is_zero(&self) -> bool {
        matches!(self, Expr::Const(v) if *v == 0.0)
    }

    /// Check if expression is one.
    pub fn is_one(&self) -> bool {
        matches!(self, Expr::Const(v) if *v == 1.0)
    }

    /// Number of nodes in the expression tree (complexity metric).
    pub fn node_count(&self) -> usize {
        match self {
            Expr::Const(_) | Expr::Var(_) | Expr::Symbol(_) => 1,
            Expr::Neg(a) | Expr::Inv(a) | Expr::Sin(a) | Expr::Cos(a)
            | Expr::Exp(a) | Expr::Ln(a) => 1 + a.node_count(),
            Expr::Add(a, b) | Expr::Mul(a, b) | Expr::Pow(a, b)
            | Expr::Eq(a, b) | Expr::Lt(a, b) => 1 + a.node_count() + b.node_count(),
            Expr::Derivative(a, _) | Expr::Integral(a, _) => 1 + a.node_count(),
            Expr::Vec(elems) => 1 + elems.iter().map(|e| e.node_count()).sum::<usize>(),
            Expr::Matrix(rows) => {
                1 + rows
                    .iter()
                    .flat_map(|r| r.iter())
                    .map(|e| e.node_count())
                    .sum::<usize>()
            }
            Expr::Sum(body, _, lo, hi) => 1 + body.node_count() + lo.node_count() + hi.node_count(),
        }
    }

    /// Collect all free variable names in this expression.
    pub fn free_vars(&self) -> Vec<String> {
        let mut vars = std::collections::HashSet::new();
        self.collect_vars(&mut vars);
        let mut result: Vec<_> = vars.into_iter().collect();
        result.sort();
        result
    }

    fn collect_vars(&self, out: &mut std::collections::HashSet<String>) {
        match self {
            Expr::Var(name) => {
                out.insert(name.clone());
            }
            Expr::Const(_) | Expr::Symbol(_) => {}
            Expr::Neg(a) | Expr::Inv(a) | Expr::Sin(a) | Expr::Cos(a)
            | Expr::Exp(a) | Expr::Ln(a) => a.collect_vars(out),
            Expr::Add(a, b) | Expr::Mul(a, b) | Expr::Pow(a, b)
            | Expr::Eq(a, b) | Expr::Lt(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
            }
            Expr::Derivative(a, _) | Expr::Integral(a, _) => a.collect_vars(out),
            Expr::Vec(elems) => {
                for e in elems {
                    e.collect_vars(out);
                }
            }
            Expr::Matrix(rows) => {
                for row in rows {
                    for e in row {
                        e.collect_vars(out);
                    }
                }
            }
            Expr::Sum(body, var, lo, hi) => {
                body.collect_vars(out);
                lo.collect_vars(out);
                hi.collect_vars(out);
                // The summation variable is bound, not free
                out.remove(var);
            }
        }
    }

    /// Evaluate the expression with given variable bindings.
    ///
    /// Returns `None` if the expression contains unevaluable parts
    /// (unbound variables, symbolic operations like Derivative).
    pub fn eval(&self, bindings: &HashMap<String, f64>) -> Option<f64> {
        match self {
            Expr::Const(v) => Some(*v),
            Expr::Var(name) => bindings.get(name).copied(),
            Expr::Symbol(name) => match name.as_str() {
                "pi" | "π" => Some(std::f64::consts::PI),
                "e" => Some(std::f64::consts::E),
                _ => None,
            },
            Expr::Add(a, b) => Some(a.eval(bindings)? + b.eval(bindings)?),
            Expr::Mul(a, b) => Some(a.eval(bindings)? * b.eval(bindings)?),
            Expr::Neg(a) => Some(-a.eval(bindings)?),
            Expr::Inv(a) => {
                let v = a.eval(bindings)?;
                if v.abs() < 1e-15 {
                    None
                } else {
                    Some(1.0 / v)
                }
            }
            Expr::Pow(a, b) => Some(a.eval(bindings)?.powf(b.eval(bindings)?)),
            Expr::Sin(a) => Some(a.eval(bindings)?.sin()),
            Expr::Cos(a) => Some(a.eval(bindings)?.cos()),
            Expr::Exp(a) => Some(a.eval(bindings)?.exp()),
            Expr::Ln(a) => {
                let v = a.eval(bindings)?;
                if v <= 0.0 {
                    None
                } else {
                    Some(v.ln())
                }
            }
            // Cannot evaluate symbolic operations
            Expr::Derivative(..) | Expr::Integral(..) => None,
            Expr::Eq(a, b) => {
                let va = a.eval(bindings)?;
                let vb = b.eval(bindings)?;
                Some(if (va - vb).abs() < 1e-12 { 1.0 } else { 0.0 })
            }
            Expr::Lt(a, b) => {
                let va = a.eval(bindings)?;
                let vb = b.eval(bindings)?;
                Some(if va < vb { 1.0 } else { 0.0 })
            }
            Expr::Vec(_) | Expr::Matrix(_) | Expr::Sum(..) => None,
        }
    }

    /// Substitute all occurrences of `var` with `replacement`.
    pub fn substitute(&self, var: &str, replacement: &Expr) -> Expr {
        match self {
            Expr::Var(name) if name == var => replacement.clone(),
            Expr::Var(_) | Expr::Const(_) | Expr::Symbol(_) => self.clone(),
            Expr::Neg(a) => Expr::Neg(Box::new(a.substitute(var, replacement))),
            Expr::Inv(a) => Expr::Inv(Box::new(a.substitute(var, replacement))),
            Expr::Sin(a) => Expr::Sin(Box::new(a.substitute(var, replacement))),
            Expr::Cos(a) => Expr::Cos(Box::new(a.substitute(var, replacement))),
            Expr::Exp(a) => Expr::Exp(Box::new(a.substitute(var, replacement))),
            Expr::Ln(a) => Expr::Ln(Box::new(a.substitute(var, replacement))),
            Expr::Add(a, b) => Expr::Add(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Expr::Mul(a, b) => Expr::Mul(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Expr::Pow(a, b) => Expr::Pow(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Expr::Eq(a, b) => Expr::Eq(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Expr::Lt(a, b) => Expr::Lt(
                Box::new(a.substitute(var, replacement)),
                Box::new(b.substitute(var, replacement)),
            ),
            Expr::Derivative(a, v) => {
                if v == var {
                    // The derivative variable shadows the substitution
                    self.clone()
                } else {
                    Expr::Derivative(Box::new(a.substitute(var, replacement)), v.clone())
                }
            }
            Expr::Integral(a, v) => {
                if v == var {
                    self.clone()
                } else {
                    Expr::Integral(Box::new(a.substitute(var, replacement)), v.clone())
                }
            }
            Expr::Vec(elems) => {
                Expr::Vec(elems.iter().map(|e| e.substitute(var, replacement)).collect())
            }
            Expr::Matrix(rows) => Expr::Matrix(
                rows.iter()
                    .map(|row| row.iter().map(|e| e.substitute(var, replacement)).collect())
                    .collect(),
            ),
            Expr::Sum(body, v, lo, hi) => {
                if v == var {
                    // Summation variable shadows
                    Expr::Sum(
                        body.clone(),
                        v.clone(),
                        Box::new(lo.substitute(var, replacement)),
                        Box::new(hi.substitute(var, replacement)),
                    )
                } else {
                    Expr::Sum(
                        Box::new(body.substitute(var, replacement)),
                        v.clone(),
                        Box::new(lo.substitute(var, replacement)),
                        Box::new(hi.substitute(var, replacement)),
                    )
                }
            }
        }
    }
}

// ── Display ────────────────────────────────────────────────────

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(v) => {
                if *v == v.floor() && v.abs() < 1e12 {
                    write!(f, "{}", *v as i64)
                } else {
                    write!(f, "{v}")
                }
            }
            Expr::Var(name) => write!(f, "{name}"),
            Expr::Symbol(name) => write!(f, "{name}"),
            Expr::Add(a, b) => write!(f, "({a} + {b})"),
            Expr::Mul(a, b) => write!(f, "({a} * {b})"),
            Expr::Neg(a) => write!(f, "(-{a})"),
            Expr::Inv(a) => write!(f, "(1/{a})"),
            Expr::Pow(a, b) => write!(f, "({a}^{b})"),
            Expr::Sin(a) => write!(f, "sin({a})"),
            Expr::Cos(a) => write!(f, "cos({a})"),
            Expr::Exp(a) => write!(f, "exp({a})"),
            Expr::Ln(a) => write!(f, "ln({a})"),
            Expr::Derivative(a, var) => write!(f, "d/d{var}[{a}]"),
            Expr::Integral(a, var) => write!(f, "∫{a} d{var}"),
            Expr::Eq(a, b) => write!(f, "({a} = {b})"),
            Expr::Lt(a, b) => write!(f, "({a} < {b})"),
            Expr::Vec(elems) => {
                let items: Vec<String> = elems.iter().map(|e| e.to_string()).collect();
                write!(f, "[{}]", items.join(", "))
            }
            Expr::Matrix(rows) => {
                let row_strs: Vec<String> = rows
                    .iter()
                    .map(|row| {
                        let items: Vec<String> = row.iter().map(|e| e.to_string()).collect();
                        format!("[{}]", items.join(", "))
                    })
                    .collect();
                write!(f, "[{}]", row_strs.join("; "))
            }
            Expr::Sum(body, var, lo, hi) => write!(f, "Σ_{{{var}={lo}}}^{{{hi}}} {body}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_polynomial() {
        // 3x² + 2x + 1 at x=5 → 86
        let expr = Expr::num(3.0)
            .mul(Expr::var("x").pow(Expr::num(2.0)))
            .add(Expr::num(2.0).mul(Expr::var("x")))
            .add(Expr::num(1.0));

        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), 5.0);

        let result = expr.eval(&bindings).unwrap();
        assert!((result - 86.0).abs() < 1e-10, "Expected 86, got {result}");
    }

    #[test]
    fn test_eval_trig() {
        let expr = Expr::var("x").sin().pow(Expr::num(2.0))
            .add(Expr::var("x").cos().pow(Expr::num(2.0)));

        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), 1.234);

        let result = expr.eval(&bindings).unwrap();
        assert!(
            (result - 1.0).abs() < 1e-10,
            "sin²(x) + cos²(x) should be 1.0, got {result}"
        );
    }

    #[test]
    fn test_substitute() {
        let expr = Expr::var("x").add(Expr::num(1.0));
        let result = expr.substitute("x", &Expr::num(5.0));
        let val = result.eval(&HashMap::new()).unwrap();
        assert!((val - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_vars() {
        let expr = Expr::var("x")
            .mul(Expr::var("y"))
            .add(Expr::var("z"));
        let vars = expr.free_vars();
        assert_eq!(vars, vec!["x", "y", "z"]);
    }

    #[test]
    fn test_node_count() {
        let expr = Expr::var("x").add(Expr::num(1.0));
        assert_eq!(expr.node_count(), 3); // Add, Var, Const
    }

    #[test]
    fn test_display() {
        let expr = Expr::var("x").pow(Expr::num(2.0)).add(Expr::num(1.0));
        let s = expr.to_string();
        assert!(s.contains("x") && s.contains("2") && s.contains("1"));
    }

    #[test]
    fn test_is_zero_one() {
        assert!(Expr::num(0.0).is_zero());
        assert!(!Expr::num(1.0).is_zero());
        assert!(Expr::num(1.0).is_one());
        assert!(!Expr::num(0.0).is_one());
    }
}
