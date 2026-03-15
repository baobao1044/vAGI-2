//! Rewrite rule engine for algebraic transformations.
//!
//! Provides verified algebraic rewrite rules that transform expressions
//! while preserving mathematical equivalence.

use crate::expr::Expr;

/// Category of a rewrite rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuleCategory {
    Arithmetic,
    Algebra,
    Trigonometry,
    Calculus,
    LinearAlgebra,
    Logic,
}

/// A rewrite rule: pattern → replacement, with conditions.
#[derive(Clone, Debug)]
pub struct RewriteRule {
    pub name: String,
    pub category: RuleCategory,
    /// A function that attempts to apply this rule to an expression.
    /// Returns `Some(new_expr)` if the rule applies, `None` otherwise.
    apply_fn: fn(&Expr) -> Option<Expr>,
}

impl RewriteRule {
    /// Create a new rewrite rule.
    pub fn new(name: &str, category: RuleCategory, apply_fn: fn(&Expr) -> Option<Expr>) -> Self {
        Self {
            name: name.to_string(),
            category,
            apply_fn,
        }
    }

    /// Try to apply this rule to an expression.
    pub fn apply(&self, expr: &Expr) -> Option<Expr> {
        (self.apply_fn)(expr)
    }
}

/// The rewrite engine holds a collection of rules and applies them.
pub struct RewriteEngine {
    rules: Vec<RewriteRule>,
}

impl RewriteEngine {
    /// Create a new engine with the standard rule sets.
    pub fn new() -> Self {
        let mut rules = Vec::new();
        rules.extend(arithmetic_rules());
        rules.extend(algebra_rules());
        rules.extend(trigonometry_rules());
        Self { rules }
    }

    /// Create an empty engine (no built-in rules).
    pub fn empty() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a custom rewrite rule.
    pub fn add_rule(&mut self, rule: RewriteRule) {
        self.rules.push(rule);
    }

    /// Number of loaded rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Try applying all rules to the expression (top-level only).
    /// Returns the first successful rewrite and the rule name.
    pub fn apply_one(&self, expr: &Expr) -> Option<(Expr, String)> {
        for rule in &self.rules {
            if let Some(result) = rule.apply(expr) {
                return Some((result, rule.name.clone()));
            }
        }
        None
    }

    /// Apply rules recursively to all subexpressions (one pass, bottom-up).
    pub fn rewrite_once(&self, expr: &Expr) -> Expr {
        // First rewrite children
        let rewritten = match expr {
            Expr::Add(a, b) => Expr::Add(
                Box::new(self.rewrite_once(a)),
                Box::new(self.rewrite_once(b)),
            ),
            Expr::Mul(a, b) => Expr::Mul(
                Box::new(self.rewrite_once(a)),
                Box::new(self.rewrite_once(b)),
            ),
            Expr::Pow(a, b) => Expr::Pow(
                Box::new(self.rewrite_once(a)),
                Box::new(self.rewrite_once(b)),
            ),
            Expr::Neg(a) => Expr::Neg(Box::new(self.rewrite_once(a))),
            Expr::Inv(a) => Expr::Inv(Box::new(self.rewrite_once(a))),
            Expr::Sin(a) => Expr::Sin(Box::new(self.rewrite_once(a))),
            Expr::Cos(a) => Expr::Cos(Box::new(self.rewrite_once(a))),
            Expr::Exp(a) => Expr::Exp(Box::new(self.rewrite_once(a))),
            Expr::Ln(a) => Expr::Ln(Box::new(self.rewrite_once(a))),
            other => other.clone(),
        };

        // Then try to apply a rule at the top level
        match self.apply_one(&rewritten) {
            Some((result, _)) => result,
            None => rewritten,
        }
    }

    /// Repeatedly apply rules until no more changes occur (fixed point).
    /// Caps at `max_iterations` to prevent infinite loops.
    pub fn rewrite_fixpoint(&self, expr: &Expr, max_iterations: usize) -> Expr {
        let mut current = expr.clone();
        for _ in 0..max_iterations {
            let next = self.rewrite_once(&current);
            if next == current {
                break;
            }
            current = next;
        }
        current
    }
}

impl Default for RewriteEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Arithmetic rules ───────────────────────────────────────────

fn arithmetic_rules() -> Vec<RewriteRule> {
    vec![
        // a + 0 → a
        RewriteRule::new("add_zero", RuleCategory::Arithmetic, |expr| {
            if let Expr::Add(a, b) = expr {
                if b.is_zero() {
                    return Some(*a.clone());
                }
                if a.is_zero() {
                    return Some(*b.clone());
                }
            }
            None
        }),
        // a * 1 → a
        RewriteRule::new("mul_one", RuleCategory::Arithmetic, |expr| {
            if let Expr::Mul(a, b) = expr {
                if b.is_one() {
                    return Some(*a.clone());
                }
                if a.is_one() {
                    return Some(*b.clone());
                }
            }
            None
        }),
        // a * 0 → 0
        RewriteRule::new("mul_zero", RuleCategory::Arithmetic, |expr| {
            if let Expr::Mul(a, b) = expr {
                if a.is_zero() || b.is_zero() {
                    return Some(Expr::num(0.0));
                }
            }
            None
        }),
        // --a → a (double negation)
        RewriteRule::new("double_neg", RuleCategory::Arithmetic, |expr| {
            if let Expr::Neg(inner) = expr {
                if let Expr::Neg(a) = inner.as_ref() {
                    return Some(*a.clone());
                }
            }
            None
        }),
        // a^0 → 1 (when a ≠ 0)
        RewriteRule::new("pow_zero", RuleCategory::Arithmetic, |expr| {
            if let Expr::Pow(a, b) = expr {
                if b.is_zero() && !a.is_zero() {
                    return Some(Expr::num(1.0));
                }
            }
            None
        }),
        // a^1 → a
        RewriteRule::new("pow_one", RuleCategory::Arithmetic, |expr| {
            if let Expr::Pow(a, b) = expr {
                if b.is_one() {
                    return Some(*a.clone());
                }
            }
            None
        }),
        // const + const → const
        RewriteRule::new("const_add", RuleCategory::Arithmetic, |expr| {
            if let Expr::Add(a, b) = expr {
                if let (Some(va), Some(vb)) = (a.as_f64(), b.as_f64()) {
                    return Some(Expr::num(va + vb));
                }
            }
            None
        }),
        // const * const → const
        RewriteRule::new("const_mul", RuleCategory::Arithmetic, |expr| {
            if let Expr::Mul(a, b) = expr {
                if let (Some(va), Some(vb)) = (a.as_f64(), b.as_f64()) {
                    return Some(Expr::num(va * vb));
                }
            }
            None
        }),
        // const ^ const → const
        RewriteRule::new("const_pow", RuleCategory::Arithmetic, |expr| {
            if let Expr::Pow(a, b) = expr {
                if let (Some(va), Some(vb)) = (a.as_f64(), b.as_f64()) {
                    let result = va.powf(vb);
                    if result.is_finite() {
                        return Some(Expr::num(result));
                    }
                }
            }
            None
        }),
        // -0 → 0
        RewriteRule::new("neg_zero", RuleCategory::Arithmetic, |expr| {
            if let Expr::Neg(a) = expr {
                if a.is_zero() {
                    return Some(Expr::num(0.0));
                }
            }
            None
        }),
        // 1/(1/a) → a
        RewriteRule::new("double_inv", RuleCategory::Arithmetic, |expr| {
            if let Expr::Inv(inner) = expr {
                if let Expr::Inv(a) = inner.as_ref() {
                    return Some(*a.clone());
                }
            }
            None
        }),
    ]
}

// ── Algebra rules ──────────────────────────────────────────────

fn algebra_rules() -> Vec<RewriteRule> {
    vec![
        // a + (-a) → 0  (requires structural equality check)
        RewriteRule::new("add_neg_cancel", RuleCategory::Algebra, |expr| {
            if let Expr::Add(a, b) = expr {
                if let Expr::Neg(bn) = b.as_ref() {
                    if **a == **bn {
                        return Some(Expr::num(0.0));
                    }
                }
                if let Expr::Neg(an) = a.as_ref() {
                    if **b == **an {
                        return Some(Expr::num(0.0));
                    }
                }
            }
            None
        }),
        // (a^m)^n → a^(m*n)
        RewriteRule::new("pow_mul", RuleCategory::Algebra, |expr| {
            if let Expr::Pow(base, n) = expr {
                if let Expr::Pow(a, m) = base.as_ref() {
                    return Some(Expr::Pow(
                        a.clone(),
                        Box::new(Expr::Mul(m.clone(), n.clone())),
                    ));
                }
            }
            None
        }),
        // a * a → a^2
        RewriteRule::new("square", RuleCategory::Algebra, |expr| {
            if let Expr::Mul(a, b) = expr {
                if **a == **b {
                    return Some(Expr::Pow(a.clone(), Box::new(Expr::num(2.0))));
                }
            }
            None
        }),
    ]
}

// ── Trigonometry rules ─────────────────────────────────────────

fn trigonometry_rules() -> Vec<RewriteRule> {
    vec![
        // sin(0) → 0
        RewriteRule::new("sin_zero", RuleCategory::Trigonometry, |expr| {
            if let Expr::Sin(a) = expr {
                if a.is_zero() {
                    return Some(Expr::num(0.0));
                }
            }
            None
        }),
        // cos(0) → 1
        RewriteRule::new("cos_zero", RuleCategory::Trigonometry, |expr| {
            if let Expr::Cos(a) = expr {
                if a.is_zero() {
                    return Some(Expr::num(1.0));
                }
            }
            None
        }),
        // sin²(x) + cos²(x) → 1
        RewriteRule::new("pythagorean_identity", RuleCategory::Trigonometry, |expr| {
            if let Expr::Add(a, b) = expr {
                // Check: a = sin(x)^2, b = cos(x)^2
                if let (Expr::Pow(sin_part, exp_a), Expr::Pow(cos_part, exp_b)) =
                    (a.as_ref(), b.as_ref())
                {
                    if let (Some(2.0), Some(2.0)) = (exp_a.as_f64(), exp_b.as_f64()) {
                        if let (Expr::Sin(x1), Expr::Cos(x2)) =
                            (sin_part.as_ref(), cos_part.as_ref())
                        {
                            if x1 == x2 {
                                return Some(Expr::num(1.0));
                            }
                        }
                        // Also check cos² + sin²
                        if let (Expr::Cos(x1), Expr::Sin(x2)) =
                            (sin_part.as_ref(), cos_part.as_ref())
                        {
                            if x1 == x2 {
                                return Some(Expr::num(1.0));
                            }
                        }
                    }
                }
            }
            None
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_zero() {
        let engine = RewriteEngine::new();
        let expr = Expr::var("x").add(Expr::num(0.0));
        let result = engine.rewrite_once(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_mul_one() {
        let engine = RewriteEngine::new();
        let expr = Expr::var("x").mul(Expr::num(1.0));
        let result = engine.rewrite_once(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_mul_zero() {
        let engine = RewriteEngine::new();
        let expr = Expr::var("x").mul(Expr::num(0.0));
        let result = engine.rewrite_once(&expr);
        assert_eq!(result, Expr::num(0.0));
    }

    #[test]
    fn test_const_folding() {
        let engine = RewriteEngine::new();
        let expr = Expr::num(3.0).add(Expr::num(4.0));
        let result = engine.rewrite_once(&expr);
        assert_eq!(result, Expr::num(7.0));
    }

    #[test]
    fn test_pythagorean_identity() {
        let engine = RewriteEngine::new();
        let expr = Expr::var("x")
            .sin()
            .pow(Expr::num(2.0))
            .add(Expr::var("x").cos().pow(Expr::num(2.0)));
        let result = engine.rewrite_once(&expr);
        assert_eq!(result, Expr::num(1.0));
    }

    #[test]
    fn test_fixpoint() {
        let engine = RewriteEngine::new();
        // (x + 0) * 1 → x
        let expr = Expr::var("x").add(Expr::num(0.0)).mul(Expr::num(1.0));
        let result = engine.rewrite_fixpoint(&expr, 10);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_double_negation() {
        let engine = RewriteEngine::new();
        let expr = Expr::var("x").neg().neg();
        let result = engine.rewrite_once(&expr);
        assert_eq!(result, Expr::var("x"));
    }

    #[test]
    fn test_rule_count() {
        let engine = RewriteEngine::new();
        assert!(engine.rule_count() > 10, "Should have >10 rules loaded");
    }
}
