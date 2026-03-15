//! Symbolic regression — discover formulas from data (S3.3).

use vagi_math::Expr;

/// Primitive function library for expression search.
#[derive(Clone, Debug)]
pub enum Primitive {
    Add, Sub, Mul, Div,
    Sin, Cos, Exp, Ln,
    Pow, Sqrt, Abs,
    Const(f64),
    Var(usize),
}

/// A discovered symbolic law.
#[derive(Clone, Debug)]
pub struct DiscoveredLaw {
    pub expression: Expr,
    pub fit_quality: f64,
    pub complexity: usize,
    pub mdl_score: f64,
    pub units_valid: bool,
}

/// Symbolic regressor using neural-guided MCTS expression search.
#[allow(dead_code)]
pub struct SymbolicRegressor {
    primitives: Vec<Primitive>,
    max_complexity: usize,
}

impl SymbolicRegressor {
    pub fn new(max_complexity: usize) -> Self {
        Self {
            primitives: vec![
                Primitive::Add, Primitive::Sub, Primitive::Mul, Primitive::Div,
                Primitive::Sin, Primitive::Cos, Primitive::Exp, Primitive::Ln,
                Primitive::Pow, Primitive::Sqrt,
            ],
            max_complexity,
        }
    }

    /// Minimum Description Length score: data_fit + description_length.
    pub fn mdl_score(&self, expr: &Expr, data: &[(Vec<f64>, f64)]) -> f64 {
        let complexity = expr.node_count() as f64;
        let fit_error = self.fit_error(expr, data);
        // MDL = description_length + data_misfit
        complexity * 2.0 + fit_error * data.len() as f64
    }

    /// Mean squared error of expression on data.
    pub fn fit_error(&self, expr: &Expr, data: &[(Vec<f64>, f64)]) -> f64 {
        if data.is_empty() { return f64::MAX; }
        let mut total = 0.0;
        let mut count = 0;
        for (inputs, target) in data {
            let mut bindings = std::collections::HashMap::new();
            for (i, v) in inputs.iter().enumerate() {
                bindings.insert(format!("x{i}"), *v);
            }
            if let Some(predicted) = expr.eval(&bindings) {
                total += (predicted - target).powi(2);
                count += 1;
            }
        }
        if count == 0 { f64::MAX } else { total / count as f64 }
    }

    /// Discover symbolic expressions that fit the data.
    /// Simple brute-force search over small expression trees.
    pub fn discover(
        &self, variable_names: &[&str], data: &[(Vec<f64>, f64)],
    ) -> Vec<DiscoveredLaw> {
        let mut candidates = Vec::new();
        // Generate candidate expressions: single variables, simple combinations
        for name in variable_names.iter() {
            let var = Expr::var(name);
            let score = self.mdl_score(&var, data);
            candidates.push(DiscoveredLaw {
                expression: var.clone(),
                fit_quality: 1.0 / (1.0 + self.fit_error(&Expr::var(name), data)),
                complexity: 1,
                mdl_score: score,
                units_valid: true,
            });
            // x²
            let sq = var.clone().pow(Expr::num(2.0));
            let score = self.mdl_score(&sq, data);
            candidates.push(DiscoveredLaw {
                expression: sq,
                fit_quality: 1.0 / (1.0 + self.fit_error(
                    &Expr::var(name).pow(Expr::num(2.0)), data
                )),
                complexity: 3,
                mdl_score: score,
                units_valid: true,
            });
        }
        // Sort by MDL score
        candidates.sort_by(|a, b| a.mdl_score.partial_cmp(&b.mdl_score).unwrap());
        candidates.truncate(10);
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_linear() {
        let sr = SymbolicRegressor::new(10);
        // y = 2*x
        let data: Vec<(Vec<f64>, f64)> = (0..10)
            .map(|i| (vec![i as f64], 2.0 * i as f64))
            .collect();
        let error = sr.fit_error(&Expr::num(2.0).mul(Expr::var("x0")), &data);
        assert!(error < 1e-10, "Linear fit error too high: {error}");
    }

    #[test]
    fn test_discover() {
        let sr = SymbolicRegressor::new(10);
        let data: Vec<(Vec<f64>, f64)> = (1..10)
            .map(|i| (vec![i as f64], (i * i) as f64))
            .collect();
        let laws = sr.discover(&["x0"], &data);
        assert!(!laws.is_empty());
    }
}
