//! Symbolic linear algebra operations.

use crate::expr::Expr;

/// Transpose a symbolic matrix.
pub fn transpose(matrix: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    if matrix.is_empty() {
        return vec![];
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![vec![Expr::num(0.0); rows]; cols];
    for (i, row) in matrix.iter().enumerate() {
        for (j, elem) in row.iter().enumerate() {
            result[j][i] = elem.clone();
        }
    }
    result
}

/// Compute symbolic matrix-vector product Ax.
pub fn mat_vec_mul(matrix: &[Vec<Expr>], vec: &[Expr]) -> Vec<Expr> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vec.iter())
                .fold(Expr::num(0.0), |acc, (a, b)| {
                    acc.add(a.clone().mul(b.clone()))
                })
        })
        .collect()
}

/// Add two symbolic vectors.
pub fn vec_add(a: &[Expr], b: &[Expr]) -> Vec<Expr> {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai.clone().add(bi.clone()))
        .collect()
}

/// Dot product of two symbolic vectors.
pub fn dot(a: &[Expr], b: &[Expr]) -> Expr {
    a.iter()
        .zip(b.iter())
        .fold(Expr::num(0.0), |acc, (ai, bi)| {
            acc.add(ai.clone().mul(bi.clone()))
        })
}

/// Create a symbolic identity matrix.
pub fn identity(n: usize) -> Vec<Vec<Expr>> {
    let mut result = vec![vec![Expr::num(0.0); n]; n];
    for i in 0..n {
        result[i][i] = Expr::num(1.0);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_transpose() {
        let m = vec![
            vec![Expr::num(1.0), Expr::num(2.0)],
            vec![Expr::num(3.0), Expr::num(4.0)],
        ];
        let t = transpose(&m);
        assert_eq!(t[0][0], Expr::num(1.0));
        assert_eq!(t[0][1], Expr::num(3.0));
        assert_eq!(t[1][0], Expr::num(2.0));
        assert_eq!(t[1][1], Expr::num(4.0));
    }

    #[test]
    fn test_identity() {
        let id = identity(3);
        assert_eq!(id[0][0], Expr::num(1.0));
        assert_eq!(id[0][1], Expr::num(0.0));
        assert_eq!(id[1][1], Expr::num(1.0));
    }

    #[test]
    fn test_dot_product() {
        let a = vec![Expr::num(1.0), Expr::num(2.0), Expr::num(3.0)];
        let b = vec![Expr::num(4.0), Expr::num(5.0), Expr::num(6.0)];
        let result = dot(&a, &b);
        let val = result.eval(&HashMap::new()).unwrap();
        // After simplification: 1*4 + 2*5 + 3*6 = 32
        // But without simplification we get nested adds/muls
        // The eval should still yield correct result
        assert!((val - 32.0).abs() < 1e-10);
    }
}
