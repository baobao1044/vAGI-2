//! Physical unit system and dimensional analysis (S3.4).

use std::collections::HashMap;
use vagi_math::Expr;

/// Physical unit using SI base dimensions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Unit {
    pub kg: i8,  // mass
    pub m: i8,   // length
    pub s: i8,   // time
    pub a: i8,   // electric current
    pub k: i8,   // temperature
    pub mol: i8, // amount
    pub cd: i8,  // luminosity
}

impl Unit {
    pub fn dimensionless() -> Self {
        Self { kg: 0, m: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 }
    }
    pub fn meter() -> Self {
        Self { kg: 0, m: 1, s: 0, a: 0, k: 0, mol: 0, cd: 0 }
    }
    pub fn kilogram() -> Self {
        Self { kg: 1, m: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 }
    }
    pub fn second() -> Self {
        Self { kg: 0, m: 0, s: 1, a: 0, k: 0, mol: 0, cd: 0 }
    }
    /// Newton: kg⋅m⋅s⁻²
    pub fn newton() -> Self {
        Self { kg: 1, m: 1, s: -2, a: 0, k: 0, mol: 0, cd: 0 }
    }
    /// Joule: kg⋅m²⋅s⁻²
    pub fn joule() -> Self {
        Self { kg: 1, m: 2, s: -2, a: 0, k: 0, mol: 0, cd: 0 }
    }
    /// Velocity: m⋅s⁻¹
    pub fn velocity() -> Self {
        Self { kg: 0, m: 1, s: -1, a: 0, k: 0, mol: 0, cd: 0 }
    }
    /// Acceleration: m⋅s⁻²
    pub fn acceleration() -> Self {
        Self { kg: 0, m: 1, s: -2, a: 0, k: 0, mol: 0, cd: 0 }
    }

    pub fn compatible(&self, other: &Self) -> bool {
        self == other
    }

    pub fn multiply(&self, other: &Self) -> Self {
        Self {
            kg: self.kg + other.kg, m: self.m + other.m, s: self.s + other.s,
            a: self.a + other.a, k: self.k + other.k, mol: self.mol + other.mol,
            cd: self.cd + other.cd,
        }
    }

    pub fn divide(&self, other: &Self) -> Self {
        Self {
            kg: self.kg - other.kg, m: self.m - other.m, s: self.s - other.s,
            a: self.a - other.a, k: self.k - other.k, mol: self.mol - other.mol,
            cd: self.cd - other.cd,
        }
    }

    pub fn pow(&self, n: i8) -> Self {
        Self {
            kg: self.kg * n, m: self.m * n, s: self.s * n,
            a: self.a * n, k: self.k * n, mol: self.mol * n,
            cd: self.cd * n,
        }
    }

    pub fn is_dimensionless(&self) -> bool {
        *self == Self::dimensionless()
    }
}

impl std::fmt::Display for Unit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts = Vec::new();
        let dims = [
            (self.kg, "kg"), (self.m, "m"), (self.s, "s"),
            (self.a, "A"), (self.k, "K"), (self.mol, "mol"), (self.cd, "cd"),
        ];
        for (exp, name) in dims {
            if exp == 1 { parts.push(name.to_string()); }
            else if exp != 0 { parts.push(format!("{name}^{exp}")); }
        }
        if parts.is_empty() { write!(f, "1") } else { write!(f, "{}", parts.join("⋅")) }
    }
}

/// Dimensional analysis error.
#[derive(Debug)]
pub struct DimError {
    pub message: String,
}

impl std::fmt::Display for DimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Type-checks expressions for dimensional consistency.
pub struct DimensionalAnalyzer {
    pub variable_units: HashMap<String, Unit>,
}

impl DimensionalAnalyzer {
    pub fn new() -> Self {
        Self { variable_units: HashMap::new() }
    }

    pub fn set_unit(&mut self, var: &str, unit: Unit) {
        self.variable_units.insert(var.to_string(), unit);
    }

    /// Check dimensional consistency. Returns output unit or error.
    pub fn check(&self, expr: &Expr) -> Result<Unit, DimError> {
        match expr {
            Expr::Const(_) => Ok(Unit::dimensionless()),
            Expr::Var(name) => self.variable_units.get(name).cloned()
                .ok_or_else(|| DimError { message: format!("Unknown unit for {name}") }),
            Expr::Symbol(_) => Ok(Unit::dimensionless()),
            Expr::Add(a, b) | Expr::Eq(a, b) | Expr::Lt(a, b) => {
                let ua = self.check(a)?;
                let ub = self.check(b)?;
                if ua.compatible(&ub) { Ok(ua) }
                else { Err(DimError { message: format!("Cannot add {ua} and {ub}") }) }
            }
            Expr::Mul(a, b) => {
                let ua = self.check(a)?;
                let ub = self.check(b)?;
                Ok(ua.multiply(&ub))
            }
            Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) => self.check(a),
            Expr::Inv(a) => {
                let u = self.check(a)?;
                Ok(Unit::dimensionless().divide(&u))
            }
            Expr::Pow(base, exp) => {
                let ub = self.check(base)?;
                if let Some(n) = exp.as_f64() {
                    Ok(ub.pow(n as i8))
                } else {
                    if ub.is_dimensionless() { Ok(Unit::dimensionless()) }
                    else { Err(DimError { message: "Non-const exponent with dimensioned base".into() }) }
                }
            }
            Expr::Exp(a) | Expr::Ln(a) => {
                let u = self.check(a)?;
                if u.is_dimensionless() { Ok(Unit::dimensionless()) }
                else { Err(DimError { message: format!("exp/ln argument must be dimensionless, got {u}") }) }
            }
            _ => Ok(Unit::dimensionless()),
        }
    }

    /// Filter expressions by dimensional validity.
    pub fn filter_valid<'a>(&self, candidates: &'a [Expr]) -> Vec<&'a Expr> {
        candidates.iter().filter(|e| self.check(e).is_ok()).collect()
    }
}

impl Default for DimensionalAnalyzer {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_units() {
        // F = m * a → kg * m/s² = Newton
        let mut da = DimensionalAnalyzer::new();
        da.set_unit("m", Unit::kilogram());
        da.set_unit("a", Unit::acceleration());
        let f = Expr::var("m").mul(Expr::var("a"));
        let unit = da.check(&f).unwrap();
        assert_eq!(unit, Unit::newton());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut da = DimensionalAnalyzer::new();
        da.set_unit("x", Unit::meter());
        da.set_unit("t", Unit::second());
        // x + t → error
        let expr = Expr::var("x").add(Expr::var("t"));
        assert!(da.check(&expr).is_err());
    }

    #[test]
    fn test_kinetic_energy() {
        // KE = 0.5 * m * v² → kg * (m/s)² = kg⋅m²⋅s⁻² = Joule
        let mut da = DimensionalAnalyzer::new();
        da.set_unit("m", Unit::kilogram());
        da.set_unit("v", Unit::velocity());
        let ke = Expr::num(0.5).mul(
            Expr::var("m").mul(Expr::var("v").pow(Expr::num(2.0)))
        );
        let unit = da.check(&ke).unwrap();
        assert_eq!(unit, Unit::joule());
    }

    #[test]
    fn test_spring_energy_units() {
        // Spring PE: ½kx² where k = N/m = kg⋅s⁻², x = m
        // ½kx² = kg⋅s⁻² ⋅ m² = kg⋅m²⋅s⁻² = Joule ✓
        let mut da = DimensionalAnalyzer::new();
        // k has units: kg⋅s⁻² (spring constant = N/m)
        let spring_const_unit = Unit { kg: 1, m: 0, s: -2, a: 0, k: 0, mol: 0, cd: 0 };
        da.set_unit("k", spring_const_unit);
        da.set_unit("x", Unit::meter());

        let pe = Expr::num(0.5).mul(
            Expr::var("k").mul(Expr::var("x").pow(Expr::num(2.0)))
        );
        let unit = da.check(&pe).unwrap();
        assert_eq!(unit, Unit::joule(), "½kx² should be in Joules");

        // Total spring energy: ½kx² + ½mv²
        da.set_unit("m", Unit::kilogram());
        da.set_unit("v", Unit::velocity());
        let ke = Expr::num(0.5).mul(
            Expr::var("m").mul(Expr::var("v").pow(Expr::num(2.0)))
        );
        let total = pe.add(ke);
        let total_unit = da.check(&total).unwrap();
        assert_eq!(total_unit, Unit::joule(), "½kx² + ½mv² should be in Joules");
    }
}
