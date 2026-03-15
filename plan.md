# vAGI v2 — SUPPLEMENT: MATHEMATICAL FOUNDATIONS & ADVANCED TRAINING ARCHITECTURE

> **Purpose**: This supplement extends the Master Prompt with two critical missing components:
> 1. A **Knowledge Foundation Layer** that gives the model deep mathematical and physical intuition
> 2. An **Advanced Training Architecture** that replaces the conventional distill→finetune pipeline with something fundamentally more capable

---

## Implementation Status (Updated 2025-03-15)

| Section | Specification | Implementation | Tests |
|---------|:---:|:---:|:---:|
| S2. Mathematical Foundation | ✅ | ✅ `vagi-math` (8 files) | 38 |
| S3. Physics World Model | ✅ | ✅ `vagi-physics` (5 files) | 17 |
| S4. GENESIS Protocol | ✅ | ✅ `vagi-train` (9 files) | 6 |
| S5. `vagi-math` Crate | ✅ | ✅ Complete | 38 |
| S6. `vagi-physics` Crate | ✅ | ✅ Complete | 17 |
| S7. Training Pipeline | ✅ | ✅ Framework + vertical slice | 1 |

**Additional implementations beyond this spec:**

| Component | Crate | Tests |
|-----------|-------|:---:|
| Real ternary engine (2-bit packed, SIMD) | `vagi-core` | 61 |
| HDC memory (SQLite, forgetting, parallel) | `vagi-hdc` | 31 |
| Streaming state (5-level EMA) | `vagi-memory` | 9 |
| Two-phase attention (HDC scout + focus) | `vagi-memory` | 9 |
| Sparse MoE + predictive gate | `vagi-reason` | 16 |
| Causal graph + planner | `vagi-world` | 9 |
| OODA runtime loop | `vagi-runtime` | 9 |
| **Total** | **9 crates, 50 files** | **208** |

---


## TABLE OF CONTENTS

S1. [The Problem: Why Standard Training Produces "Hollow" Models](#s1-the-problem)
S2. [Mathematical Foundation Layer (New Layer 2.5)](#s2-mathematical-foundation-layer)
S3. [Physics-Grounded World Model (Upgraded Layer 3)](#s3-physics-grounded-world-model)
S4. [Advanced Training Architecture: GENESIS Protocol](#s4-advanced-training-architecture-genesis-protocol)
S5. [New Crate: vagi-math](#s5-new-crate-vagi-math)
S6. [New Crate: vagi-physics](#s6-new-crate-vagi-physics)
S7. [Upgraded Training Pipeline](#s7-upgraded-training-pipeline)
S8. [Evaluation: How to Know the Model Actually "Understands"](#s8-evaluation)

---

## S1. THE PROBLEM: WHY STANDARD TRAINING PRODUCES "HOLLOW" MODELS {#s1-the-problem}

A model trained via standard distillation + self-play has a critical flaw: it learns to **mimic outputs** without internalizing the **structural constraints** that govern reality. Consider:

- It may learn that `F = ma` produces correct answers, but it doesn't know **why** — it has no concept of conservation of momentum, no notion that this equation is a consequence of translational symmetry of spacetime.
- It can pattern-match `x² + 2x + 1 = (x+1)²` but doesn't understand algebraic structure — it can't derive the factorization, only recall it.
- It predicts "object falls down" but has no internal model of gravitational fields, energy conservation, or the relationship between potential and kinetic energy.

**The core insight**: Real understanding comes from **structural constraints**, not statistical patterns. Physics works because of symmetries. Mathematics works because of axioms and logical entailment. A model that internalizes these structures will generalize far beyond its training data.

### What We Need

| Capability | What It Means | How to Achieve |
|---|---|---|
| **Mathematical intuition** | Model's internal representations respect algebraic structure (commutativity, associativity, distributivity, etc.) | Equivariant architecture + symbolic verification |
| **Physical intuition** | Model's world model conserves energy, momentum, respects symmetries | Hamiltonian/Lagrangian neural network for dynamics |
| **Formal derivation** | Model can chain logical steps to derive new results from axioms | Symbolic reasoning engine + proof search |
| **Self-discovery** | Model discovers laws from observation data without being told | Symbolic regression + compression-driven abstraction |

---

## S2. MATHEMATICAL FOUNDATION LAYER (NEW LAYER 2.5) {#s2-mathematical-foundation-layer}

This is a **new architectural component** that sits between the Reasoning Engine (Layer 2) and the World Model (Layer 3). It provides the model with an internal "algebra engine" — a structured way to manipulate mathematical objects.

### Architecture: Dual-Track Reasoning

```
Input (mathematical expression / problem)
    │
    ├──► Track 1: NEURAL (BitNet)
    │    Fast, approximate, pattern-based
    │    "This looks like a quadratic → try factoring"
    │    Latency: ~2ms
    │
    ├──► Track 2: SYMBOLIC (Rule Engine)
    │    Exact, guaranteed correct, axiom-based
    │    Apply distributive law: a(b+c) = ab + ac
    │    Latency: ~0.1ms per step, but may need many steps
    │
    └──► ARBITRATOR
         Neural track proposes candidate transformations
         Symbolic track verifies and executes them
         Result is GUARANTEED correct (symbolic) but 
         GUIDED efficiently (neural)
```

### S2.1: Expression Representation

Mathematical expressions are trees, not strings. We represent them as such:

```rust
/// Mathematical expression as an abstract syntax tree
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    // Atoms
    Const(f64),                              // numeric constant
    Var(String),                             // named variable
    Symbol(String),                          // symbolic constant (π, e, etc.)
    
    // Arithmetic
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),                          // unary negation
    Inv(Box<Expr>),                          // multiplicative inverse (1/x)
    Pow(Box<Expr>, Box<Expr>),               // exponentiation
    
    // Transcendental
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
    
    // Calculus (symbolic)
    Derivative(Box<Expr>, String),           // d(expr)/d(var)
    Integral(Box<Expr>, String),             // ∫ expr d(var)
    
    // Logic / Relations
    Eq(Box<Expr>, Box<Expr>),                // equality
    Lt(Box<Expr>, Box<Expr>),                // less than
    
    // Structured
    Vec(Vec<Expr>),                          // vector
    Matrix(Vec<Vec<Expr>>),                  // matrix
    Sum(Box<Expr>, String, Box<Expr>, Box<Expr>), // Σ_{var=lo}^{hi} expr
}
```

### S2.2: Rewrite Rule Engine

A set of verified algebraic rewrite rules that the model can apply:

```rust
/// A rewrite rule: pattern → replacement, with conditions
pub struct RewriteRule {
    pub name: String,
    pub pattern: Expr,        // template with wildcards
    pub replacement: Expr,    // what to replace with
    pub conditions: Vec<Condition>,  // when applicable
    pub category: RuleCategory,
}

pub enum RuleCategory {
    Arithmetic,        // a + 0 = a, a * 1 = a, etc.
    Algebra,           // distributive, factoring, etc.
    Trigonometry,      // sin²+cos²=1, double angle, etc.
    Calculus,          // power rule, chain rule, etc.
    LinearAlgebra,     // matrix transpose, determinant, etc.
    Logic,             // De Morgan, modus ponens, etc.
}
```

**Core rule sets** (non-exhaustive, model can discover more):

```
ARITHMETIC (20 rules):
  a + 0 → a
  a * 1 → a
  a * 0 → 0
  a + (-a) → 0
  a * (1/a) → 1  [a ≠ 0]
  a^0 → 1  [a ≠ 0]
  a^1 → a
  (a^m)^n → a^(m*n)
  ...

ALGEBRA (30 rules):
  a * (b + c) → a*b + a*c           [distributive]
  a*b + a*c → a * (b + c)           [factoring]
  (a + b)^2 → a^2 + 2*a*b + b^2    [binomial]
  a^2 - b^2 → (a+b)*(a-b)          [difference of squares]
  ...

CALCULUS (25 rules):
  d/dx [c] → 0                      [constant rule]
  d/dx [x] → 1                      [identity]
  d/dx [x^n] → n * x^(n-1)         [power rule]
  d/dx [f(g(x))] → f'(g(x)) * g'(x)  [chain rule]
  d/dx [f*g] → f'*g + f*g'          [product rule]
  d/dx [sin(x)] → cos(x)
  d/dx [cos(x)] → -sin(x)
  d/dx [e^x] → e^x
  d/dx [ln(x)] → 1/x
  ∫ x^n dx → x^(n+1)/(n+1) + C  [n ≠ -1]
  ...

TRIGONOMETRY (15 rules):
  sin²(x) + cos²(x) → 1
  sin(a+b) → sin(a)cos(b) + cos(a)sin(b)
  cos(a+b) → cos(a)cos(b) - sin(a)sin(b)
  sin(2x) → 2*sin(x)*cos(x)
  ...

LINEAR ALGEBRA (20 rules):
  (A*B)^T → B^T * A^T
  (A + B)^T → A^T + B^T
  det(A*B) → det(A) * det(B)
  A * I → A
  A * A^(-1) → I  [A invertible]
  ...
```

### S2.3: Neural-Guided Proof Search

The neural track doesn't do math directly — it guides the symbolic engine:

```rust
pub struct MathReasoner {
    /// Neural model that scores which rewrite rules to try
    rule_scorer: BitNetBlock,       // d_model → n_rules (score each rule)
    
    /// Neural model that predicts which subexpression to target
    target_selector: BitNetBlock,   // d_model → position scores
    
    /// Symbolic rule engine
    rule_engine: RewriteEngine,
    
    /// Maximum proof steps before giving up
    max_steps: usize,               // 100
}

impl MathReasoner {
    /// Attempt to transform `from` into `to` using rewrite rules
    /// Returns: proof (sequence of rule applications) or None
    pub fn prove(&self, from: &Expr, to: &Expr) -> Option<Proof>;
    
    /// Simplify an expression (neural-guided greedy search)
    pub fn simplify(&self, expr: &Expr) -> Expr;
    
    /// Solve equation for variable
    pub fn solve(&self, equation: &Expr, var: &str) -> Vec<Expr>;
    
    /// Compute symbolic derivative
    pub fn differentiate(&self, expr: &Expr, var: &str) -> Expr;
    
    /// Attempt symbolic integration
    pub fn integrate(&self, expr: &Expr, var: &str) -> Option<Expr>;
}
```

**The key insight**: The neural model learns heuristics ("when I see x² + bx + c, try completing the square"), while the symbolic engine guarantees correctness ("each step follows from axioms"). This is how human mathematicians work — intuition guides, rigor validates.

### S2.4: Expression Embedding

To connect the symbolic world with the neural world, we need to embed expressions into the model's latent space:

```rust
/// Embed a mathematical expression tree into a fixed-size vector
pub struct ExprEncoder {
    /// Encode each node type
    node_embeddings: HashMap<&'static str, Vec<f32>>,  // "Add", "Mul", etc.
    
    /// Tree-recursive encoder (BitNet)
    tree_encoder: BitNetBlock,
    
    /// Output dimension
    d_model: usize,
}

impl ExprEncoder {
    /// Recursive tree encoding:
    /// embed(Const(c)) = const_embed(c)
    /// embed(Add(a, b)) = tree_encoder([add_embed; embed(a); embed(b)])
    /// etc.
    pub fn encode(&self, expr: &Expr) -> Vec<f32>;
    
    /// Encode and cache for attention lookup
    pub fn encode_for_attention(&self, expr: &Expr) -> (Vec<f32>, HyperVector);
}
```

**Training signal**: The expression encoder is trained so that semantically equivalent expressions have similar embeddings:
```
embed(x² + 2x + 1) ≈ embed((x+1)²)
embed(sin²x + cos²x) ≈ embed(1)
embed(d/dx [x³]) ≈ embed(3x²)
```
This is achieved via contrastive learning: equivalent expressions (generated by applying rewrite rules) form positive pairs.

---

## S3. PHYSICS-GROUNDED WORLD MODEL (UPGRADED LAYER 3) {#s3-physics-grounded-world-model}

The original world model is a generic dynamics predictor. The upgraded version has **physical structure baked into its architecture**.

### S3.1: Hamiltonian Neural Network (replacing generic dynamics model)

Instead of learning arbitrary dynamics `z_next = f(z, action)`, we learn a **Hamiltonian**:

```rust
/// Hamiltonian Neural Network: learns energy function, derives dynamics via Hamilton's equations
pub struct HamiltonianNN {
    /// H(q, p) → scalar energy
    /// q = generalized positions, p = generalized momenta
    hamiltonian: BitNetBlock,  // [q; p] → scalar
    
    d_state: usize,  // dimension of q (= dimension of p)
}

impl HamiltonianNN {
    /// Compute Hamiltonian (total energy) for state (q, p)
    pub fn energy(&self, q: &[f32], p: &[f32]) -> f32;
    
    /// Compute dynamics via Hamilton's equations:
    /// dq/dt = ∂H/∂p
    /// dp/dt = -∂H/∂q
    /// This AUTOMATICALLY conserves energy (by construction)
    pub fn dynamics(&self, q: &[f32], p: &[f32]) -> (Vec<f32>, Vec<f32>);
    
    /// Integrate forward in time using symplectic integrator
    /// (preserves phase space volume = physically correct)
    pub fn integrate(&self, q: &[f32], p: &[f32], dt: f32, steps: usize) 
        -> Vec<(Vec<f32>, Vec<f32>)>;
}
```

**Why this matters**: A generic `z_next = f(z, a)` model can learn to violate energy conservation. A Hamiltonian model **cannot** — energy conservation is a structural property of the architecture, not something it needs to learn from data. This means:
- It needs far less training data (conservation is free)
- It generalizes to unseen time horizons (energy is always conserved)
- Its predictions are physically meaningful (states live on energy surfaces)

### S3.2: Symmetry-Aware Architecture

We extend the Hamiltonian with learnable symmetries:

```rust
/// Learnable symmetry group for the Hamiltonian
pub struct SymmetryModule {
    /// Generators of continuous symmetries
    /// Each generator is a vector field on state space
    generators: Vec<BitNetLinear>,  // each: state_dim → state_dim
    
    /// Learnable flag: is this symmetry active?
    /// Trained via Bayesian model selection (Noether's Razor)
    symmetry_active: Vec<f32>,     // logit per symmetry
}

impl SymmetryModule {
    /// Apply symmetry transformation to state
    pub fn transform(&self, state: &[f32], generator_idx: usize, tau: f32) -> Vec<f32>;
    
    /// Compute conserved quantity associated with generator (Noether's theorem)
    /// If generator i is a symmetry → quantity i is conserved
    pub fn conserved_quantity(&self, q: &[f32], p: &[f32], generator_idx: usize) -> f32;
    
    /// Regularization loss: encourage Hamiltonian to be invariant
    /// under active symmetries
    pub fn symmetry_loss(&self, hamiltonian: &HamiltonianNN, 
                         q: &[f32], p: &[f32]) -> f32;
}
```

**Learnable symmetries** mean the model can **discover** conservation laws:
- If it discovers translational symmetry → it has discovered conservation of momentum
- If it discovers rotational symmetry → it has discovered conservation of angular momentum
- If it discovers time-translation symmetry → it has discovered conservation of energy

These aren't hard-coded — they're learned from data, but the framework guarantees that discovered symmetries produce exact conservation laws.

### S3.3: Symbolic Regression Module

For discovering explicit mathematical laws from observed data:

```rust
/// Discover symbolic expressions that describe observed dynamics
pub struct SymbolicRegressor {
    /// Neural network that proposes expression templates
    template_proposer: BitNetBlock,  // data features → template logits
    
    /// Library of primitive functions
    primitives: Vec<Primitive>,  // +, -, *, /, sin, cos, exp, ln, pow
    
    /// MCTS-based expression search
    mcts: MCTSEngine,
    
    /// Dimensional analysis checker
    dim_checker: DimensionalAnalyzer,
}

pub enum Primitive {
    Add, Sub, Mul, Div,
    Sin, Cos, Exp, Ln,
    Pow, Sqrt, Abs,
    Const(f64),
    Var(usize),  // input variable index
}

impl SymbolicRegressor {
    /// Given data (inputs, outputs), find a symbolic expression
    /// that fits the data with minimum description length
    ///
    /// Example: given data from F = G*m1*m2/r², discover:
    /// - Neural proposes: "looks like inverse square"
    /// - MCTS searches: tries a/r², a*m1/r², a*m1*m2/r²
    /// - Dimensional analysis: only a*m1*m2/r² has correct units
    /// - Fit constant: a ≈ 6.674e-11 = G
    /// - Output: F = 6.674e-11 * m1 * m2 / r²
    pub fn discover(&self, inputs: &[Vec<f32>], outputs: &[f32],
                    variable_names: &[&str], 
                    variable_units: Option<&[Unit]>) -> Vec<DiscoveredLaw>;
    
    /// Score an expression: balances fit quality vs complexity
    /// Using Minimum Description Length (MDL)
    /// score = data_fit_bits + description_bits
    /// Lower is better (Occam's razor)
    pub fn mdl_score(&self, expr: &Expr, data: &[(Vec<f32>, f32)]) -> f64;
}

pub struct DiscoveredLaw {
    pub expression: Expr,
    pub fit_quality: f64,     // R² or similar
    pub complexity: usize,    // number of nodes in expression tree
    pub mdl_score: f64,       // total MDL score
    pub units_valid: bool,    // dimensional analysis passed
}
```

### S3.4: Dimensional Analysis System

A critical component that most ML systems lack — understanding of physical units:

```rust
/// Physical unit representation using SI base dimensions
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    pub kg: i8,    // mass
    pub m: i8,     // length
    pub s: i8,     // time
    pub a: i8,     // electric current
    pub k: i8,     // temperature
    pub mol: i8,   // amount of substance
    pub cd: i8,    // luminous intensity
}

impl Unit {
    pub fn dimensionless() -> Self;
    pub fn meter() -> Self;
    pub fn kilogram() -> Self;
    pub fn second() -> Self;
    pub fn newton() -> Self;    // kg⋅m⋅s⁻²
    pub fn joule() -> Self;     // kg⋅m²⋅s⁻²
    
    /// Check if two units are compatible (same dimensions)
    pub fn compatible(&self, other: &Self) -> bool;
    
    /// Multiply units: m * kg/s² = kg⋅m⋅s⁻²
    pub fn multiply(&self, other: &Self) -> Self;
    
    /// Divide units
    pub fn divide(&self, other: &Self) -> Self;
    
    /// Power: m² → m^2
    pub fn pow(&self, n: i8) -> Self;
}

/// Type-check a mathematical expression for dimensional consistency
pub struct DimensionalAnalyzer {
    /// Known variable → unit mappings
    variable_units: HashMap<String, Unit>,
}

impl DimensionalAnalyzer {
    /// Check if expression is dimensionally consistent
    /// Returns the output unit if valid, or an error describing the inconsistency
    pub fn check(&self, expr: &Expr) -> Result<Unit, DimError>;
    
    /// Filter candidate expressions by dimensional validity
    pub fn filter_valid(&self, candidates: &[Expr]) -> Vec<&Expr>;
}
```

**Example**: If the model proposes `F = m * v²` as a law:
- `m` has unit `kg`, `v` has unit `m/s`
- `m * v²` = `kg * m²/s²` = Joules (energy), not Newtons (force)
- Dimensional analysis REJECTS this → model must try again
- Correct: `F = m * a` → `kg * m/s²` = Newtons ✓

This single check eliminates vast swaths of nonsensical hypotheses and is one of the most powerful inductive biases in all of science.

---

## S4. ADVANCED TRAINING ARCHITECTURE: GENESIS PROTOCOL {#s4-advanced-training-architecture-genesis-protocol}

GENESIS = **G**enerative **E**pistemological **N**eural **E**volution via **S**ymbolic-**I**nductive **S**ynthesis

This replaces the conventional distill→finetune→RL pipeline with a fundamentally different approach inspired by how humans actually develop understanding.

### Core Philosophy

Humans don't learn physics by memorizing equations. They learn through:
1. **Embodied experience** → developing intuitions about how the world works
2. **Abstraction** → noticing patterns and forming general rules
3. **Formalization** → expressing rules symbolically and proving them
4. **Composition** → combining simple rules to explain complex phenomena
5. **Sleep consolidation** → compressing experience into compact knowledge

GENESIS replicates this progression.

### S4.1: The Five Stages of GENESIS

```
┌─────────────────────────────────────────────────────────────────┐
│                    GENESIS TRAINING PROTOCOL                     │
│                                                                  │
│   Stage 1: EMBODY                                                │
│   "Experience the world through simulation"                      │
│   Learn: intuitions, patterns, approximate dynamics              │
│   Method: JEPA-style self-supervised learning in microworlds     │
│   Duration: 40% of total training compute                        │
│                                                                  │
│   Stage 2: ABSTRACT                                              │
│   "Notice the patterns, form hypotheses"                         │
│   Learn: invariants, symmetries, conservation laws               │
│   Method: Symmetry discovery + symbolic regression               │
│   Duration: 20% of total training compute                        │
│                                                                  │
│   Stage 3: FORMALIZE                                             │
│   "Express knowledge as rules, verify them"                      │
│   Learn: rewrite rules, proof strategies, formal reasoning       │
│   Method: Neural-guided proof search + verification              │
│   Duration: 15% of total training compute                        │
│                                                                  │
│   Stage 4: COMPOSE                                               │
│   "Combine rules to solve novel problems"                        │
│   Learn: multi-step reasoning, problem decomposition             │
│   Method: Curriculum self-play with compositional problems       │
│   Duration: 15% of total training compute                        │
│                                                                  │
│   Stage 5: CONSOLIDATE                                           │
│   "Compress, prune, sleep on it"                                 │
│   Learn: compact representations, long-term memory               │
│   Method: MDL compression + memory consolidation + dream cycle   │
│   Duration: 10% of total training compute (runs continuously)    │
│                                                                  │
│   Then: REPEAT from Stage 1 at higher difficulty                 │
└─────────────────────────────────────────────────────────────────┘
```

### S4.2: Stage 1 — EMBODY (Self-Supervised World Learning)

The model learns by **predicting what happens next** in simulated microworlds. This is inspired by JEPA but adapted for CPU and applied to physics simulations instead of video.

#### Microworld Environments

```rust
/// A microworld is a simple physics simulation that generates (state, action, next_state) data
pub trait Microworld: Send + Sync {
    /// Get current state as a vector
    fn state(&self) -> Vec<f32>;
    
    /// Apply action and advance simulation
    fn step(&mut self, action: &[f32]) -> Vec<f32>;  // returns next state
    
    /// Reset to random initial conditions
    fn reset(&mut self, rng: &mut impl Rng);
    
    /// Name for logging
    fn name(&self) -> &str;
    
    /// State dimension
    fn state_dim(&self) -> usize;
    
    /// Action dimension
    fn action_dim(&self) -> usize;
}
```

**Microworld catalog** (ordered by complexity — curriculum learning):

```
TIER 1: Basic mechanics (1D)
├── FreeFall          — object falls under gravity, learn g
├── Spring            — harmonic oscillator, learn ω = √(k/m)
├── Pendulum          — simple pendulum, learn period vs length
├── Collision1D       — elastic collision, learn momentum conservation
└── Friction          — sliding with friction, learn energy dissipation

TIER 2: 2D mechanics
├── Projectile        — 2D projectile motion
├── Orbit2D           — two-body gravitational orbit
├── Collision2D       — 2D elastic/inelastic collisions
├── RigidBody2D       — rotation + translation
└── Fluid2D           — simple fluid cells (pressure, flow)

TIER 3: Fields and waves
├── ElectricField     — point charges, Coulomb's law
├── MagneticField     — moving charges, Lorentz force
├── Wave1D            — standing/traveling waves
├── HeatDiffusion     — temperature diffusion
└── WaveInterference  — superposition, diffraction

TIER 4: Complex systems
├── NBody             — N gravitating bodies (chaotic)
├── GasParticles      — ideal gas simulation (statistical mechanics)
├── CircuitSimulator  — resistors, capacitors, inductors
├── ChemicalReaction  — reaction kinetics, equilibrium
└── Ecosystem         — predator-prey dynamics (Lotka-Volterra)

TIER 5: Abstract mathematical worlds
├── NumberTheory      — sequences, primes, divisibility
├── GraphDynamics     — random walks, diffusion on graphs
├── GameTheory        — repeated games, Nash equilibria
├── Optimization      — gradient descent on loss landscapes
└── CellularAutomata  — emergence from simple rules
```

#### JEPA-Style Training on Microworlds

```rust
pub struct EmbodimentTrainer {
    /// The model being trained
    model: BitNetModel,
    
    /// Context encoder: observes partial state trajectory
    context_encoder: BitNetBlock,
    
    /// Target encoder: encodes the full future state (EMA-updated, not gradient)
    target_encoder: BitNetBlock,
    
    /// Predictor: predicts target representation from context representation
    predictor: BitNetBlock,
    
    /// EMA momentum for target encoder
    ema_momentum: f32,  // 0.996
}

impl EmbodimentTrainer {
    /// One training step:
    /// 1. Observe trajectory: [s0, s1, ..., sT] from microworld
    /// 2. Context = encode([s0, ..., sK]) — partial observation
    /// 3. Target = target_encode([sK+1, ..., sT]) — what we want to predict
    /// 4. Prediction = predictor(context) — model's guess
    /// 5. Loss = ||prediction - target||² (in LATENT space, not raw state)
    /// 6. Update context_encoder + predictor via gradient
    /// 7. Update target_encoder via EMA (no gradient)
    ///
    /// KEY: Loss is in REPRESENTATION space, not state space.
    /// The model learns to ignore irrelevant details (noise, exact positions)
    /// and focus on predictable structure (energy, momentum, phase).
    pub fn train_step(&mut self, trajectory: &[Vec<f32>], 
                      mask_ratio: f32) -> f32;
}
```

**Why JEPA instead of autoregressive?**
- Autoregressive models predict every detail of the next state → waste compute on noise
- JEPA predicts in representation space → learn only the predictable structure
- This is exactly what physical intuition is: knowing the essential dynamics without tracking every particle

#### Continual Learning: Elastic Weight Consolidation (EWC)

To prevent catastrophic forgetting when moving between microworlds:

```rust
pub struct EWCRegularizer {
    /// Fisher information matrix diagonal (importance of each parameter)
    fisher_diag: Vec<f32>,
    
    /// Reference parameters (from when previous task was learned)
    reference_params: Vec<f32>,
    
    /// Regularization strength
    lambda: f32,
}

impl EWCRegularizer {
    /// Compute Fisher information after learning a task
    pub fn compute_fisher(&mut self, model: &BitNetModel, data: &DataLoader);
    
    /// EWC penalty: λ/2 * Σ_i F_i * (θ_i - θ*_i)²
    /// Penalizes changing parameters that were important for previous tasks
    pub fn penalty(&self, current_params: &[f32]) -> f32;
    
    /// Merge Fisher from new task (running average)
    pub fn merge(&mut self, new_fisher: &[f32], alpha: f32);
}
```

**Why EWC?**: When the model learns spring dynamics (Tier 1), then moves to orbits (Tier 2), EWC ensures it doesn't forget how springs work. The Fisher information identifies which parameters encode "spring knowledge" and protects them.

### S4.3: Stage 2 — ABSTRACT (Symmetry Discovery)

After experiencing microworlds, the model actively searches for patterns.

```rust
pub struct AbstractionEngine {
    /// Symmetry discovery module
    symmetry_module: SymmetryModule,
    
    /// Symbolic regression for law discovery
    symbolic_regressor: SymbolicRegressor,
    
    /// Dimensional analyzer
    dim_analyzer: DimensionalAnalyzer,
    
    /// Discovered invariants database
    invariants: Vec<DiscoveredInvariant>,
}

pub struct DiscoveredInvariant {
    pub name: String,           // auto-generated or human-assigned
    pub expression: Expr,       // symbolic form
    pub microworld: String,     // where discovered
    pub conservation_error: f64, // how well it's conserved in simulations
    pub generality: f32,        // fraction of microworlds where it holds
}

impl AbstractionEngine {
    /// Given trajectory data, discover conserved quantities
    /// Method:
    /// 1. Train HamiltonianNN on trajectory
    /// 2. SymmetryModule discovers symmetry generators
    /// 3. Noether's theorem → conserved quantities
    /// 4. Symbolic regression → express conserved quantity as formula
    /// 5. Verify: does the formula hold in OTHER microworlds?
    pub fn discover_invariants(&mut self, 
                               trajectories: &[Trajectory],
                               microworld: &dyn Microworld) -> Vec<DiscoveredInvariant>;
    
    /// Test if a discovered invariant generalizes to new microworlds
    pub fn test_generality(&self, invariant: &DiscoveredInvariant,
                           test_worlds: &[Box<dyn Microworld>]) -> f32;
    
    /// Cross-domain transfer: if momentum is conserved in collisions
    /// AND in projectile motion, these are the SAME invariant
    pub fn unify_invariants(&mut self);
}
```

**The dream**: The model runs Spring simulation → discovers that `m*v` is conserved → calls this "invariant_17". Then it runs Collision1D → discovers `m1*v1 + m2*v2` is conserved → recognizes this is the SAME invariant (momentum). Then in Orbit2D → discovers `r × (m*v)` is conserved → recognizes this is the angular version. The model has **discovered** conservation of momentum without anyone telling it.

### S4.4: Stage 3 — FORMALIZE (Neural-Guided Theorem Proving)

The model learns to express its discoveries as formal rules and prove things from them.

```rust
pub struct FormalizationTrainer {
    /// The math reasoner (dual-track)
    math_reasoner: MathReasoner,
    
    /// Problem generator: creates solvable problems from discovered rules
    problem_gen: ProblemGenerator,
    
    /// Verifier: checks proofs symbolically (always correct)
    verifier: SymbolicVerifier,
}

impl FormalizationTrainer {
    /// Training loop:
    /// 1. Generator creates a problem (e.g., "simplify (x+1)² - x² - 2x")
    /// 2. Neural model proposes a strategy (e.g., "expand, collect terms")
    /// 3. Symbolic engine executes each step, verifying correctness
    /// 4. If successful: reward = 1/proof_length (shorter proofs = better)
    /// 5. If stuck: reward = 0, train on the failure
    /// 6. Curriculum: as model gets better, problems get harder
    pub fn train_step(&mut self) -> TrainResult;
}

/// Generates problems at appropriate difficulty
pub struct ProblemGenerator {
    /// Current difficulty level
    difficulty: f32,
    
    /// Problem templates by category
    templates: HashMap<RuleCategory, Vec<ProblemTemplate>>,
}

impl ProblemGenerator {
    /// Generate a problem:
    /// 1. Pick a target expression
    /// 2. Apply random rewrite rules to scramble it
    /// 3. Challenge: recover the original (or a simpler form)
    ///
    /// Difficulty controls:
    /// - Number of scramble steps
    /// - Number of variables
    /// - Which rule categories are involved
    /// - Whether the problem requires multi-step reasoning
    pub fn generate(&self, rng: &mut impl Rng) -> Problem;
    
    /// Adjust difficulty based on success rate
    /// Target: 50-70% success rate (zone of proximal development)
    pub fn adjust_difficulty(&mut self, recent_success_rate: f32);
}
```

### S4.5: Stage 4 — COMPOSE (Compositional Problem Solving)

The model learns to combine multiple discovered rules to solve novel, multi-step problems.

```rust
pub struct CompositionTrainer {
    /// All previous components
    math_reasoner: MathReasoner,
    world_model: HamiltonianNN,
    causal_graph: CausalGraph,
    
    /// Compositional problem generator
    comp_gen: CompositionalGenerator,
}

impl CompositionalGenerator {
    /// Generate problems that require COMBINING knowledge:
    /// 
    /// Example (physics + calculus):
    ///   "An object is thrown upward at 20 m/s. When does it reach max height?"
    ///   Requires: kinematics (v = v0 - g*t) + calculus (set dh/dt = 0)
    ///
    /// Example (algebra + physics):
    ///   "Two objects collide. m1=3kg at v1=4m/s, m2=2kg at v2=-1m/s. 
    ///    Find final velocities for elastic collision."
    ///   Requires: momentum conservation + energy conservation → 2 equations, 2 unknowns
    ///
    /// Example (multi-step reasoning):
    ///   "Prove that the period of a simple pendulum is T = 2π√(L/g)"
    ///   Requires: Newton's law → differential equation → small angle approximation
    ///             → solve ODE → extract period
    pub fn generate(&self, n_concepts: usize, rng: &mut impl Rng) -> CompositionalProblem;
}
```

### S4.6: Stage 5 — CONSOLIDATE (Sleep & Compress)

Runs continuously in the background, especially after each stage transition:

```rust
pub struct ConsolidationEngine {
    /// Memory consolidation (from original architecture)
    memory: HDCMemory,
    forgetting: ForgettingPolicy,
    
    /// Knowledge compression
    compressor: KnowledgeCompressor,
    
    /// Dream replay
    dreamer: DreamEngine,
}

pub struct KnowledgeCompressor {
    /// MDL-based model pruning
    /// Remove any component where: cost_of_keeping > utility
    pub fn prune_model(&self, model: &mut BitNetModel) -> PruneReport;
    
    /// Compress discovered rules into more compact forms
    /// E.g., if the model has separate rules for "F=ma in 1D" and "F=ma in 2D",
    /// compress into "F=ma in any dimension" (vector form)
    pub fn compress_rules(&self, rules: &mut Vec<RewriteRule>) -> usize;
    
    /// Distill microworld knowledge into the Hamiltonian
    /// After experiencing 20 microworlds, the Hamiltonian should be
    /// a general-purpose physics simulator, not 20 separate models
    pub fn distill_physics(&self, hamiltonian: &mut HamiltonianNN,
                           microworld_data: &[MicroworldData]);
}

pub struct DreamEngine {
    /// Replay past experiences in random order
    /// This helps the model find cross-domain connections
    pub fn dream_cycle(&self, model: &mut BitNetModel, 
                       memory: &HDCMemory,
                       duration: Duration) -> DreamReport;
}
```

### S4.7: Curriculum Scheduling

The GENESIS protocol doesn't just run once — it cycles with increasing difficulty:

```rust
pub struct GenesisScheduler {
    pub current_cycle: usize,
    pub current_stage: GenesisStage,
    pub tier_progression: Vec<usize>,  // which microworld tiers unlocked
}

pub enum GenesisStage {
    Embody { tier: usize, progress: f32 },
    Abstract { n_invariants_target: usize },
    Formalize { difficulty: f32 },
    Compose { n_concepts: usize },
    Consolidate { duration: Duration },
}

impl GenesisScheduler {
    /// Determine next training action based on progress
    pub fn next(&mut self, metrics: &TrainingMetrics) -> TrainingAction;
    
    /// Progress criteria for stage transitions:
    /// Embody → Abstract: prediction loss plateaus AND model has enough experience
    /// Abstract → Formalize: ≥ N invariants discovered with generality > 0.7
    /// Formalize → Compose: proof success rate > 60% at current difficulty
    /// Compose → Consolidate: composition success rate > 50%
    /// Consolidate → Embody (next tier): MDL score improved
    pub fn should_advance(&self, metrics: &TrainingMetrics) -> bool;
}
```

---

## S5. NEW CRATE: vagi-math {#s5-new-crate-vagi-math}

```
crates/vagi-math/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── expr.rs           # Expr enum, parsing, display
    ├── rewrite.rs         # RewriteRule, RewriteEngine, pattern matching
    ├── simplify.rs        # Simplification strategies
    ├── calculus.rs         # Symbolic differentiation, integration
    ├── linear_algebra.rs  # Symbolic matrix operations
    ├── solver.rs          # Equation solving
    ├── proof.rs           # Proof representation, verification
    ├── embedding.rs       # ExprEncoder (neural embedding of expressions)
    └── reasoner.rs        # MathReasoner (neural-guided proof search)
```

**Key dependency**: `vagi-core` (for BitNet blocks used in neural components)

---

## S6. NEW CRATE: vagi-physics {#s6-new-crate-vagi-physics}

```
crates/vagi-physics/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── units.rs           # Unit system, dimensional analysis
    ├── hamiltonian.rs     # HamiltonianNN, symplectic integrator
    ├── symmetry.rs        # SymmetryModule, Noether's theorem
    ├── symbolic_reg.rs    # SymbolicRegressor, MCTS expression search
    ├── microworlds/
    │   ├── mod.rs         # Microworld trait
    │   ├── mechanics.rs   # FreeFall, Spring, Pendulum, etc.
    │   ├── fields.rs      # ElectricField, MagneticField, etc.
    │   ├── complex.rs     # NBody, GasParticles, etc.
    │   └── abstract.rs    # NumberTheory, GraphDynamics, etc.
    └── discovery.rs       # AbstractionEngine, invariant discovery
```

**Key dependencies**: `vagi-core`, `vagi-math`, `vagi-hdc`

---

## S7. UPGRADED TRAINING PIPELINE {#s7-upgraded-training-pipeline}

The original Phase 6 training pipeline is replaced with:

```
crates/vagi-train/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── genesis.rs          # GenesisScheduler, training orchestrator
    ├── embody.rs           # EmbodimentTrainer (JEPA on microworlds)
    ├── abstract_.rs        # AbstractionEngine training
    ├── formalize.rs        # FormalizationTrainer
    ├── compose.rs          # CompositionTrainer
    ├── consolidate.rs      # ConsolidationEngine, DreamEngine
    ├── ewc.rs              # Elastic Weight Consolidation
    ├── curriculum.rs       # CurriculumManager, difficulty scaling
    ├── optimizer.rs        # Sophia optimizer + STE (unchanged)
    └── distill.rs          # Optional: bootstrap from teacher (still available but secondary)
```

### Training Compute Budget (for 100M param model on 8-core CPU)

```
GENESIS Cycle 1 (Tier 1 microworlds):
  Embody:      ~3 days   (1M trajectories, 100 steps each)
  Abstract:    ~1 day    (symmetry search + symbolic regression)
  Formalize:   ~2 days   (10K proof problems)
  Compose:     ~2 days   (5K compositional problems)
  Consolidate: ~1 day    (compression + dream replay)
  
  Total Cycle 1: ~9 days

GENESIS Cycle 2 (Tier 2 microworlds):
  ~12 days (harder problems, more data needed)

GENESIS Cycle 3-5: 
  ~15-20 days each

TOTAL: ~2-3 months for 5 cycles on 8-core CPU
```

This is slower than pure distillation (~1 week), but the model that comes out is fundamentally different: it has **discovered** physics, not memorized it.

### Hybrid Approach: Distillation + GENESIS

For practical purposes, you can combine both:

```
Phase A: Quick distillation from teacher (1 week)
  → Gets basic language/reasoning capability
  → 70-80% of "standard" model quality

Phase B: GENESIS training (2-3 months)
  → Adds deep mathematical/physical understanding
  → Emergent capabilities not present in teacher
  → Self-discovered conservation laws and symmetries
```

The distilled knowledge provides a warm start, and GENESIS adds the deep understanding on top.

---

## S8. EVALUATION: HOW TO KNOW THE MODEL ACTUALLY "UNDERSTANDS" {#s8-evaluation}

### S8.1: Mathematical Understanding Tests

```rust
pub struct MathEvalSuite {
    /// Level 1: Computation (can it get the right answer?)
    /// "What is 3x² + 2x + 1 when x = 5?" → 86
    computation: Vec<(Expr, HashMap<String, f64>, f64)>,
    
    /// Level 2: Manipulation (can it transform expressions correctly?)
    /// "Factor x² - 9" → (x+3)(x-3)
    manipulation: Vec<(Expr, Expr)>,
    
    /// Level 3: Derivation (can it prove things?)
    /// "Prove that d/dx[x^n] = n*x^(n-1) using the limit definition"
    derivation: Vec<(String, Proof)>,
    
    /// Level 4: Discovery (can it find patterns?)
    /// Given: f(1)=1, f(2)=4, f(3)=9, f(4)=16. Find f(n).
    discovery: Vec<(Vec<(f64, f64)>, Expr)>,
    
    /// Level 5: Composition (can it combine concepts?)
    /// "Find the maximum volume of a box with surface area 100"
    /// Requires: geometry + calculus + algebra
    composition: Vec<(String, f64)>,
}
```

### S8.2: Physical Understanding Tests

```rust
pub struct PhysicsEvalSuite {
    /// Conservation tests: does the model's world model conserve energy/momentum?
    /// Run simulation → check if conserved quantities stay constant
    conservation_tests: Vec<ConservationTest>,
    
    /// Generalization: trained on Spring + FreeFall, test on Pendulum
    /// Can it predict pendulum dynamics without training on pendulums?
    generalization_tests: Vec<GeneralizationTest>,
    
    /// Counterfactual: "What if gravity were twice as strong?"
    /// Model should predict: periods halved, terminal velocity changed, etc.
    counterfactual_tests: Vec<CounterfactualTest>,
    
    /// Symbolic accuracy: does the model recover correct formulas?
    /// Train on FreeFall data → should discover s = ½gt²
    law_discovery_tests: Vec<LawDiscoveryTest>,
}
```

### S8.3: Emergent Capability Tests

These test for capabilities that were **never explicitly trained**:

```
Test 1: TRANSFER
  Train: Spring (1D harmonic oscillator)
  Test: LC Circuit (electrical harmonic oscillator)
  Expected: Model recognizes same mathematical structure
  Pass criterion: prediction error < 2× the Spring prediction error

Test 2: ANALOGY  
  Train: Gravitational orbits
  Test: Charged particle in magnetic field (also circular motion)
  Expected: Model transfers orbital dynamics knowledge
  
Test 3: NOVEL COMPOSITION
  Train: Calculus rules + kinematics equations (separately)
  Test: "Derive the range equation for projectile motion"
  Expected: Model chains v=v0+at → integrate → apply boundary conditions → solve
  
Test 4: COUNTERFACTUAL PHYSICS
  Test: "In a universe where F = m*a², what would orbits look like?"
  Expected: Model simulates with modified dynamics, predicts non-elliptical orbits
  
Test 5: MATHEMATICAL CREATIVITY
  Test: Given only + and * operations, can the model discover/derive 
        the concept of exponentiation as repeated multiplication?
```

---

## UPDATED DEPENDENCY MATRIX

```
vagi-core    → (no internal deps)
vagi-hdc     → vagi-core
vagi-math    → vagi-core                     [NEW]
vagi-physics → vagi-core + vagi-math + vagi-hdc  [NEW]
vagi-memory  → vagi-core + vagi-hdc
vagi-reason  → vagi-core + vagi-memory + vagi-math
vagi-world   → vagi-core + vagi-reason + vagi-physics
vagi-runtime → ALL crates
vagi-train   → ALL crates
```

## UPDATED PHASE TIMELINE

```
Phase 1: Rust Kernel + BitNet          (3-5 weeks)  [unchanged]
Phase 2: HDC Engine                    (2-3 weeks)  [unchanged]
Phase 2.5: Math Foundation (vagi-math) (3-4 weeks)  [NEW]
Phase 3: Memory Pyramid               (2-3 weeks)  [unchanged]
Phase 3.5: Physics Engine (vagi-physics) (4-6 weeks) [NEW]
Phase 4: Reasoning Engine              (3-4 weeks)  [updated: integrates math]
Phase 5: World Model                   (4-6 weeks)  [updated: Hamiltonian + symmetry]
Phase 6: Integration + GENESIS         (4-6 weeks)  [updated: GENESIS replaces basic pipeline]

GENESIS Training: 2-3 months continuous

Total: ~6-8 months development + 2-3 months training
```

---

**End of Supplement.**