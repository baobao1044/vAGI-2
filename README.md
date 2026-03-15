# vAGI v2 — Mathematical Foundations & Advanced Training Architecture

> A CPU-first artificial general intelligence research platform built in Rust,  
> combining symbolic mathematics with physics-grounded world models and  
> a novel training protocol (GENESIS).

[![CI](https://github.com/baobao1044/vAGI-2/actions/workflows/ci.yml/badge.svg)](https://github.com/baobao1044/vAGI-2/actions)
[![Rust](https://img.shields.io/badge/rust-1.94+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## What Is This?

vAGI v2 is a research prototype exploring a fundamentally different approach to AGI:

1. **Physics-first learning** — learn by experiencing simulated worlds, not by memorizing text
2. **Symbolic + Neural reasoning** — neural networks propose solutions, symbolic engines verify them
3. **Mathematical foundation** — build understanding from first principles (calculus, algebra, conservation laws)
4. **CPU-efficient** — designed to run on commodity hardware using BitNet ternary weights {-1, 0, +1}

### The Core Idea

Instead of training a language model on internet text and hoping intelligence emerges, we:

```
Simulated World → Experience → Discover Patterns → Formalize as Math → Compose → Verify
```

This mirrors how physicists actually develop understanding: observe phenomena, find invariants, express them as equations, then compose known laws to predict new phenomena.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     vagi-runtime                        │
│                   (OODA orchestrator)                   │
├──────────────┬──────────────┬──────────────┬────────────┤
│  vagi-reason │  vagi-world  │  vagi-train  │            │
│  (MoE engine)│ (causal DAG) │  (GENESIS)   │            │
├──────────────┼──────────────┼──────────────┤            │
│  vagi-memory │  vagi-math   │ vagi-physics │            │
│  (hierarchy) │ (symbolic    │ (Hamiltonian │            │
│              │  algebra)    │  + microworlds)            │
├──────────────┴──────────────┴──────────────┤            │
│              vagi-hdc                      │            │
│         (hyperdimensional computing)       │            │
├────────────────────────────────────────────┤            │
│              vagi-core                     │            │
│    (BitNet, error types, SIMD kernels)     │            │
└────────────────────────────────────────────┘
```

### Crate Dependency Graph

```
vagi-core ← vagi-hdc ← vagi-memory ← vagi-reason ← vagi-world ← vagi-runtime
                  ↑                        ↑                          ↑
              vagi-math ← vagi-physics ← vagi-train ─────────────────┘
```

---

## Crates

| Crate | Purpose | Tests |
|-------|---------|-------|
| **vagi-core** | BitNet ternary building blocks, RMSNorm, error types | 4 |
| **vagi-hdc** | 10,240-bit binary hypervectors (XOR binding, Hamming distance) | 2 |
| **vagi-math** | Symbolic algebra: Expr AST, rewrite engine (14+ rules), calculus, equation solver, proof chains, expression embedding | 38 |
| **vagi-physics** | SI units + dimensional analysis, Hamiltonian Neural Networks, symmetry discovery (Noether's theorem), symbolic regression, microworlds (Spring, FreeFall, Pendulum) | 17 |
| **vagi-train** | GENESIS 5-stage training protocol, JEPA embodiment, EWC regularization, curriculum management, Sophia optimizer with STE | 6 |
| **vagi-memory** | Hierarchical memory pyramid (stub) | — |
| **vagi-reason** | Mixture-of-Experts reasoning engine (stub) | — |
| **vagi-world** | Causal graph world model (stub) | — |
| **vagi-runtime** | OODA loop orchestrator (stub) | — |

---

## Key Components

### 1. Mathematical Foundation Layer (`vagi-math`)

A dual-track reasoning system:

- **Symbolic track**: Rule-based algebraic rewriting with guaranteed correctness
- **Neural track**: Heuristic guidance for proof search (via expression embeddings)

```rust
use vagi_math::{Expr, MathReasoner};
use vagi_math::calculus::differentiate;

// Build expression: x² + sin(x)
let expr = Expr::var("x").pow(Expr::num(2.0)).add(Expr::var("x").sin());

// Symbolic derivative: d/dx[x² + sin(x)] = 2x + cos(x)
let deriv = differentiate(&expr, "x");

// Simplify using rewrite rules
let reasoner = MathReasoner::default();
let simplified = reasoner.simplify(&deriv);
```

**Rewrite rules** include arithmetic identities (x+0→x, x*1→x), constant folding, Pythagorean identity (sin²+cos²→1), power rules, and more.

### 2. Physics-Grounded World Model (`vagi-physics`)

Learns physics from simulated experience:

- **Hamiltonian Neural Networks**: Energy-conserving dynamics by construction
- **Symplectic integration**: Leapfrog integrator preserves phase space volume
- **Dimensional analysis**: Type-checks physical expressions at the unit level
- **Symbolic regression**: Discovers formulas from data using MDL scoring

```rust
use vagi_physics::units::{DimensionalAnalyzer, Unit};
use vagi_math::Expr;

let mut da = DimensionalAnalyzer::new();
da.set_unit("m", Unit::kilogram());
da.set_unit("v", Unit::velocity());

// ½mv² → Joule ✓
let ke = Expr::num(0.5).mul(Expr::var("m").mul(Expr::var("v").pow(Expr::num(2.0))));
assert_eq!(da.check(&ke).unwrap(), Unit::joule());

// m + v → DimError! Cannot add kg and m/s
let bad = Expr::var("m").add(Expr::var("v"));
assert!(da.check(&bad).is_err());
```

### 3. GENESIS Training Protocol (`vagi-train`)

A 5-stage training cycle that replaces conventional distill→finetune:

| Stage | Name | What happens |
|-------|------|-------------|
| 1 | **Embody** | JEPA-style self-supervised learning on microworld trajectories |
| 2 | **Abstract** | Discover conserved quantities and symmetries (Noether's theorem) |
| 3 | **Formalize** | Neural-guided theorem proving with difficulty scaling |
| 4 | **Compose** | Solve problems requiring multiple concepts |
| 5 | **Consolidate** | Sleep phase: prune, compress (MDL), replay dreams |

Each cycle increases the microworld complexity tier:
- **Tier 1**: FreeFall, Spring, Pendulum
- **Tier 2**: Projectile, Collision1D (planned)
- **Tier 3+**: Orbit, NBody, Fluid, Electromagnetism (planned)

### 4. Vertical Slice (Proof of Concept)

The full pipeline works end-to-end:

```
Spring microworld (10k trajectories)
  → Math engine verifies: d/dx[½kx²] = kx ✓
  → Dimensional analysis: ½kx² + ½mv² = Joule ✓
  → Linear predictor learns dynamics in 50 epochs
  → Prediction MSE ≈ 0.0, Energy violation = 0.000045
```

The learned weight matrix `[[0.9999, 0.010], [-0.010, 1.000]]` matches the analytical symplectic Euler transition matrix for the spring system.

---

## Getting Started

### Prerequisites

- Rust 1.70+ (`rustup install stable`)
- A C linker:
  - **Windows**: MinGW-w64 or Visual Studio Build Tools
  - **Linux/macOS**: gcc/clang (usually pre-installed)

### Build & Test

```bash
# Clone
git clone https://github.com/baobao1044/vAGI-2.git
cd vAGI-2

# Build all crates
cargo build --workspace

# Run all 67 tests
cargo test --workspace

# Run the vertical slice demo (shows training output)
cargo test -p vagi-train --test vertical_slice -- --nocapture

# Lint
cargo clippy --workspace
```

---

## Roadmap

See [plan.md](plan.md) for the full project roadmap and architectural specification.

---

## Research Foundations

This project draws from several areas of research:

### BitNet & Ternary Neural Networks
- **[BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453)** (Wang et al., 2023) — Ternary weight matrices {-1, 0, +1} with comparable quality to full-precision at fraction of compute.
- **[The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)** (Ma et al., 2024) — BitNet b1.58 achieving competitive performance with 16-bit models.

### Hyperdimensional Computing
- **[Computing with High-Dimensional Vectors](https://redwood.berkeley.edu/wp-content/uploads/2020/08/kanerva2009hyperdimensional.pdf)** (Kanerva, 2009) — Binary hypervectors for symbolic AI with O(1) binding operations.
- **[A Survey on Hyperdimensional Computing](https://arxiv.org/abs/2111.06077)** (Ge & Parhi, 2022) — Comprehensive survey of HDC applications.

### Physics-Informed Neural Networks
- **[Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563)** (Greydanus et al., 2019) — Learning energy functions that structurally conserve energy.
- **[Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630)** (Cranmer et al., 2020) — Learning dynamics from Lagrangian mechanics.
- **[Discovering Physical Concepts with Neural Networks](https://arxiv.org/abs/1807.10300)** (Iten et al., 2020) — Neural networks that discover conservation laws.

### Symbolic Regression & Program Synthesis
- **[AI Feynman: A Physics-Inspired Method for Symbolic Regression](https://arxiv.org/abs/1905.11481)** (Udrescu & Tegmark, 2020) — Using dimensional analysis and symmetries to guide formula discovery.
- **[Discovering Symbolic Models from Deep Learning](https://arxiv.org/abs/2006.11287)** (Cranmer et al., 2020) — Graph networks + symbolic regression.

### Self-Supervised Learning (JEPA)
- **[A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)** (LeCun, 2022) — Joint Embedding Predictive Architecture for world models.
- **[I-JEPA](https://arxiv.org/abs/2301.08243)** (Assran et al., 2023) — Image-based JEPA implementation.

### Continual Learning
- **[Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796)** (Kirkpatrick et al., 2017) — Preventing catastrophic forgetting via Fisher information.

### Minimum Description Length
- **[Minimum Description Length Principle](https://arxiv.org/abs/math/0406077)** (Grünwald, 2004) — Occam's razor formalized: the best model compresses the data maximally.

### Training & Optimization
- **[Sophia: A Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342)** (Liu et al., 2023) — Diagonal Hessian-based optimizer, 2x faster than Adam.
- **[Straight-Through Estimator](https://arxiv.org/abs/1308.3432)** (Bengio et al., 2013) — Gradient estimation for discrete/quantized weights.

---

## Design Principles

1. **CPU-First** — No GPU required. All operations designed for commodity hardware.
2. **Ternary Weights** — Strict {-1, 0, +1}. Multiplications become additions.
3. **Correct by Construction** — Hamiltonian dynamics conserve energy structurally; symbolic proofs guarantee algebraic correctness.
4. **No `unsafe`** — Safe Rust throughout, except where justified with documented invariants.
5. **Incremental** — Each crate builds and tests independently. The vertical slice proves end-to-end viability.
6. **MDL Compression** — Occam's razor at every level: simpler models preferred unless data demands complexity.

---

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core types (BitNet, errors) | ✅ Implemented | 4 tests |
| Hyperdimensional Computing | ✅ Basic | XOR bind, Hamming distance |
| Math engine (Expr, rewrite, calculus) | ✅ Implemented | 38 tests, 14+ rewrite rules |
| Physics (units, HNN, microworlds) | ✅ Implemented | 17 tests, energy conservation verified |
| GENESIS training protocol | ✅ Framework | 5-stage scheduler, EWC, curriculum |
| Vertical slice MVP | ✅ Complete | Spring → Math → Train → Predict → Evaluate |
| Memory pyramid | 🔲 Stub | |
| Reasoning engine (MoE) | 🔲 Stub | |
| World model (causal DAG) | 🔲 Stub | |
| Runtime (OODA loop) | 🔲 Stub | |
| Tier 2+ microworlds | 🔲 Planned | Projectile, Orbit, NBody |
| Python bindings (PyO3) | 🔲 Planned | |

---

## Contributing

This is a research project. Contributions welcome in:

- Additional rewrite rules (target: 100+ rules across all categories)
- New microworlds (Tier 2-5: projectile, orbit, N-body, fluids)
- MCTS-based symbolic regression improvements
- Benchmark comparisons with standard ML baselines
- Documentation and examples

---

## License

MIT License. See [LICENSE](LICENSE) for details.
