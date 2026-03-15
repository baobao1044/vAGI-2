# vAGI v2 — Mathematical Foundations & Advanced Training Architecture

> A CPU-first artificial general intelligence research platform built in Rust,  
> combining symbolic mathematics with physics-grounded world models,  
> hyperdimensional memory, sparse mixture-of-experts reasoning,  
> and a novel training protocol (GENESIS).

[![CI](https://github.com/baobao1044/vAGI-2/actions/workflows/ci.yml/badge.svg)](https://github.com/baobao1044/vAGI-2/actions)
[![Rust](https://img.shields.io/badge/rust-1.94+-orange.svg)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-208_passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## What Is This?

vAGI v2 is a research prototype exploring a fundamentally different approach to AGI:

1. **Physics-first learning** — learn by experiencing simulated worlds, not by memorizing text
2. **Symbolic + Neural reasoning** — neural networks propose solutions, symbolic engines verify them
3. **Mathematical foundation** — build understanding from first principles (calculus, algebra, conservation laws)
4. **CPU-efficient** — designed to run on commodity hardware using BitNet ternary weights {-1, 0, +1}
5. **Hyperdimensional memory** — O(1) binary retrieval with SQLite persistence and forgetting curves
6. **Sparse compute** — energy-based MoE routing with ~95% expert sparsity

### The Core Idea

Instead of training a language model on internet text and hoping intelligence emerges, we:

```
Simulated World → Experience → Discover Patterns → Formalize as Math → Compose → Verify
         ↓              ↓              ↓                ↓              ↓         ↓
   [vagi-physics] [vagi-memory]  [vagi-math]      [vagi-train]   [vagi-reason] [vagi-world]
```

This mirrors how physicists actually develop understanding: observe phenomena, find invariants, express them as equations, then compose known laws to predict new phenomena.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     vagi-runtime                            │
│                   (OODA loop agent)                          │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  vagi-reason │  vagi-world  │  vagi-train  │                │
│  (sparse MoE │ (causal DAG  │  (GENESIS    │                │
│   + predict  │  + planner)  │   protocol)  │                │
│   gate)      │              │              │                │
├──────────────┼──────────────┼──────────────┤                │
│  vagi-memory │  vagi-math   │ vagi-physics │                │
│  (streaming  │ (symbolic    │ (Hamiltonian │                │
│   state +    │  algebra +   │  + microworlds│                │
│   2-phase    │  calculus)   │  + units)    │                │
│   attention) │              │              │                │
├──────────────┴──────────────┴──────────────┤                │
│              vagi-hdc                      │                │
│    (10,240-bit hypervectors + SQLite       │                │
│     memory + forgetting + parallel query)  │                │
├────────────────────────────────────────────┤                │
│              vagi-core                     │                │
│  (BitNet, AdaptiveBasis, TernaryMatrix,    │                │
│   SIMD matvec, STE training)              │                │
└────────────────────────────────────────────┘
```

---

## Crates

| Crate | Purpose | Source Files | Tests |
|-------|---------|:---:|:---:|
| **[vagi-core](crates/vagi-core)** | BitNet ternary engine (2-bit packed TernaryMatrix, mask-extract SIMD matvec 3×, AdaptiveBasis 3-basis activations, STE training, RMSNorm) | 5 | 61 |
| **[vagi-hdc](crates/vagi-hdc)** | 10,240-bit binary hypervectors, HDCEncoder (token + embedding), HDCMemory (SQLite, top-K query, forgetting policy, parallel rayon query) | 3 | 31 |
| **[vagi-math](crates/vagi-math)** | Symbolic algebra: Expr AST, rewrite engine (14+ rules), calculus, equation solver, proof chains, expression embedding, linear algebra | 8 | 38 |
| **[vagi-physics](crates/vagi-physics)** | SI units + dimensional analysis, Hamiltonian Neural Networks, symmetry discovery (Noether's theorem), symbolic regression, microworlds (Spring, FreeFall, Pendulum) | 5 | 17 |
| **[vagi-train](crates/vagi-train)** | GENESIS 5-stage training protocol, JEPA embodiment, EWC regularization, curriculum management, Sophia optimizer with STE | 9 | 6 |
| **[vagi-memory](crates/vagi-memory)** | StreamingState (5-level EMA, O(1)/token, 100K constant memory), TwoPhaseAttention (HDC scout + softmax focus) | 2 | 18 |
| **[vagi-reason](crates/vagi-reason)** | Energy-based MoE routing (top-K sparse compute, ~95% sparsity), per-expert AdaptiveBasis, PredictiveGate (surprise-driven gating) | 3 | 16 |
| **[vagi-world](crates/vagi-world)** | Causal graph (petgraph DAG), intervention analysis, topological reasoning, goal-directed planner (A* with heuristic) | 2 | 9 |
| **[vagi-runtime](crates/vagi-runtime)** | OODA loop agent: Observe (streaming state) → Orient (memory) → Decide (MoE + gate) → Act (output), surprise detection, expert usage tracking | 1 | 9 |
| | **Total** | **50** | **208** |

---

## Key Components

### 1. Real Ternary Engine (`vagi-core`)

Production-grade BitNet implementation:

- **TernaryMatrix**: 2-bit packed storage (32 weights/u64, ~16× smaller than f32)
- **Mask-extract matvec**: Extract pos/neg bitmasks via bitwise ops, 3× speedup over scalar (3.1ms for 768×3072 release)
- **AdaptiveBasis**: Learnable activation functions with 3 basis functions (identity + sin + tanh), 84% improvement over fixed activations
- **STE Training**: Straight-through estimator for ternary weight training (latent f32 → quantize forward → gradient pass-through)

### 2. Hyperdimensional Memory (`vagi-hdc`)

Sub-millisecond episodic memory:

- **HyperVector**: 10,240-bit binary vectors, XOR binding, Hamming distance, majority-rule bundling
- **HDCEncoder**: Token → HyperVector (permute+bundle), Embedding → HyperVector (random projection)
- **HDCMemory**: In-memory index + SQLite persistence, 10K query in 31ms
- **ForgettingPolicy**: Exponential decay × access boost × surprise, similarity merging, hard cap

### 3. Mathematical Foundation (`vagi-math`)

Dual-track reasoning system:

- **Symbolic track**: Rule-based algebraic rewriting with guaranteed correctness
- **Neural track**: Heuristic guidance for proof search (via expression embeddings)

```rust
use vagi_math::{Expr, MathReasoner};
use vagi_math::calculus::differentiate;

// Symbolic derivative: d/dx[x² + sin(x)] = 2x + cos(x)
let expr = Expr::var("x").pow(Expr::num(2.0)).add(Expr::var("x").sin());
let deriv = differentiate(&expr, "x");
```

### 4. Physics-Grounded World Model (`vagi-physics`)

- **Hamiltonian Neural Networks**: Energy-conserving dynamics by construction
- **Symplectic integration**: Leapfrog integrator preserves phase space volume
- **Dimensional analysis**: Type-checks physical expressions at the unit level
- **Symbolic regression**: Discovers formulas from data using MDL scoring

### 5. Streaming State Machine (`vagi-memory`)

Multi-scale running state for infinite context:

| Level | Scale | Update Interval | EMA α |
|-------|-------|:---:|:---:|
| L0 | Word | Every token | 0.30 |
| L1 | Sentence | Every 10 tokens | 0.20 |
| L2 | Paragraph | Every 50 tokens | 0.15 |
| L3 | Topic | Every 200 tokens | 0.10 |
| L4 | Episode | Every 1000 tokens | 0.05 |

**O(1) compute per token, constant memory** — verified with 100K tokens.

### 6. Sparse Reasoning Engine (`vagi-reason`)

- **EnergyRouter**: Dot-product energy scores → top-K selection → softmax gating
- **ExpertPool**: Per-expert AdaptiveBasis activations, ~95% sparsity (1/20 active)
- **PredictiveGate**: Predicts next state, computes surprise, gates novel vs predicted information

### 7. OODA Runtime Loop (`vagi-runtime`)

```
Observe → Orient → Decide → Act → (repeat)
   ↓         ↓        ↓       ↓
 Stream   Memory    MoE +   Output
 State    Query     Gate
```

End-to-end pipeline wiring all layers. Surprise detection, expert usage tracking, batch processing.

### 8. GENESIS Training Protocol (`vagi-train`)

| Stage | Name | What happens |
|-------|------|-------------|
| 1 | **Embody** | JEPA-style self-supervised learning on microworld trajectories |
| 2 | **Abstract** | Discover conserved quantities and symmetries (Noether's theorem) |
| 3 | **Formalize** | Neural-guided theorem proving with difficulty scaling |
| 4 | **Compose** | Solve problems requiring multiple concepts |
| 5 | **Consolidate** | Sleep phase: prune, compress (MDL), replay dreams |

### 9. Causal World Model (`vagi-world`)

- **CausalGraph**: petgraph-backed DAG with labeled nodes and weighted edges
- **Intervention**: Set a node's value, propagate downstream via causal structure
- **Planner**: A* goal-directed planning over causal graph with heuristic distance

---

## Getting Started

### Prerequisites

- Rust 1.70+ (`rustup install stable`)
- A C linker (needed for SQLite bundled compilation):
  - **Windows**: MinGW-w64 or Visual Studio Build Tools
  - **Linux/macOS**: gcc/clang (usually pre-installed)

### Build & Test

```bash
# Clone
git clone https://github.com/baobao1044/vAGI-2.git
cd vAGI-2

# Build all crates
cargo build --workspace

# Run all 208 tests
cargo test --workspace

# Run the vertical slice demo (shows training output)
cargo test -p vagi-train --test vertical_slice -- --nocapture

# Lint (same as CI)
cargo clippy --workspace -- -D warnings
```

---

## Performance Benchmarks

| Benchmark | Result |
|-----------|--------|
| TernaryMatrix 768×3072 matvec (release) | **3.1ms** (scalar: 5.7ms) |
| Mask-extract speedup (debug) | **3.07×** |
| Mask-extract speedup (release) | **1.84×** |
| HDC 10K query_topk(32) | **31ms** |
| StreamingState 100K tokens | **constant memory** |
| MoE sparsity (top-1/20) | **95%** |

---

## Project Status

| Component | Status | Tests |
|-----------|--------|:---:|
| Ternary engine (packed, SIMD, STE) | ✅ Complete | 61 |
| AdaptiveBasis (3-basis, warmup) | ✅ Complete | — |
| Hyperdimensional Computing (full) | ✅ Complete | 31 |
| Math engine (Expr, rewrite, calculus) | ✅ Complete | 38 |
| Physics (units, HNN, microworlds) | ✅ Complete | 17 |
| GENESIS training protocol | ✅ Framework | 6 |
| Streaming state (5-level EMA) | ✅ Complete | 9 |
| Two-phase attention (HDC scout) | ✅ Complete | 9 |
| Sparse MoE reasoning | ✅ Complete | 16 |
| Predictive coding gate | ✅ Complete | — |
| Causal world model | ✅ Complete | 6 |
| Goal-directed planner | ✅ Complete | 3 |
| OODA runtime loop | ✅ Complete | 9 |
| Vertical slice MVP | ✅ Complete | 1 |
| Tier 2+ microworlds | 🔲 Planned | — |
| Python bindings (PyO3) | 🔲 Planned | — |

---

## Roadmap

See [plan.md](plan.md) for the full project roadmap and architectural specification.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

See [docs/RESEARCH.md](docs/RESEARCH.md) for research foundations and references.

---

## Design Principles

1. **CPU-First** — No GPU required. All operations designed for commodity hardware
2. **Ternary Weights** — Strict {-1, 0, +1}. Multiplications become additions
3. **Correct by Construction** — Hamiltonian dynamics conserve energy structurally; symbolic proofs guarantee algebraic correctness
4. **No `unsafe`** — Safe Rust throughout, except where justified with documented invariants
5. **Incremental** — Each crate builds and tests independently
6. **MDL Compression** — Occam's razor at every level: simpler models preferred unless data demands complexity
7. **Sparse Compute** — Only ~5% of experts active per input, O(1) memory retrieval

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
