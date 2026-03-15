# AI.md — Complete Project Context for AI Agents

> **READ THIS FIRST.** This document gives you full context to work on any part of this codebase.
> After reading this, you should understand: what the project does, how every crate fits together,
> what conventions to follow, what not to break, and where to find anything.

---

## 1. Project Identity

**Name**: vAGI v2  
**Repo**: `https://github.com/baobao1044/vAGI-2`  
**Language**: Rust (edition 2021, resolver 2)  
**Target**: CPU-first AGI research platform  
**License**: MIT  

**One-sentence summary**: A modular AGI prototype that learns physics from simulated worlds using ternary neural networks, hyperdimensional memory, symbolic math, and a 5-stage training protocol called GENESIS.

---

## 2. Workspace Structure

```
BDC-AI/
├── Cargo.toml              ← workspace root (9 crates)
├── README.md               ← user-facing overview
├── plan.md                 ← full research spec (1100+ lines)
├── AI.md                   ← THIS FILE
├── docs/
│   ├── ARCHITECTURE.md     ← layer-by-layer technical docs
│   └── RESEARCH.md         ← academic references
├── .github/workflows/
│   └── ci.yml              ← CI: check + test + clippy
└── crates/
    ├── vagi-core/           ← [Layer 0] Ternary compute engine
    ├── vagi-hdc/            ← [Layer 1] Hyperdimensional computing
    ├── vagi-math/           ← [Layer 2a] Symbolic math engine
    ├── vagi-physics/        ← [Layer 2b] Physics simulation
    ├── vagi-memory/         ← [Layer 3] Streaming state + attention
    ├── vagi-reason/         ← [Layer 4] Sparse MoE reasoning
    ├── vagi-world/          ← [Layer 5a] Causal world model
    ├── vagi-train/          ← [Layer 5b] GENESIS training protocol
    └── vagi-runtime/        ← [Layer 6] OODA loop orchestrator
```

---

## 3. Dependency Graph (Build Order)

```
vagi-core          (no internal deps — foundation)
  ↓
vagi-hdc           (depends on: vagi-core)
vagi-math          (depends on: nothing — standalone)
  ↓
vagi-physics       (depends on: vagi-core, vagi-math)
vagi-memory        (depends on: vagi-core, vagi-hdc)
  ↓
vagi-reason        (depends on: vagi-core)
vagi-world         (depends on: petgraph — no internal deps)
vagi-train         (depends on: vagi-core, vagi-math, vagi-physics)
  ↓
vagi-runtime       (depends on: vagi-memory, vagi-reason)
```

**Critical rule**: Changes to `vagi-core` affect ALL downstream crates. Be extra careful with its public API.

---

## 4. Crate-by-Crate Reference

### 4.1 `vagi-core` — Ternary Compute Engine (61 tests)

**Files**: `adaptive.rs`, `bitnet.rs`, `error.rs`, `ste.rs`, `ternary.rs`

**Public API**:
```rust
// ternary.rs — 2-bit packed weight storage
TernaryMatrix::zeros(rows, cols) -> Self
TernaryMatrix::pack(weights: &[f32], rows, cols, gamma) -> Self  // f32 → ternary
TernaryMatrix::from_ternary(ternary: &[i8], rows, cols) -> Self  // i8 → packed
ternary_matvec(w: &TernaryMatrix, x: &[f32], y: &mut [f32])      // dispatches to fast
ternary_matvec_scalar(w, x, y)                                    // reference impl

// bitnet.rs — neural network layers
RMSNorm { scale: Vec<f32> }
BitNetLinear { ternary: TernaryMatrix, norm: RMSNorm }
BitNetLinear::forward(&mut self, input: &mut [f32]) -> &[f32]
BitNetBlock { layers: Vec<BitNetLinear>, d_model, n_layers }

// adaptive.rs — learnable activations
AdaptiveBasis { weights, basis_fns, d_model }  // 3 basis: identity, sin, tanh
AdaptiveBlock { linear: BitNetLinear, basis: AdaptiveBasis }
AdaptiveBlock::trimmed(in_dim, out_dim) -> Self  // creates with 3-basis default
AdaptiveBlock::forward(&mut self, buf: &mut [f32]) -> &[f32]
BasisConfig { n_basis, warmup_steps, names }
BasisScheduler  // cosine warmup for basis learning rate

// ste.rs — training with straight-through estimator
STEQuantizer::quantize_ternary(latent: &[f32]) -> (Vec<i8>, f32)  // + scale
STEQuantizer::clip_gradients(grad: &mut [f32], clip: f32)
STELinear { latent_weights: Vec<f32>, d_in, d_out }  // trainable ternary layer
STELinear::forward(&self, input: &[f32]) -> Vec<f32>
STELinear::backward_sgd(&mut self, grad: &[f32], lr: f32)

// error.rs
VagiError — unified error enum (Dimension, NotFound, Overflow, etc.)
```

**Key invariants**:
- TernaryMatrix encoding: `00 = 0, 01 = +1, 11 = -1` (2 bits per weight, 32 per u64)
- `cols_padded` is always rounded up to multiple of 32 (= `WEIGHTS_PER_U64`)
- `N_BASIS = 3` (identity, sin, tanh) — this was chosen after ablation study
- All matvec functions expect `y` to be pre-zeroed

---

### 4.2 `vagi-hdc` — Hyperdimensional Computing (31 tests)

**Files**: `vector.rs`, `encoder.rs`, `memory.rs`

**Public API**:
```rust
// vector.rs — 10,240-bit binary hypervectors
HyperVector { bits: [u64; 160] }  // 160 × 64 = 10,240 bits
HyperVector::zero() / ::random(rng)
HyperVector::bind(&self, other) -> Self                  // XOR
HyperVector::bundle(vecs: &[&HyperVector]) -> Self       // majority rule
HyperVector::permute(&self, n: usize) -> Self            // cyclic shift
HyperVector::hamming_distance(&self, other) -> u32
HyperVector::similarity(&self, other) -> f32              // 1 - 2*hamming/total
HyperVector::to_bytes(&self) -> Vec<u8> / ::from_bytes(&[u8]) -> Option<Self>
HyperVector::popcount(&self) -> u32

// encoder.rs
HDCEncoder::new(vocab_size, seed) -> Self
HDCEncoder::encode_tokens(tokens: &[usize]) -> HyperVector     // permute+bundle
HDCEncoder::encode_embedding(embedding: &[f32]) -> HyperVector  // random projection

// memory.rs — episodic memory with SQLite
Episode { id, vector, metadata, timestamp, importance, access_count, surprise_score }
MemoryConfig { max_episodes: usize }
ForgettingPolicy { decay_rate, min_importance, merge_similarity, max_episodes }
HDCMemory::open(db_path, config) / ::in_memory(config)
HDCMemory::insert(vector, metadata, importance, surprise) -> u64
HDCMemory::query_topk(query, k) -> Vec<(u64, f32)>
HDCMemory::query_topk_parallel(query, k) -> Vec<(u64, f32)>   // rayon
HDCMemory::get(id) -> Option<&Episode>
HDCMemory::touch(id)                                            // increment access_count
HDCMemory::sync_to_disk(&mut self) / load_from_disk(&mut self)
HDCMemory::maintain(&mut self, policy) -> MaintenanceReport     // prune+merge+cap
HDCMemory::effective_importance(episode, policy) -> f32
```

**Key invariants**:
- HyperVector is `[u64; 160]` — fixed size, not dynamically allocated
- Similarity is `1.0 - 2.0 * hamming / 10240.0` (range: -1.0 to 1.0, exact match = 1.0)
- Random vectors have ~50% bits set → expected similarity ≈ 0.0
- Forgetting formula: `importance × exp(-λ×age_sec) × ln(2+accesses) × (1+surprise)`
  - The `2+` (not `1+`) is critical — `ln(1)=0` would zero everything
- SQLite uses `rusqlite` with `bundled` feature (compiles SQLite from source)
- `sync_to_disk` uses `&mut self` (not `&self`) — was changed to fix transaction issues

---

### 4.3 `vagi-math` — Symbolic Math Engine (38 tests)

**Files**: `expr.rs`, `rewrite.rs`, `simplify.rs`, `calculus.rs`, `linear_algebra.rs`, `solver.rs`, `proof.rs`, `embedding.rs`, `reasoner.rs`

**Public API**:
```rust
// expr.rs — expression AST
Expr::Const(f64) | Var(String) | Symbol(String)
Expr::Add(Box, Box) | Mul(Box, Box) | Neg(Box) | Inv(Box) | Pow(Box, Box)
Expr::Sin(Box) | Cos(Box) | Exp(Box) | Ln(Box)
Expr::Derivative(Box, String) | Integral(Box, String)
Expr::Eq(Box, Box) | Lt(Box, Box)
Expr::num(v) / ::var(s) / ::sym(s)  // constructors
Expr.add(other) / .mul(other) / .pow(other) / .sin() / .cos()  // builder pattern

// rewrite.rs — rule engine
RewriteEngine::default()  // loads all built-in rules
RewriteEngine::apply_once(&self, expr) -> Option<Expr>
RewriteEngine::rewrite(&self, expr, max_steps) -> Expr  // iterate to fixpoint

// calculus.rs
differentiate(expr: &Expr, var: &str) -> Expr  // symbolic derivative
// Supports: power rule, trig, exp, ln, chain rule, product rule

// solver.rs
solve(equation: &Expr, var: &str) -> Vec<Expr>  // solve for variable

// proof.rs
Proof { steps: Vec<ProofStep> }
ProofStep { from: Expr, to: Expr, rule: String }

// reasoner.rs
MathReasoner { engine: RewriteEngine }
MathReasoner::simplify(&self, expr) -> Expr

// embedding.rs
ExprEncoder::encode(expr: &Expr) -> Vec<f32>  // tree → fixed-size vector
```

**14+ rewrite rules**: x+0→x, x×1→x, x×0→0, x^0→1, x^1→x, constant folding, sin²+cos²→1, double negation, exp(ln(x))→x, ln(exp(x))→x, power rules, etc.

---

### 4.4 `vagi-physics` — Physics Engine (17 tests)

**Files**: `units.rs`, `hamiltonian.rs`, `symmetry.rs`, `symbolic_reg.rs`, `discovery.rs`  
**Also**: `microworlds` module (within `discovery.rs` or standalone)

**Public API**:
```rust
// units.rs — dimensional analysis
Unit { kg, m, s, a, k, mol, cd }  // SI base dimension exponents (all i8)
Unit::dimensionless() / ::kilogram() / ::meter() / ::second()
Unit::newton() / ::joule() / ::velocity() / ::acceleration()
Unit::multiply(&self, other) / ::divide(&self, other) / ::pow(&self, n: i8)
DimensionalAnalyzer::new()
DimensionalAnalyzer::set_unit(var, unit)
DimensionalAnalyzer::check(expr: &Expr) -> Result<Unit, DimError>

// hamiltonian.rs — energy-conserving dynamics
HamiltonianNN { weights, d_state }
HamiltonianNN::energy(q, p) -> f32
HamiltonianNN::dynamics(q, p) -> (dq, dp)  // Hamilton's equations
HamiltonianNN::leapfrog_step(q, p, dt) -> (q', p')  // symplectic integration

// symmetry.rs
SymmetryModule::discover(trajectories) -> Vec<SymmetryInfo>
// Discovers continuous symmetries → Noether's theorem → conserved quantities

// discovery.rs — microworlds
Spring { k, m, dt } / FreeFall { g, dt } / Pendulum { L, g, dt }
trait Microworld: state(), step(), reset(), name(), state_dim(), action_dim()
AbstractionEngine::discover_invariants(...) -> Vec<DiscoveredInvariant>
```

---

### 4.5 `vagi-memory` — Streaming State + Attention (18 tests)

**Files**: `streaming.rs`, `attention.rs`

**Public API**:
```rust
// streaming.rs — 5-level EMA state
LevelConfig { update_interval, ema_alpha, label }
StreamingState::new(d_model) -> Self      // default 5 levels
StreamingState::update(&mut self, token: &[f32])
StreamingState::level_state(level) -> Option<&[f32]>
StreamingState::concat_states() -> Vec<f32>  // all 5 levels concatenated
StreamingState::reset(&mut self)

// attention.rs — two-phase attention
TwoPhaseConfig { scout_k, raw_buffer_size, d_model, vocab_size, encoder_seed }
TwoPhaseAttention::new(config) -> Self
TwoPhaseAttention::ingest(&mut self, token: &[f32], metadata: &str)
TwoPhaseAttention::scout(&self, query: &[f32]) -> Vec<(u64, f32)>  // phase 1
TwoPhaseAttention::focus(&self, query: &[f32]) -> Vec<f32>         // phase 2
TwoPhaseAttention::forward(&mut self, query: &[f32]) -> Vec<f32>   // both phases
```

**Key invariants**:
- StreamingState levels: word(1), sentence(10), paragraph(50), topic(200), episode(1000)
- Raw buffer is a ring buffer (FIFO eviction when full)
- `focus()` on empty buffer returns `vec![0.0; d_model]`

---

### 4.6 `vagi-reason` — Sparse MoE Reasoning (16 tests)

**Files**: `router.rs`, `expert.rs`, `gate.rs`

**Public API**:
```rust
// router.rs — energy-based expert routing
RouterConfig { n_experts, d_model, top_k, balance_coeff }
RoutingDecision { expert_indices, expert_weights, all_energies }
EnergyRouter::new(config) -> Self
EnergyRouter::route(&mut self, input: &[f32]) -> RoutingDecision
EnergyRouter::load_balance_loss(&self) -> f32   // CV²-based
EnergyRouter::sparsity(&self) -> f32             // 1 - top_k/n_experts

// expert.rs — MoE pool
ExpertPoolConfig { n_experts, d_model, top_k, balance_coeff }
ExpertPool::new(config) -> Self
ExpertPool::forward(&mut self, input: &[f32]) -> (Vec<f32>, f32)  // (output, aux_loss)

// gate.rs — predictive coding
PredictiveGateConfig { d_model, surprise_threshold, temperature }
PredictiveGate::new(config) -> Self
PredictiveGate::forward(&mut self, actual: &[f32]) -> (output, surprise, gate_value)
// gate = sigmoid(temperature × (surprise - threshold))
// output = gate × actual + (1-gate) × predicted
```

---

### 4.7 `vagi-world` — Causal World Model (9 tests)

**Files**: `causal.rs`, `planner.rs`

**Public API**:
```rust
// causal.rs — DAG
CausalNode { label, value, confidence }
CausalEdge { strength, lag }
CausalGraph::new() -> Self
CausalGraph::add_node(label, value) -> NodeIndex
CausalGraph::add_edge(cause, effect, strength, lag)
CausalGraph::causes(label) -> Vec<(String, f32)>
CausalGraph::effects(label) -> Vec<(String, f32)>
CausalGraph::is_dag() -> bool
CausalGraph::topological_order() -> Option<Vec<String>>
CausalGraph::intervene(&mut self, label, value) -> Vec<(String, f32)>

// planner.rs — A* search
Planner::new(graph: &CausalGraph)
Planner::plan(goals: HashMap<String, f32>) -> Option<Plan>
Plan { actions: Vec<PlannedAction>, total_cost }
```

---

### 4.8 `vagi-train` — GENESIS Training (6 tests + 1 integration)

**Files**: `genesis.rs`, `embody.rs`, `abstract_.rs`, `formalize.rs`, `compose.rs`, `consolidate.rs`, `ewc.rs`, `curriculum.rs`, `optimizer.rs`

**Public API**:
```rust
GenesisStage::Embody | Abstract | Formalize | Compose | Consolidate
GenesisScheduler::new(config) -> Self
GenesisScheduler::current_stage() -> GenesisStage
GenesisScheduler::advance() -> bool

// ewc.rs — continual learning
EWCRegularizer { fisher_diag, reference_params, lambda }
EWCRegularizer::penalty(current_params) -> f32

// optimizer.rs — Sophia-style
SophiaOptimizer { params, momentum, hessian_diag, lr, beta1, beta2, weight_decay }
```

**Integration test**: `tests/vertical_slice.rs` — end-to-end Spring → Math → Train → Predict pipeline.

---

### 4.9 `vagi-runtime` — OODA Loop (9 tests)

**Files**: `ooda.rs`

**Public API**:
```rust
OODAConfig { d_model, n_experts, top_k }
CycleMetrics { surprise, gate_value, aux_loss, cycle_count }
OODALoop::new(config) -> Self
OODALoop::cycle(&mut self, input: &[f32]) -> (Vec<f32>, CycleMetrics)
OODALoop::run_batch(&mut self, inputs: &[Vec<f32>]) -> Vec<(Vec<f32>, CycleMetrics)>
OODALoop::reset(&mut self)
```

**OODA cycle**: `Observe (StreamingState) → Orient (L0 state) → Decide (MoE + Gate) → Act (output)`

---

## 5. Build & Test Commands

```bash
# Build
cargo build --workspace

# Test (208 tests)
cargo test --workspace

# Lint (CI uses this exact command)
cargo clippy --workspace -- -D warnings -A clippy::should_implement_trait -A clippy::needless_range_loop -A clippy::type_complexity

# Test specific crate
cargo test -p vagi-core
cargo test -p vagi-hdc -- --nocapture   # with println output

# Release benchmark
cargo test -p vagi-core --release --lib -- test_benchmark --nocapture
```

**Environment on this machine (Windows)**:
```powershell
$env:Path = "C:\mingw64\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
$env:CARGO_INCREMENTAL = "0"
```

---

## 6. Conventions & Patterns

### Code Style
- **No `unsafe`** — safe Rust throughout
- **`#[inline]`** on hot-path methods (matvec inner loops)
- **Builder pattern** for Expr: `Expr::var("x").pow(Expr::num(2.0)).add(Expr::num(1.0))`
- **Config structs** with `Default` impl for every major component
- **Tests in same file** as implementation (`#[cfg(test)] mod tests { ... }`)
- **Integration tests** in `tests/` directory (vertical_slice.rs, phase3_experiments.rs)

### Naming
- Crate names: `vagi-{name}` (kebab-case in Cargo, `vagi_{name}` in Rust)
- Modules: one concept per file (e.g., `ternary.rs`, `encoder.rs`, `memory.rs`)
- Types: PascalCase, descriptive (`TernaryMatrix`, `HDCMemory`, `StreamingState`)
- Config types: `{Component}Config` (e.g., `RouterConfig`, `TwoPhaseConfig`)

### Error Handling
- `VagiError` enum in `vagi-core` for domain errors
- `rusqlite::Error` for database operations
- Most methods return `Result<T, E>` or `Option<T>`
- Internal methods that can't fail return bare types

### Testing
- Deterministic RNG: `StdRng::seed_from_u64(12345)` for reproducible tests
- All tests must pass on both debug and release builds
- **Clippy with `-D warnings`** — zero warnings policy (CI enforced)
- Timing tests have generous bounds (e.g., `< 50ms`) to account for CI variance

---

## 7. Critical Pitfalls (Read Before Editing)

### Ternary Encoding
The encoding `00=0, 01=+1, 11=-1` is load-bearing. Changing it breaks `extract_masks()`, `pack()`, `from_ternary()`, and all tests. The choice of `11` for -1 (not `10`) is deliberate — it makes the extraction bitmask logic simpler.

### Forgetting Formula
The access boost MUST be `ln(2 + access_count)`, NOT `ln(1 + access_count)`. The latter produces `ln(1) = 0` when `access_count = 0`, which zeroes out ALL effective importance and causes the maintain() function to prune everything.

### HDCMemory Persistence
`sync_to_disk()` takes `&mut self` (not `&self`). This was changed from `unchecked_transaction()` to direct `execute()` calls to fix persistence issues with SQLite `:memory:` databases in tests.

### AdaptiveBasis
N_BASIS = 3 (identity, sin, tanh). This was determined by ablation: `xabs(x)` and `relu` hurt performance. **Do not add basis functions without re-running ablation tests.**

### SiLU Decomposition Init
AdaptiveBasis weights can be initialized from SiLU decomposition: `SiLU(x) ≈ 0.5·identity + 0.25·tanh + small·sin`. This gives a warm start. The coefficients are in `BasisConfig`.

### Clippy CI
CI runs `cargo clippy --workspace -- -D warnings` with 3 allowed lints. ANY new warning = CI failure. Common culprits:
- `manual_div_ceil` — use `n.div_ceil(d)` instead of `(n + d - 1) / d`
- `bool_comparison` — use `!expr` instead of `expr == false`
- `unused_imports` / `dead_code` / `unused_variables` — must be cleaned up or `#[allow]`'d

### PowerShell Git Push
`git push` on this machine shows exit code 1 due to PowerShell stderr handling. The push actually succeeds — check for `main -> main` in output.

---

## 8. Performance Reference

| Operation | Debug | Release |
|-----------|-------|---------|
| TernaryMatrix 768×3072 matvec (scalar) | 43.8ms | 5.7ms |
| TernaryMatrix 768×3072 matvec (fast) | 14.3ms | 3.1ms |
| Speedup (fast/scalar) | 3.07× | 1.84× |
| HDC 10K query_topk(32) | 35ms | ~15ms |
| StreamingState 100K tokens | ~1s | ~200ms |
| MoE forward (8 experts, top-2, d=64) | <1ms | <0.1ms |

---

## 9. Workspace Dependencies

| Dependency | Version | Used By | Purpose |
|------------|---------|---------|---------|
| `rayon` | 1.10 | vagi-hdc | Parallel query |
| `rusqlite` | 0.31 (bundled) | vagi-hdc | SQLite persistence for HDC memory |
| `petgraph` | 0.6 | vagi-world | Causal DAG graph operations |
| `rand` | 0.8 | most crates | RNG for random init, tests |
| `thiserror` | 1.0 | most crates | Error derive macros |
| `tracing` | 0.1 | most crates | Structured logging |
| `approx` | 0.5 | tests | Float comparison in tests |
| `serde` | 1.0 | planned | Serialization (not yet widely used) |
| `criterion` | 0.5 | planned | Benchmarking framework |

**Not yet used but declared**: `wide` (SIMD), `cranelift`/`wasmtime` (JIT), `pyo3` (Python), `safetensors`, `crossbeam`, `redb`, `proptest`.

---

## 10. File Quick Index

When you need to find something, use this index:

| I need to... | Go to |
|---|---|
| Change ternary weight packing | `vagi-core/src/ternary.rs` |
| Modify matvec algorithm | `vagi-core/src/ternary.rs` (line ~250+) |
| Change basis functions | `vagi-core/src/adaptive.rs` |
| Add STE training features | `vagi-core/src/ste.rs` |
| Modify HyperVector operations | `vagi-hdc/src/vector.rs` |
| Change HDC encoding strategy | `vagi-hdc/src/encoder.rs` |
| Modify forgetting/memory | `vagi-hdc/src/memory.rs` |
| Add rewrite rules | `vagi-math/src/rewrite.rs` |
| Modify differentiation | `vagi-math/src/calculus.rs` |
| Add physical units | `vagi-physics/src/units.rs` |
| Add a microworld | `vagi-physics/src/discovery.rs` |
| Change streaming state levels | `vagi-memory/src/streaming.rs` |
| Modify attention mechanism | `vagi-memory/src/attention.rs` |
| Change expert routing | `vagi-reason/src/router.rs` |
| Modify predictive gate | `vagi-reason/src/gate.rs` |
| Change causal graph | `vagi-world/src/causal.rs` |
| Modify OODA loop | `vagi-runtime/src/ooda.rs` |
| Change training stages | `vagi-train/src/genesis.rs` |
| Run end-to-end test | `vagi-train/tests/vertical_slice.rs` |

---

## 11. Current State & What's Next

### Implemented (as of 2025-03-15)
- ✅ All 9 crates functional with 208 passing tests
- ✅ Real ternary engine with 2-bit packing and mask-extract SIMD
- ✅ HDC memory with SQLite, forgetting, parallel query
- ✅ 5-level streaming state (verified 100K tokens, constant memory)
- ✅ Two-phase attention (HDC scout + softmax focus)
- ✅ Sparse MoE with predictive coding gate
- ✅ Causal graph + goal-directed planner
- ✅ OODA runtime loop wiring everything together
- ✅ CI/CD green (check + test + clippy)

### Not Yet Implemented
- 🔲 Tier 2+ microworlds (Projectile, Orbit, NBody, etc.)
- 🔲 Python bindings via PyO3
- 🔲 JIT compilation via Cranelift
- 🔲 WASM sandbox via Wasmtime
- 🔲 SafeTensors model serialization
- 🔲 Criterion benchmarks
- 🔲 Property-based testing (proptest)
- 🔲 Real training runs (current tests use small d_model=8-64)
- 🔲 Multi-threaded OODA pipeline
- 🔲 GPU offloading (intentionally deferred — CPU-first philosophy)

---

## 12. How to Make Changes Safely

1. **Read the relevant crate's tests first** — they document expected behavior
2. **Run `cargo test -p {crate}` after every change** — catch regressions early
3. **Run `cargo clippy --workspace -- -D warnings ...` before committing** — CI will reject warnings
4. **If changing public API in vagi-core**: check all downstream crates that import it
5. **If changing HyperVector**: it's `[u64; 160]` — any size change breaks serialization
6. **If changing forgetting formula**: verify `test_prune_low_importance` still passes
7. **Commit atomically per feature** — one commit per logical change, descriptive message
8. **Use conventional commit format**: `feat(crate): description` or `fix(crate): description`
