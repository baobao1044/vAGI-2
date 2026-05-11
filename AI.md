# AI.md — Complete Project Context for AI Agents

> **READ THIS FIRST.** This document gives an AI coding agent the working context it needs
> to navigate this codebase without guessing about crate boundaries, conventions, or known pitfalls.

---

## 1. Project Identity

**Name**: vAGI v2  
**Repo**: `https://github.com/baobao1044/vAGI-2`  
**Language**: Rust (edition 2021, resolver 2)  
**Target**: Experimental CPU-first AI research workspace  
**License**: MIT  

**One-sentence summary**: A modular research prototype that combines ternary neural networks, hyperdimensional memory, symbolic math, physics-oriented modules, and a staged training protocol called GENESIS.

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
│   └── RESEARCH.md         ← references and implementation notes
└── .github/workflows/
    └── ci.yml              ← CI: check + test + clippy
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

**Critical rule**: Changes to `vagi-core` affect many downstream crates. Be extra careful with its public API.

---

## 4. Crate-by-Crate Reference

### 4.1 `vagi-core` — Ternary Compute Engine

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
AdaptiveBasis { weights, basis_fns, d_model }
AdaptiveBlock { linear: BitNetLinear, basis: AdaptiveBasis }
AdaptiveBlock::trimmed(in_dim, out_dim) -> Self
AdaptiveBlock::forward(&mut self, buf: &mut [f32]) -> &[f32]
BasisConfig { n_basis, warmup_steps, names }
BasisScheduler

// ste.rs — training with straight-through estimator
STEQuantizer::quantize_ternary(latent: &[f32]) -> (Vec<i8>, f32)
STEQuantizer::clip_gradients(grad: &mut [f32], clip: f32)
STELinear { latent_weights: Vec<f32>, d_in, d_out }
STELinear::forward(&self, input: &[f32]) -> Vec<f32>
STELinear::backward_sgd(&mut self, grad: &[f32], lr: f32)

// error.rs
VagiError — unified error enum (Dimension, NotFound, Overflow, etc.)
```

**Key invariants**:
- TernaryMatrix encoding: `00 = 0, 01 = +1, 11 = -1`
- `cols_padded` is rounded up to a multiple of 32
- `N_BASIS = 3` (identity, sin, tanh)
- All matvec functions expect `y` to be pre-zeroed

---

### 4.2 `vagi-hdc` — Hyperdimensional Computing

**Files**: `vector.rs`, `encoder.rs`, `memory.rs`

**Public API**:
```rust
HyperVector { bits: [u64; 160] }
HyperVector::zero() / ::random(rng)
HyperVector::bind(&self, other) -> Self
HyperVector::bundle(vecs: &[&HyperVector]) -> Self
HyperVector::permute(&self, n: usize) -> Self
HyperVector::hamming_distance(&self, other) -> u32
HyperVector::similarity(&self, other) -> f32
HyperVector::to_bytes(&self) -> Vec<u8> / ::from_bytes(&[u8]) -> Option<Self>
HyperVector::popcount(&self) -> u32

HDCEncoder::new(vocab_size, seed) -> Self
HDCEncoder::encode_tokens(tokens: &[usize]) -> HyperVector
HDCEncoder::encode_embedding(embedding: &[f32]) -> HyperVector

Episode { id, vector, metadata, timestamp, importance, access_count, surprise_score }
MemoryConfig { max_episodes: usize }
ForgettingPolicy { decay_rate, min_importance, merge_similarity, max_episodes }
HDCMemory::open(db_path, config) / ::in_memory(config)
HDCMemory::insert(vector, metadata, importance, surprise) -> u64
HDCMemory::query_topk(query, k) -> Vec<(u64, f32)>
HDCMemory::query_topk_parallel(query, k) -> Vec<(u64, f32)>
HDCMemory::get(id) -> Option<&Episode>
HDCMemory::touch(id)
HDCMemory::sync_to_disk(&mut self) / load_from_disk(&mut self)
HDCMemory::maintain(&mut self, policy) -> MaintenanceReport
HDCMemory::effective_importance(episode, policy) -> f32
```

**Key invariants**:
- HyperVector is `[u64; 160]`
- Similarity is `1.0 - 2.0 * hamming / 10240.0`
- Random vectors have expected similarity near 0.0
- Forgetting formula: `importance × exp(-λ×age_sec) × ln(2+accesses) × (1+surprise)`
- SQLite uses `rusqlite` with `bundled`
- `sync_to_disk` uses `&mut self`

---

## 5. Build & Test Commands

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings -A clippy::should_implement_trait -A clippy::needless_range_loop -A clippy::type_complexity
```

---

## 6. Conventions & Patterns

### Code Style
- **No `unsafe` by default**
- **`#[inline]`** on hot-path methods where justified
- **Builder pattern** for Expr
- **Config structs** with `Default` impl for major components
- **Tests in same file** as implementation where practical
- **Integration tests** in `tests/` for vertical slices

### Testing
- Deterministic RNG: `StdRng::seed_from_u64(12345)` for reproducible tests
- CI expects zero clippy warnings under the configured command
- Timing tests use loose bounds to account for CI variance

---

## 7. Critical Pitfalls (Read Before Editing)

### Ternary Encoding
The encoding `00=0, 01=+1, 11=-1` is load-bearing. Changing it breaks packing, extraction logic, and tests.

### Forgetting Formula
The access boost must be `ln(2 + access_count)`, not `ln(1 + access_count)`.

### HDCMemory Persistence
`sync_to_disk()` takes `&mut self`.

### AdaptiveBasis
`N_BASIS = 3` (identity, sin, tanh). Treat changes here as experimental and verify them carefully.

### Clippy CI
CI runs `cargo clippy --workspace -- -D warnings` with a short allowlist. New warnings will fail CI.

---

## 8. Current State

### Implemented in the repo
- Core ternary engine with packed weights and optimized matvec paths
- HDC memory with SQLite persistence and forgetting heuristics
- Streaming state and two-phase attention prototypes
- Sparse MoE routing and predictive gating
- Causal graph and goal-directed planning utilities
- OODA-style runtime loop wiring core modules together
- CI for check, test, and clippy

### Still experimental or incomplete
- Broader training validation on larger workloads
- Additional microworlds and training curricula
- Python bindings, JIT, WASM sandboxing, and other declared-but-not-central integrations
- Production benchmarking across hardware configurations

---

## 9. How to Make Changes Safely

1. Read the relevant crate tests first.
2. Run `cargo test -p {crate}` after each behavior change.
3. Run the workspace clippy command before publishing changes.
4. If changing public API in `vagi-core`, inspect downstream crates.
5. If changing HyperVector size, update serialization expectations.
6. If changing forgetting behavior, re-check related memory tests.
7. Keep commits atomic and scoped.
