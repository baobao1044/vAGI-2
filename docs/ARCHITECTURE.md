# vAGI v2 — Architecture Documentation

## Crate Dependency Graph

```
vagi-runtime
    ├── vagi-reason
    │   └── vagi-core
    ├── vagi-memory
    │   ├── vagi-hdc
    │   │   └── vagi-core (via rusqlite, rayon)
    │   └── vagi-core
    └── (vagi-world, vagi-train — sibling crates)

vagi-train
    ├── vagi-core
    ├── vagi-math
    └── vagi-physics
        ├── vagi-math
        └── vagi-core

vagi-world
    └── petgraph
```

## Layer Architecture

### Layer 0: Ternary Compute (`vagi-core`)

The foundation layer provides efficient ternary {-1, 0, +1} neural network primitives.

**Key types:**

| Type | Description |
|------|-------------|
| `TernaryMatrix` | 2-bit packed storage (32 weights per u64). `pack()` from f32 via absmax thresholding, `from_ternary()` from i8 |
| `ternary_matvec()` | Runtime dispatch: `ternary_matvec_fast()` (mask-extract, 3× debug / 1.8× release speedup) → fallback to scalar |
| `BitNetLinear` | Full ternary linear layer: TernaryMatrix + per-row scale + RMSNorm |
| `AdaptiveBasis` | Learnable activation: `Σ weight_i × basis_i(x)` with identity, sin, tanh |
| `AdaptiveBlock` | BitNetLinear → AdaptiveBasis pipeline |
| `STEQuantizer` | Straight-through estimator: quantize forward, gradient pass-through |
| `STELinear` | Trainable layer with latent f32 weights, SGD update, sparsity reporting |

**Mask-extract algorithm:**

```
For each packed u64 containing 32 ternary weights:
  pos_mask = extract bits where encoding == 01 (+1)
  neg_mask = extract bits where encoding == 11 (-1)
  
  While pos_mask != 0:
    bit = pos_mask.trailing_zeros()
    accumulator += input[base + bit]
    pos_mask &= pos_mask - 1
  
  While neg_mask != 0:
    bit = neg_mask.trailing_zeros()
    accumulator -= input[base + bit]
    neg_mask &= neg_mask - 1
```

### Layer 1: Hyperdimensional Computing (`vagi-hdc`)

Binary hypervector engine for sub-millisecond memory retrieval.

**HyperVector (10,240 bits = 160 × u64):**
- `bind(a, b)` → XOR (associative, commutative, self-inverse)
- `bundle(vecs)` → majority rule (consensus)
- `permute(n)` → cyclic bit shift (sequence encoding)
- `similarity(a, b)` → 1 - 2×hamming/total_bits
- `to_bytes()` / `from_bytes()` → serialization for SQLite BLOB

**HDCEncoder:**
- Token encoding: `permute(token_id) ⊕ position_vector` → `bundle(all)`
- Embedding encoding: Random projection matrix × f32 → threshold → binary

**HDCMemory:**
- Episodes: (id, vector, metadata, timestamp, importance, access_count, surprise)
- `query_topk(query, k)`: Brute-force XOR + popcount, O(n) scan
- `query_topk_parallel()`: Rayon parallel iteration
- `sync_to_disk()` / `load_from_disk()`: SQLite round-trip (BLOB storage)
- `maintain(policy)`: Decay pruning → similarity merging → hard cap

**Forgetting formula:**

```
eff_importance = importance × exp(-λ × age_sec) × ln(2 + accesses) × (1 + surprise)
```

### Layer 2: Memory (`vagi-memory`)

**StreamingState (5-level EMA):**

Each level maintains a running state vector updated at different intervals.
On update: compress buffer via mean, then EMA: `state = (1-α)×state + α×compressed`.

| Level | Interval | α | Captures |
|-------|----------|---|----------|
| L0 | 1 | 0.30 | Current word context |
| L1 | 10 | 0.20 | Sentence-level patterns |
| L2 | 50 | 0.15 | Paragraph topics |
| L3 | 200 | 0.10 | Topic drift |
| L4 | 1000 | 0.05 | Episode-level themes |

**TwoPhaseAttention:**

```
Phase 1 (Scout): Query → HDCEncoder → HyperVector → XOR scan → top-K episodes
Phase 2 (Focus): Query × raw_buffer^T / √d → softmax → weighted sum
```

Ring buffer caps recent tokens (default 256). HDC memory stores all.

### Layer 3: Reasoning (`vagi-reason`)

**EnergyRouter:**
- Gate weights: `[n_experts × d_model]` random init
- Energy: `e_i = gate_weights[i] · input`
- Top-K selection → softmax over selected → gating weights
- Load balancing: `loss = coeff × CV²(usage_counts)`

**ExpertPool:**
- Each expert = `AdaptiveBlock` (BitNetLinear + AdaptiveBasis)
- Forward: `output = Σ weight_i × expert_i(input)` for selected experts

**PredictiveGate:**
- Linear predictor: `predicted = W × prev_state + bias` (near-identity init)
- Surprise: RMSE(actual, predicted)
- Gate: `σ(temperature × (surprise - threshold))`
- Output: `gate × actual + (1-gate) × predicted`

### Layer 4: World Model (`vagi-world`)

**CausalGraph (petgraph DiGraph):**
- Nodes: `{label, value, confidence}`
- Edges: `{strength, lag}`
- Operations: `causes()`, `effects()`, `is_dag()`, `topological_order()`
- Intervention: set node value, propagate downstream in topological order

**Planner (A* search):**
- State: current node values as HashMap
- Actions: set any node to a target value
- Cost: edge weight inverse
- Heuristic: number of unmet goals

### Layer 5: Runtime (`vagi-runtime`)

**OODA Loop:**

```
cycle(input):
  1. Observe: StreamingState.update(input)
  2. Orient:  level_state(0) as context
  3. Decide:  ExpertPool.forward(oriented) → (output, aux_loss)
  4. Gate:    PredictiveGate.forward(expert_out) → (gated, surprise, gate)
  5. Return:  (gated_output, CycleMetrics)
```

CycleMetrics: `{surprise, gate_value, aux_loss, cycle_count}`

---

## Data Flow

```
Raw Input
  ↓
StreamingState (5-level EMA update)
  ↓
Level 0 state → EnergyRouter (dot-product energy scores)
  ↓
Top-K expert selection (softmax gating)
  ↓
Expert_i.forward() for each selected expert
  ↓
Weighted sum of expert outputs
  ↓
PredictiveGate (surprise-driven gating)
  ↓
Final Output + CycleMetrics
```

---

## Test Coverage

| Crate | Tests | Key validations |
|---|:---:|---|
| vagi-core | 61 | Ternary packing roundtrip, matvec bit-exact, SIMD correctness, AdaptiveBasis convergence |
| vagi-hdc | 31 | XOR self-inverse, bundle majority, persistence roundtrip, 10K query <50ms, forgetting |
| vagi-math | 38 | Rewrite rules, calculus derivatives, equation solving, proof chains |
| vagi-physics | 17 | Unit algebra, energy conservation, symmetry detection, microworld trajectories |
| vagi-memory | 18 | EMA convergence, constant memory 100K tokens, attention output shape, ring buffer |
| vagi-reason | 16 | Routing correctness, load balancing, sparsity, surprise detection, gate passthrough |
| vagi-world | 9 | DAG validation, topological order, intervention propagation, planning |
| vagi-runtime | 9 | OODA cycle, batch run, surprise detection, expert usage tracking |
| vagi-train | 6+1 | GENESIS stages, vertical slice end-to-end |
| **Total** | **208** | |
