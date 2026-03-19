# vAGI-2 — CPU-First Ternary AGI Research Platform

> A CPU-first artificial general intelligence research platform built in Rust,
> featuring ternary weights {-1, 0, +1}, BPE tokenization, SIMD-accelerated inference,
> and a multi-crate architecture spanning symbolic math, physics simulation,
> hyperdimensional memory, sparse reasoning, and language modeling.

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Highlights

- **Ternary Weights** — All linear layers use {-1, 0, +1} weights via BitNet. Multiplications become additions, enabling CPU-efficient inference
- **CPU-First Training** — f32 batch-parallel training with AdamW + exact attention gradients, achieving **150+ sequences/second** on commodity CPUs
- **AVX2 SIMD** — Runtime-detected SIMD ternary kernels process 128 weights per instruction cycle
- **BPE Tokenizer** — Byte-Pair Encoding with 2.6x compression ratio for Vietnamese text
- **11 Crates** — Modular architecture: core engine, language model, math, physics, memory, reasoning, chat, and more

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     vagi-runtime                            │
│                   (OODA loop agent)                         │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  vagi-reason │  vagi-world  │  vagi-train  │   vagi-chat    │
│  (sparse MoE │ (causal DAG  │  (GENESIS    │  (multi-turn   │
│   routing)   │  + planner)  │   protocol)  │   dialogue)    │
├──────────────┼──────────────┼──────────────┼────────────────┤
│  vagi-memory │  vagi-math   │ vagi-physics │    vagi-lm     │
│  (streaming  │ (symbolic    │ (Hamiltonian │  (transformer  │
│   state +    │  algebra +   │  + microworld│  + BPE + fast  │
│   2-phase    │  calculus)   │  + units)    │   training)    │
│   attention) │              │              │                │
├──────────────┴──────────────┴──────────────┴────────────────┤
│              vagi-hdc                                       │
│    (10,240-bit hypervectors + SQLite memory)                │
├─────────────────────────────────────────────────────────────┤
│              vagi-core                                      │
│  (BitNet, TernaryMatrix, SIMD matvec, STE training,         │
│   AVX2 ternary kernels, AdaptiveBasis)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Crates

| Crate | Description |
|-------|-------------|
| **vagi-core** | BitNet ternary engine: 2-bit packed TernaryMatrix, mask-extract SIMD matvec, AVX2 ternary kernels, AdaptiveBasis activations, STE training |
| **vagi-lm** | Transformer language model: RoPE attention, BPE tokenizer, f32 batch-parallel training, AdamW/SignSGD optimizers, checkpoint save/load |
| **vagi-hdc** | 10,240-bit binary hypervectors, HDCMemory with SQLite persistence, top-K query, forgetting policy |
| **vagi-math** | Symbolic algebra: Expr AST, rewrite engine, calculus, equation solver, proof chains |
| **vagi-physics** | SI units, Hamiltonian Neural Networks, symmetry discovery, symbolic regression, microworlds |
| **vagi-train** | GENESIS 5-stage training protocol, JEPA embodiment, EWC regularization |
| **vagi-memory** | StreamingState (5-level EMA, O(1)/token), TwoPhaseAttention (HDC scout + softmax) |
| **vagi-reason** | Energy-based MoE routing, sparse expert compute (~95% sparsity) |
| **vagi-world** | Causal graph DAG, intervention analysis, A* goal-directed planner |
| **vagi-runtime** | OODA loop agent: Observe → Orient → Decide → Act |
| **vagi-chat** | Multi-turn ChatSession, top-k/top-p sampling, repetition penalty |

---

## Quick Start

### Prerequisites

- **Rust 1.70+** (`rustup install stable`)
- **C linker** (for SQLite):
  - Windows: MinGW-w64 or Visual Studio Build Tools
  - Linux/macOS: gcc/clang

### Build & Test

```bash
git clone https://github.com/baobao1044/vAGI-2.git
cd vAGI-2

cargo build --workspace
cargo test --workspace
```

### Download Vietnamese Training Data

```powershell
# PowerShell
powershell -ExecutionPolicy Bypass -File scripts/download_vi_data.ps1
```

This downloads Vietnamese sentence data to `data/vi_sentences.txt`.

### Train a Model

```bash
# TINY model (460K params, ~150 sps on CPU)
cargo run --example train_cpu_beast -p vagi-lm --release -- --epochs 20 --batch 16

# SMALL model (5M params, deeper training)
cargo run --example train_cpu_beast -p vagi-lm --release -- --small --epochs 20 --batch 8

# BASE model (largest)
cargo run --example train_cpu_beast -p vagi-lm --release -- --base --epochs 10 --batch 4
```

Training outputs checkpoints to `data/` directory:
- `vi_beast_best.bin` — best model by loss
- `vi_beast_final.bin` — final model
- `vi_beast_e{N}.bin` — periodic checkpoints

### Alternative Training Pipelines

```bash
# STE ternary training (quantize-aware)
cargo run --example train_vietnamese -p vagi-lm --release

# Fine-tuning from checkpoint
cargo run --example finetune_vietnamese -p vagi-lm --release
```

### Chat with a Trained Model

```bash
cargo run --example chat_vi -p vagi-lm --release
```

---

## Training Pipeline

The **CPU Beast** training pipeline (`train_cpu_beast`) combines three optimizations:

### 1. BPE Tokenizer
- Byte-Pair Encoding with configurable merge count (default: 1800 merges)
- 2.6x compression: "thầy giảng bài hay" = 22 tokens vs 58 raw bytes
- Trained on corpus and persisted to `data/bpe_merges.txt`

### 2. f32 Batch-Parallel Training
- Direct f32 matrix multiplication (no ternary quantization overhead during training)
- Rayon-parallelized batch forward passes across CPU cores
- Exact attention gradient backpropagation through softmax and Q/K/V

### 3. AdamW Optimizer
- Per-parameter adaptive learning rates
- Cosine LR schedule with warmup
- Weight decay regularization

### Performance

| Metric | TINY (d=64) | Notes |
|--------|-------------|-------|
| Speed | 150 sps | sequences per second |
| Model size | 460K params | ~1.8 MB checkpoint |
| BPE vocab | 2059 | 1800 merges |
| AVX2 | Auto-detected | 128 weights/cycle |

---

## Core Components

### Ternary Engine (`vagi-core`)

- **TernaryMatrix**: 2-bit packed storage (32 weights per u64, ~16x smaller than f32)
- **Mask-extract matvec**: AVX2-accelerated, 3x speedup over scalar
- **AdaptiveBasis**: Learnable activations with 3 basis functions (identity + sin + tanh)
- **STE Training**: Straight-through estimator for gradient flow through ternary quantization

### Language Model (`vagi-lm`)

```rust
use vagi_lm::{VagiLM, LMConfig, batch_train_step};
use vagi_lm::tokenizer_bpe::BPETokenizer;

// Create model
let config = LMConfig { vocab_size: 2059, ..LMConfig::tiny() };
let mut model = VagiLM::new(config);

// BPE tokenization
let bpe = BPETokenizer::train(&corpus, 1800);
let tokens = bpe.encode("xin chào thế giới");

// Training step
let mut adam_m = Vec::new();
let mut adam_v = Vec::new();
let (loss, acc) = batch_train_step(&mut model, &batch, &mut adam_m, &mut adam_v, step, lr);

// Generation
let generated = model.generate_fast(&tokens, 50, 0.8);
let text = bpe.decode(&generated);
```

### Symbolic Math (`vagi-math`)

```rust
use vagi_math::{Expr, calculus::differentiate};

let expr = Expr::var("x").pow(Expr::num(2.0)).add(Expr::var("x").sin());
let deriv = differentiate(&expr, "x"); // 2x + cos(x)
```

### Chat Interface (`vagi-chat`)

```rust
use vagi_chat::{ChatSession, ChatConfig};

let mut session = ChatSession::new(model, ChatConfig::default());
let response = session.send("Hello!");
```

---

## Project Structure

```
vAGI-2/
├── crates/
│   ├── vagi-core/          # Ternary engine + SIMD
│   ├── vagi-lm/            # Language model + training
│   │   ├── src/
│   │   │   ├── model.rs        # VagiLM transformer
│   │   │   ├── fast_train.rs   # f32 batch-parallel training
│   │   │   ├── tokenizer_bpe.rs# BPE tokenizer
│   │   │   ├── sign_sgd.rs     # SignSGD optimizer
│   │   │   └── checkpoint.rs   # Model save/load
│   │   └── examples/
│   │       ├── train_cpu_beast.rs    # Main training binary
│   │       ├── train_vietnamese.rs   # STE training
│   │       ├── finetune_vietnamese.rs# Fine-tuning
│   │       ├── chat_vi.rs           # Interactive chat
│   │       ├── eval_vietnamese.rs   # Model evaluation
│   │       └── convert_model.rs     # Model conversion
│   ├── vagi-hdc/           # Hyperdimensional computing
│   ├── vagi-math/          # Symbolic algebra + calculus
│   ├── vagi-physics/       # Physics simulation
│   ├── vagi-train/         # GENESIS protocol
│   ├── vagi-memory/        # Streaming state
│   ├── vagi-reason/        # Sparse MoE reasoning
│   ├── vagi-world/         # Causal world model
│   ├── vagi-runtime/       # OODA loop agent
│   └── vagi-chat/          # Chat interface
├── scripts/
│   ├── download_vi_data.ps1    # Download Vietnamese data
│   └── download_vi_large.ps1   # Download larger dataset
├── data/                   # Training data & checkpoints (gitignored)
├── docs/
│   ├── ARCHITECTURE.md
│   └── RESEARCH.md
└── experiments/
    └── ternary-hnn/        # Hamiltonian Neural Network experiments
```

---

## Design Principles

1. **CPU-First** — No GPU required. Ternary weights turn multiplications into additions
2. **Ternary Weights** — Strict {-1, 0, +1}. 2-bit packed storage, 16x memory reduction
3. **Correct by Construction** — Hamiltonian dynamics conserve energy; symbolic proofs guarantee algebraic correctness
4. **Safe Rust** — No `unsafe` except where justified with documented invariants
5. **Modular** — Each crate builds and tests independently
6. **Sparse Compute** — ~5% of experts active per input, O(1) memory retrieval

---

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) — Detailed system architecture
- [Research Foundations](docs/RESEARCH.md) — Academic references and theoretical basis

---

## License

MIT License. See [LICENSE](LICENSE) for details.
