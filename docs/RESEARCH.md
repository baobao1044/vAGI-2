# vAGI v2 — Research Notes and References

> References and theoretical background for the ideas explored in this repository.

---

## 1. BitNet & Ternary Neural Networks

The core compute engine explores ternary {-1, 0, +1} weights as a CPU-friendly representation.

| Paper | Relevance |
|-------|-----------|
| [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) (Wang et al., 2023) | Foundation architecture for ternary weight matrices |
| [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024) | Motivation for low-bit models with reduced memory/compute |
| [Straight-Through Estimator](https://arxiv.org/abs/1308.3432) (Bengio et al., 2013) | Gradient estimation for quantized weights during training |

**Current implementation in this repo**: `TernaryMatrix` with 2-bit packed storage (32 weights/u64), mask-extract SIMD matvec, and `STELinear` for ternary-training experiments.

---

## 2. AdaptiveBasis — Learnable Activations

Replaces a fixed activation with a learnable basis-function combination while keeping the rest of the ternary pipeline simple.

| Paper | Relevance |
|-------|-----------|
| [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (Liu et al., 2024) | Inspiration for more expressive learned nonlinearities |
| [Adaptive Activation Functions](https://arxiv.org/abs/1906.01170) (Jagtap et al., 2020) | Parameterized activations in scientific ML |

**Current approach**: 3 basis functions (identity, sin, tanh) with per-neuron learnable weights. The repository treats this as an experimental design choice rather than a broadly validated result.

---

## 3. Hyperdimensional Computing

Binary hypervectors are used here as a compact representation for associative memory experiments.

| Paper | Relevance |
|-------|-----------|
| [Computing with High-Dimensional Vectors](https://redwood.berkeley.edu/wp-content/uploads/2020/08/kanerva2009hyperdimensional.pdf) (Kanerva, 2009) | Foundational theory for high-dimensional binary vectors |
| [A Survey on Hyperdimensional Computing](https://arxiv.org/abs/2111.06077) (Ge & Parhi, 2022) | Applications survey: classification, language, robotics |
| [Torchhd: An Open Source Python Library](https://arxiv.org/abs/2205.09208) (Heddes et al., 2022) | Implementation patterns and encoding strategies |

**Current implementation in this repo**: 10,240-bit HyperVectors with SQLite-backed episodic memory, a forgetting heuristic, and optional rayon-parallel queries.

---

## 4. Physics-Informed Neural Networks

The physics crate explores learning dynamics with conservation-aware structure where possible.

| Paper | Relevance |
|-------|-----------|
| [Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563) (Greydanus et al., 2019) | Learn energy functions that structurally conserve energy |
| [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630) (Cranmer et al., 2020) | Dynamics from Lagrangian mechanics |
| [Discovering Physical Concepts](https://arxiv.org/abs/1807.10300) (Iten et al., 2020) | Neural networks discovering conservation laws |
| [Noether Networks](https://arxiv.org/abs/2112.03321) (Alet et al., 2021) | Learnable symmetries and conserved quantities |

**Current implementation in this repo**: HNN experiments, a symplectic leapfrog integrator, symmetry-related utilities, and dimensional analysis helpers.

---

## 5. Symbolic Regression & Formula Discovery

This part of the project explores explicit mathematical-law discovery from observed data.

| Paper | Relevance |
|-------|-----------|
| [AI Feynman](https://arxiv.org/abs/1905.11481) (Udrescu & Tegmark, 2020) | Dimensional analysis + symmetries for formula discovery |
| [Discovering Symbolic Models from Deep Learning](https://arxiv.org/abs/2006.11287) (Cranmer et al., 2020) | Graph networks and symbolic regression pipelines |
| [Minimum Description Length Principle](https://arxiv.org/abs/math/0406077) (Grünwald, 2004) | MDL as Occam's razor for model selection |

**Current approach**: MCTS-style search with MDL scoring, with dimensional analysis used as a filter and neural components intended as proposal mechanisms.

---

## 6. Self-Supervised Learning (JEPA)

The training crate borrows from representation-prediction ideas rather than direct next-token-only supervision.

| Paper | Relevance |
|-------|-----------|
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (LeCun, 2022) | JEPA theory — predict in latent space |
| [I-JEPA](https://arxiv.org/abs/2301.08243) (Assran et al., 2023) | Image-based JEPA implementation |

**Current adaptation in this repo**: JEPA-inspired ideas applied to physics microworld trajectories rather than images.

---

## 7. Continual Learning

The project uses continual-learning ideas to reduce catastrophic forgetting across staged training.

| Paper | Relevance |
|-------|-----------|
| [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) (Kirkpatrick et al., 2017) | Fisher information identifies important parameters, penalizes changes |
| [Progress & Compress](https://arxiv.org/abs/1805.06370) (Schwarz et al., 2018) | Knowledge base + active column for continual learning |

**Current usage**: EWC-style regularization between GENESIS training cycles.

---

## 8. Mixture of Experts

Sparse routing is used here as a compute-efficiency experiment.

| Paper | Relevance |
|-------|-----------|
| [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2022) | Simplified MoE routing and expert balancing |
| [Mixture-of-Experts Meets Instruction Tuning](https://arxiv.org/abs/2305.14705) (Shen et al., 2023) | MoE benefits for diverse task handling |

**Current implementation**: Energy-based routing, top-K selection, softmax gating, and an auxiliary load-balancing loss.

---

## 9. Predictive Coding

The reasoning stack includes a simple surprise-gated filter inspired by predictive-processing ideas.

| Paper | Relevance |
|-------|-----------|
| [Whatever next? Predictive brains](https://doi.org/10.1017/S0140525X12000477) (Clark, 2013) | Predictive processing framework |
| [An Approximation of the Error Backpropagation Algorithm](https://doi.org/10.1162/neco.1996.8.7.1341) (Lee & Mumford, 2003) | Predictive coding as approximate backprop |

**Current implementation**: Linear predictor, surprise measured from prediction error, and sigmoid-based gating.

---

## 10. Training Optimization

| Paper | Relevance |
|-------|-----------|
| [Sophia: Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342) (Liu et al., 2023) | Diagonal Hessian optimizer inspiration |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2019) | AdamW weight decay formulation |

---

## Project-Specific Ideas

1. **AdaptiveBasis**: A learnable 3-function activation block (identity + sin + tanh) used throughout the ternary stack.
2. **HDC episodic memory**: Hyperdimensional memory backed by SQLite with a heuristic forgetting formula.
3. **Two-phase attention**: HDC-based coarse retrieval followed by focused attention over a smaller context window.
4. **GENESIS protocol**: A staged training workflow the repo uses for its experiments.
5. **Streaming state machine**: A 5-level EMA hierarchy for constant-memory context summaries.

## Scope Note

These notes explain the inspirations behind the repository. They should not be read as proof that the full stack has already achieved AGI-level behavior, broad scientific validation, or production-grade benchmarking.
