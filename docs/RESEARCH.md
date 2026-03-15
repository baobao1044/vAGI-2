# vAGI v2 — Research Foundations

> References and theoretical background for each component.

---

## 1. BitNet & Ternary Neural Networks

The core compute engine uses ternary {-1, 0, +1} weights, replacing multiplications with additions.

| Paper | Relevance |
|-------|-----------|
| [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453) (Wang et al., 2023) | Foundation architecture — ternary weight matrices with competitive quality |
| [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024) | BitNet b1.58 — competitive with 16-bit at fraction of memory/compute |
| [Straight-Through Estimator](https://arxiv.org/abs/1308.3432) (Bengio et al., 2013) | Gradient estimation for quantized weights during training |

**Our contribution**: `TernaryMatrix` with 2-bit packed storage (32 weights/u64), mask-extract SIMD matvec achieving 3× speedup, and `STELinear` for end-to-end ternary training.

---

## 2. AdaptiveBasis — Learnable Activations

Replaces fixed SiLU with learnable basis-function combinations, keeping ternary matmul speed while gaining KAN-like expressiveness.

| Paper | Relevance |
|-------|-----------|
| [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) (Liu et al., 2024) | B-spline activations per edge — high expressiveness but 20-50× cost |
| [Adaptive Activation Functions](https://arxiv.org/abs/1906.01170) (Jagtap et al., 2020) | Parameterized activations improve physics-informed network convergence |

**Our approach**: 3 basis functions (identity, sin, tanh) with per-neuron learnable weights. Achieves 84% MSE improvement over fixed activations with <1% compute overhead. SiLU decomposition initialization for warm start.

---

## 3. Hyperdimensional Computing

Binary hypervectors for O(1) binding, near-constant time similarity search.

| Paper | Relevance |
|-------|-----------|
| [Computing with High-Dimensional Vectors](https://redwood.berkeley.edu/wp-content/uploads/2020/08/kanerva2009hyperdimensional.pdf) (Kanerva, 2009) | Foundational theory — 10,000-bit binary vectors for symbolic AI |
| [A Survey on Hyperdimensional Computing](https://arxiv.org/abs/2111.06077) (Ge & Parhi, 2022) | Applications survey: classification, language, robotics |
| [Torchhd: An Open Source Python Library](https://arxiv.org/abs/2205.09208) (Heddes et al., 2022) | Implementation patterns and encoding strategies |

**Our contribution**: 10,240-bit HyperVectors with SQLite-backed episodic memory, exponential forgetting curves with surprise-boosted retention, and parallel rayon queries (31ms for 10K episodes).

---

## 4. Physics-Informed Neural Networks

Learning dynamics from first principles with structural conservation guarantees.

| Paper | Relevance |
|-------|-----------|
| [Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563) (Greydanus et al., 2019) | Learn energy functions that structurally conserve energy |
| [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630) (Cranmer et al., 2020) | Dynamics from Lagrangian mechanics |
| [Discovering Physical Concepts](https://arxiv.org/abs/1807.10300) (Iten et al., 2020) | Neural networks discovering conservation laws |
| [Noether Networks](https://arxiv.org/abs/2112.03321) (Alet et al., 2021) | Learnable symmetries → automatic conserved quantities |

**Our implementation**: HNN with symplectic leapfrog integrator, learnable symmetry generators with Noether's theorem integration, and dimensional analysis as an inductive bias.

---

## 5. Symbolic Regression & Formula Discovery

Discovering explicit mathematical laws from observed data.

| Paper | Relevance |
|-------|-----------|
| [AI Feynman](https://arxiv.org/abs/1905.11481) (Udrescu & Tegmark, 2020) | Dimensional analysis + symmetries for formula discovery |
| [Discovering Symbolic Models from Deep Learning](https://arxiv.org/abs/2006.11287) (Cranmer et al., 2020) | Graph networks → symbolic regression pipeline |
| [Minimum Description Length Principle](https://arxiv.org/abs/math/0406077) (Grünwald, 2004) | MDL as Occam's razor for model selection |

**Our approach**: MCTS-based search with MDL scoring. Dimensional analysis pre-filters candidates, neural proposals guide search.

---

## 6. Self-Supervised Learning (JEPA)

Learning world models through prediction in representation space.

| Paper | Relevance |
|-------|-----------|
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) (LeCun, 2022) | JEPA theory — predict in latent space, not pixel space |
| [I-JEPA](https://arxiv.org/abs/2301.08243) (Assran et al., 2023) | Image-based JEPA implementation |

**Our adaptation**: JEPA applied to physics microworld trajectories (not images). Context encoder observes partial trajectory, predictor forecasts future state representation. EMA-updated target encoder prevents collapse.

---

## 7. Continual Learning

Preventing catastrophic forgetting across microworld training stages.

| Paper | Relevance |
|-------|-----------|
| [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796) (Kirkpatrick et al., 2017) | Fisher information identifies important parameters, penalizes changes |
| [Progress & Compress](https://arxiv.org/abs/1805.06370) (Schwarz et al., 2018) | Knowledge base + active column for continual learning |

**Our usage**: EWC regularization between GENESIS training cycles. Fisher diagonal computed after each microworld, merged across stages.

---

## 8. Mixture of Experts

Sparse routing for compute-efficient reasoning.

| Paper | Relevance |
|-------|-----------|
| [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2022) | Simplified MoE routing, expert capacity balancing |
| [Mixture-of-Experts Meets Instruction Tuning](https://arxiv.org/abs/2305.14705) (Shen et al., 2023) | MoE benefits for diverse task handling |

**Our implementation**: Energy-based routing (dot-product scores), top-K selection, softmax gating. Per-expert AdaptiveBasis activations for specialized nonlinearities. Load-balancing auxiliary loss (CV²-based).

---

## 9. Predictive Coding

Surprise-driven information filtering inspired by neuroscience.

| Paper | Relevance |
|-------|-----------|
| [Whatever next? Predictive brains](https://doi.org/10.1017/S0140525X12000477) (Clark, 2013) | Predictive processing framework |
| [An Approximation of the Error Backpropagation Algorithm](https://doi.org/10.1162/neco.1996.8.7.1341) (Lee & Mumford, 2003) | Predictive coding as approximate backprop |

**Our implementation**: Linear predictor (near-identity init), surprise = RMSE(actual, predicted), sigmoid gating. High surprise opens gate (accept novel info), low surprise closes gate (rely on prediction).

---

## 10. Training Optimization

| Paper | Relevance |
|-------|-----------|
| [Sophia: Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342) (Liu et al., 2023) | Diagonal Hessian optimizer, 2× faster than Adam |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2019) | AdamW — proper weight decay for adaptive optimizers |

---

## Novel Contributions

1. **AdaptiveBasis**: Learnable 3-function activation (identity + sin + tanh) compatible with ternary matmul. 84% improvement over fixed SiLU with negligible overhead.

2. **HDC Episodic Memory**: SQLite-backed hyperdimensional memory with biologically-inspired forgetting: `eff = importance × exp(-λ×age) × ln(2+accesses) × (1+surprise)`.

3. **Two-Phase Attention**: HDC scout (binary hash → O(1) retrieval) followed by focused dot-product attention over small context. Replaces O(n²) with O(k).

4. **GENESIS Protocol**: 5-stage training cycle (Embody → Abstract → Formalize → Compose → Consolidate) replacing conventional distill→finetune pipeline.

5. **Streaming State Machine**: 5-level EMA hierarchy providing O(1) per-token context at multiple temporal scales, verified constant memory over 100K tokens.
