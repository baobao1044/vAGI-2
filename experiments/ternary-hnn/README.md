# Ternary HNN Experiment

> **Energy Conservation at 1.58 Bits: Hamiltonian Neural Networks with Ternary Weights**

Can ternary (1.58-bit) neural networks learn energy-conserving Hamiltonian dynamics?

## Quick Start

```bash
cd experiments/ternary-hnn
cargo run --release -- run-all
```

## Models

| Model | Weights | Activation | Hamiltonian? |
|-------|---------|------------|--------------|
| HNN-FP32 | float32 | SiLU | Yes |
| HNN-Ternary | ternary | SiLU | Yes |
| HNN-Adaptive | ternary | AdaptiveBasis | Yes |
| MLP-FP32 | float32 | SiLU | No |

## Physical Systems

1. **Harmonic Oscillator**: H = ½p² + ½q² (linear, integrable)
2. **Simple Pendulum**: H = ½p² - cos(q) (nonlinear, integrable)
3. **Double Pendulum**: 4D chaotic system

## Metrics

- Energy drift: max|ΔE|/|E₀|
- Trajectory MSE vs ground truth (RK45)
- Sample efficiency (MSE vs training size)
- Inference speed (µs/step)
