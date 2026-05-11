# Project Status

This repository is an experimental research workspace.
It is best read as a collection of implemented prototypes plus supporting documentation,
not as a claim that the full stack is finished, production-ready, or AGI-complete.

## What is implemented

- A Rust workspace with multiple crates covering ternary compute, language modeling, memory, reasoning, world modeling, training, and chat utilities.
- CI for `cargo check`, `cargo test`, and `cargo clippy` on `main`.
- Several components with in-repo tests and local benchmark notes.
- Vietnamese-oriented data and training scripts for the language-model path.

## What is currently the strongest evidence in-repo

- Source code for the crates listed in the workspace.
- Automated tests and CI workflow configuration.
- Benchmarks and performance claims that are explicitly labeled as local measurements.
- Research references explaining which external ideas influenced the implementation.

## What should still be considered experimental

- End-to-end model quality on meaningful downstream tasks.
- Benchmark portability across different CPUs, operating systems, and datasets.
- Larger-scale training runs and repeatability across seeds.
- Claims about general reasoning, scientific discovery, or autonomous-agent performance.

## Reading guide for new visitors

- Start with `README.md` for project scope.
- Read `docs/ARCHITECTURE.md` for crate boundaries and data flow.
- Read `docs/RESEARCH.md` for references and implementation inspirations.
- Read `docs/BENCHMARKS.md` for how performance numbers should be interpreted.

## Short version

The repo contains real implementation work and real experiments.
It should be judged as a serious research prototype, not as a finished AGI system.
