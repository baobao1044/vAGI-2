# FAQ

## Is this an AGI system?

No.
This repository is an experimental research workspace.
It includes several components that explore ideas related to language models, memory, reasoning, symbolic math, and simulation, but it should not be presented as a validated AGI system.

## Is the repo production-ready?

Not as a whole.
Some crates are more mature than others, and the repository contains real implementation work, but the full stack is still best understood as research code.

## Are the benchmark numbers guaranteed?

No.
The benchmark numbers in this repo are local measurements and examples, not universal guarantees.
See `docs/BENCHMARKS.md` for the intended interpretation.

## Why is the project CPU-first?

The project is intentionally exploring CPU-friendly representations and workflows, especially ternary weights and compact runtime paths.
That makes experimentation easier on commodity hardware and helps test low-resource design ideas.

## Why are there so many crates?

The workspace is split into focused crates so ideas can be developed and tested with clearer boundaries.
That helps isolate changes in the ternary core, language model, memory system, reasoning stack, and training modules.

## Does every crate have the same maturity level?

No.
Some parts are better covered by tests and documentation than others.
That is why the README and status notes describe the repo as experimental rather than complete.

## What is the strongest evidence that the repo is real work and not just concept text?

The strongest evidence is the code, tests, CI workflow, and scoped documentation.
New visitors should look at the workspace crates, the test setup, and the benchmark notes before judging the larger claims.

## What should contributors avoid?

- broad or vague claims that are not supported by code or measurements
- mixing unrelated fixes into one change
- presenting single-machine benchmark results as universal
- changing shared core APIs without checking downstream crates

## Where should a new reader start?

1. `README.md`
2. `docs/STATUS.md`
3. `docs/ARCHITECTURE.md`
4. `docs/BENCHMARKS.md`
5. `CONTRIBUTING.md`
