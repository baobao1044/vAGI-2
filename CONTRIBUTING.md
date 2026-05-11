# Contributing

Thanks for taking a look at the project.
This repository is easiest to maintain when changes stay small, scoped, and easy to verify.

## Before opening a change

- Read `README.md` for scope and current framing.
- Read `docs/ARCHITECTURE.md` before changing shared components.
- Check `docs/STATUS.md` so expectations stay aligned with the current maturity of the repo.

## Contribution style

- Prefer the smallest change that solves the problem.
- Match existing naming, layout, and crate boundaries.
- Avoid mixing unrelated fixes into one PR.
- If a claim is added to docs, keep it specific and supportable.

## Verification expectations

Run the relevant checks before opening a PR when possible:

```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings -A clippy::should_implement_trait -A clippy::needless_range_loop -A clippy::type_complexity
```

## Documentation expectations

If you add or change any of the following, update docs in the same PR:

- architecture-sensitive behavior
- benchmark numbers
- scope or maturity claims in the README
- new crates, examples, or workflow commands

## Benchmark claims

If you mention a speedup or throughput number:

- say what was measured
- mention whether it was local, synthetic, or task-level
- avoid presenting single-machine results as universal

## Pull request guidance

A strong PR for this repo usually includes:

- one clear goal
- one focused set of code or doc changes
- a short verification note
- any assumptions or limits that still remain
