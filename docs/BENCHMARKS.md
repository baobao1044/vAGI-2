# Benchmark Notes

The performance numbers mentioned in this repository are intended as orientation,
not as universal guarantees.

## How to interpret reported numbers

- Treat them as local measurements taken on specific hardware and software setups.
- Expect results to change with CPU model, core count, RAM speed, compiler version, build profile, dataset, and batch size.
- Use them to compare configurations inside this repo before using them to compare against other projects.

## Recommended reporting format

When adding or updating a benchmark, include:

- exact command used
- build mode (`debug` or `release`)
- crate, binary, or test name
- model size or important dimensions
- hardware summary
- dataset or synthetic input description
- whether the number is median, average, or a single run

## Example benchmark context

A claim like `150 sps` should be read as:

- a local measurement
- for a small configuration
- under the command and dataset used at the time
- not a guarantee for all CPUs or all training runs

## Suggested reproduction checklist

1. Build with `--release` unless the benchmark explicitly says otherwise.
2. Record CPU model and thread count.
3. Record dataset size and tokenizer settings.
4. Run multiple times and report the spread, not only the best number.
5. Keep the command in the commit or PR description when possible.

## Current limitations

- The repo does not yet provide a single consolidated benchmark harness for all crates.
- Some older numbers may predate later code changes.
- Some performance notes are tied to tiny experimental configurations rather than realistic production workloads.

## Maintainer note

Good benchmark notes make the project look more trustworthy than large raw numbers do.
The goal is reproducibility and honest framing.
