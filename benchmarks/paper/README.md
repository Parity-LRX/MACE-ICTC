# Paper Benchmark Artifacts

This directory archives the benchmark code, validation records, raw outputs, and
figures used by the ICTD paper draft. The top-level `benchmark_results/`
directory remains an ignored scratch/output directory; this directory is the
tracked record intended to travel with the repository.

## Layout

- `scripts/operator/`: isolated tensor-product benchmark drivers and plotting
  scripts for the e3nn, cartnn Cartesian-3j, and ICTD operator comparisons.
- `scripts/model/`: whole-model MACE-ICTD versus native MACE plotting scripts.
- `results/operator/`: isolated tensor-product CSV outputs, summaries, and logs.
- `results/model/`: selected whole-model benchmark CSVs and raw benchmark
  outputs. The `pretrained/` subdirectory contains the OFF23 exploratory records
  that were kept for provenance but are not the central paper claim.
- `figures/`: final PDF/PNG/SVG benchmark figures used by the paper draft and
  intermediate figures retained for traceability.
- `validation/`: representation-equivariance notes, parity logs, environment
  records, and warmup checks.

## Scope

The archived measurements are controlled computational benchmarks, not chemical
accuracy benchmarks. The whole-model benchmark uses synthetic fixed-edge graph
workloads to compare backend throughput. The isolated operator benchmark fixes
edge count, angular cutoffs, channel count, precision, and backend path to
measure the tensor-product operator itself.

Large regenerated binary artifacts such as AOTInductor `.pt2` packages are not
tracked here. Recreate them from the scripts when needed.
