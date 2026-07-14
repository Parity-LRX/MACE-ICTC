# Official ICTP isolated-operator benchmark

This record adds the official implementation of Zaverkin et al.'s irreducible
Cartesian tensor product to the paper's isolated operator comparison.

## Scope

- GPU: NVIDIA GeForce RTX 4090 D (24 GB)
- PyTorch: 2.7.1+cu128
- e3nn: 0.5.9
- ICTP upstream commit: f40592a5687ec1d03219300ee557b2660f7d0369
- dtype: fp32
- hidden channels: 64
- directed edges: 100,000
- warmup/measured calls: 20/50
- configurations: 1/1, 1/2, 2/2, 2/3, 3/3

The official WeightedTensorProduct is used without source modification, with
connection mode uvu, external per-edge weights, and path-preserving output
multiplicities. The expected natural-parity path set is exactly equal to the
native ICTP path set in every tested configuration.

## Forward-only timing (ms)

| hidden/edge L | e3nn | cartnn | ICTP | ICTC compiled |
| --- | ---: | ---: | ---: | ---: |
| 1/1 | 1.34 | 1.34 | 2.22 | 2.51 |
| 1/2 | 3.80 | 3.80 | 4.86 | 3.42 |
| 2/2 | 13.48 | 16.38 | 24.28 | 10.70 |
| 2/3 | 18.70 | 22.48 | 31.28 | 14.03 |
| 3/3 | 34.83 | 97.63 | 154.80 | 33.91 |

The e3nn/cartnn/ICTC forward values are the archived matched-fusion records;
the ICTP values were measured in the same software environment and GPU session.

## Forward + backward timing (ms)

| hidden/edge L | e3nn | cartnn | ICTP | ICTC compiled |
| --- | ---: | ---: | ---: | ---: |
| 1/1 | 7.66 | 7.66 | 7.54 | 10.85 |
| 1/2 | 13.48 | 13.49 | 11.98 | 14.39 |
| 2/2 | 50.80 | 62.64 | 67.25 | 45.64 |
| 2/3 | 67.47 | 81.61 | 82.33 | 56.91 |
| 3/3 | 174.80 | OOM | OOM | 128.67 |

The non-ICTP rows in this table were rerun immediately after ICTP under the
same environment and launch conditions. Both ambient-$3^\ell$ implementations
exceed 24 GB for the 3/3 forward-plus-backward workload.

## Validation

Before timing, the adapter checks path-set equality, path-preserving output
width, finite input/weight gradients, rotation covariance, output symmetry,
and tracelessness in float64. Across the five configurations:

- maximum relative covariance residual: 6.03e-16
- maximum relative symmetry residual: 4.17e-16
- maximum relative trace residual: 7.55e-16
- all path sets and output widths match
- all tested gradients are finite

Raw data and provenance:

- operator_ictp_fwd.csv
- operator_ictp_fwbw.csv
- operator_ictp_validation.json
- operator_compile_fwbw_matched_rerun.csv
- operator_native_refs_rerun.csv

## License acknowledgement

The ICTP software used in this benchmark was developed by NEC Laboratories
Europe GmbH. It is an external non-commercial research dependency and is not
vendored or modified by MACE-ICTC.
