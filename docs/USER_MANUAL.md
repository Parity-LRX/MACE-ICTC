# MACE-ICTD User Manual

This manual describes the standalone MACE-ICTD repository: what each subsystem is for, which runtime mode to choose, and how to train, convert, export, benchmark, and deploy models.

Chinese version: [USER_MANUAL.zh-CN.md](USER_MANUAL.zh-CN.md)

## 1. What This Repository Contains

MACE-ICTD is a standalone implementation of MACE in the Irreducible Cartesian Tensor Decomposition (ICTD) basis. It keeps the MACE model class and deployment stack independent of the original FSCETP tree.

The repository includes:

- `mace_ictd/models/`: the `PureCartesianICTDFix` model, ICTD irreps, tensor product helpers, MACE-compatible symmetric contractions, radial basis functions, optional ZBL and long-range modules.
- `mace_ictd/training/`: the energy/force/stress trainer and the `make_fx` + Inductor force-step compiler.
- `mace_ictd/data/`: extended-XYZ parsing, H5 dataset loading, graph padding, bucket sampling, and collate functions.
- `mace_ictd/cli/`: command-line tools for training, MACE conversion, AOTInductor export, TorchScript export, and LAMMPS helper generation.
- `mace_ictd/interfaces/`: checkpoint loading, LAMMPS MLIAP wrappers, and deployment-facing compatibility code.
- `mace_ictd/evaluation/`: ASE calculator wrappers.
- `mace_ictd/bench/`: benchmark harnesses comparing MACE-ICTD modes against native `mace-torch`.
- `mace_ictd/test/`: numerical and smoke tests.
- `lammps_user_mfftorch/`: a LAMMPS `USER-MFFTORCH` package with C++ pair styles and LibTorch/AOTI integration.

The core model forward signature is:

```python
model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
```

where:

- `pos`: Cartesian coordinates, shape `[N, 3]`.
- `A`: atomic numbers, not species indices.
- `batch`: graph id for each atom, shape `[N]`.
- `edge_src`, `edge_dst`: directed edge indices.
- `edge_shifts`: integer periodic image shifts, shape `[E, 3]`.
- `cell`: cell tensor, shape `[B, 3, 3]`.
- return value: per-atom interaction energies, normally shape `[N, 1]`.

Atomic reference energies E0 are handled outside the core model in training/export wrappers.

## 2. Installation

Minimal editable install:

```bash
cd /path/to/MACE-ICTD
pip install -e .
```

Optional extras:

```bash
pip install -e ".[pyg]"   # torch-scatter and torch-cluster acceleration
pip install -e ".[cue]"   # cuEquivariance product backend
pip install -e ".[e0]"    # pandas support for fitted E0 CSV files
pip install -e ".[full]"  # all optional dependencies
```

Important runtime expectations:

- Python >= 3.9.
- PyTorch >= 2.4; PyTorch >= 2.7 is recommended for `make_fx` and AOTInductor workflows.
- `e3nn < 0.6` for compatibility with current `mace-torch`.
- CUDA is required for the cuEquivariance and serious AOTI/Inductor benchmark paths.

Optional compiled ICTD tensor-product extension:

```bash
MFF_BUILD_ICTD_TP_EXT=1 pip install -e .
```

For CUDA extension build:

```bash
MFF_BUILD_ICTD_TP_EXT=1 MFF_BUILD_ICTD_TP_CUDA=1 pip install -e .
```

The Python fallback path works without this extension.

## 3. Command-Line Tools

Installed console scripts:

| Command | Python entry point | Purpose |
|---|---|---|
| `mff-convert-mace` | `mace_ictd.cli.convert_mace` | Convert a native `mace-torch` `ScaleShiftMACE` checkpoint to MACE-ICTD. |
| `mff-export-aoti` | `mace_ictd.cli.export_aoti_core` | Export an AOTInductor `.pt2` core for Python/C++/LAMMPS deployment. |
| `mff-export-core` | `mace_ictd.cli.export_libtorch_core` | Export a TorchScript core. Mostly for legacy LibTorch deployment. |
| `mff-lammps` | `mace_ictd.cli.lammps_interface` | Generate helper files for LAMMPS-style deployment. |

Direct module commands are also supported:

```bash
python -m mace_ictd.cli.train --help
python -m mace_ictd.cli.convert_mace --help
python -m mace_ictd.cli.export_aoti_core --help
```

## 4. Core Concepts

### 4.1 ICTD Basis Versus e3nn/MACE Basis

Original MACE uses e3nn spherical features. MACE-ICTD stores equivariant features in an ICTD Cartesian basis. The two bases are related by fixed orthogonal per-`l` matrices `Q`.

For invariant outputs such as total energy, forces, and virial, the basis choice should not change the physical result. For equivariant intermediate features, use:

```python
model.to_mace_basis(x)
model.to_ictd_basis(x)
```

or the lower-level helpers in `mace_ictd.mace_basis`.

### 4.2 `angular_basis`

`angular_basis` controls which basis the model computes in internally:

| Value | Meaning | When to use |
|---|---|---|
| `ictd` | Default ICTD internal basis. | Canonical parity mode, bridge-U mode, safest baseline. |
| `e3nn` | Fold fixed angular operators once so the internal equivariant features are in the original MACE/e3nn convention. | cuEq product performance path; AOTI export with `--cueq-product`. |

Important constraints:

- `ictd-bridge-u` does not expose an e3nn-fold path. If you request `angular_basis=e3nn` without replacing the product backend, export will keep `ictd` and print a warning.
- `cueq` product supports `angular_basis=e3nn`.
- Training with `angular_basis=e3nn` saves already-folded fixed buffers. Checkpoint reload restores the runtime Q blocks and product flags without folding a second time.

### 4.3 Product Backends

| Backend | Description | Recommended use |
|---|---|---|
| `ictd-bridge-u` | Uses MACE/e3nn symmetric-contraction U tensors with the ICTD/e3nn basis bridge folded into the U tensors. | Canonical MACE parity and high-`max_ell` conversion. |
| `cueq` | Uses cuEquivariance for the product/symmetric contraction. | Performance training and inference, especially with `--angular-basis e3nn`. |
| `native-mace` | Calls MACE's native symmetric contraction in the product block. | Debug/reference path. |
| `ictd-pure-u` | Uses ICTD-generated U tensors directly. | Diagnostic path; not the primary high-`max_ell` production path. |

### 4.4 `use_reduced_cg`

`--use-reduced-cg` is a structural option for the product/symmetric contraction. It changes the CG/path layout and weight shapes.

Rules:

- When converting an existing native MACE checkpoint, follow the original `mace_model.use_reduced_cg`. Do not choose it manually.
- Native `mace-torch` training also has this option, named `--use_reduced_cg`.
- For from-scratch MACE-ICTD training, only enable it if you intentionally want the reduced-CG architecture.
- It is not a guaranteed stable-step throughput accelerator. In recent 4090 tests for `cueq + angular_basis=e3nn + make_fx`, stable step-time changed only by roughly -1% to +2%, while compile time improved more.

## 5. Which Mode Should I Use?

| Goal | Recommended mode |
|---|---|
| Exact MACE conversion/parity baseline | `ictd-bridge-u`, `angular_basis=ictd`, usually `dtype=float64`. |
| From-scratch training that should match MACE architecture semantics | `ictd-bridge-u`, `function-type=bessel`, MACE-style ScaleShift enabled. |
| Fast training | `cueq`, `angular_basis=e3nn`, `--train-makefx-compile`, bucketed shapes. |
| Fast AOTI inference from an ICTD checkpoint | `mff-export-aoti --cueq-product --angular-basis e3nn` when cuEq custom ops are deployable. |
| Most conservative deployment | Export the checkpoint without cuEq replacement, keep `angular_basis=checkpoint` or `ictd`. |

Complete canonical parity training command:

```bash
python -m mace_ictd.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --seed 123 \
  --channels 64 \
  --lmax 2 \
  --max-ell 2 \
  --num-interaction 2 \
  --correlation 2 \
  --function-type bessel \
  --product-backend ictd-bridge-u \
  --scaling rms_forces_scaling \
  --epochs 300 \
  --max-steps 200000 \
  --batch-size 4 \
  --loss smooth_l1 \
  --loss-beta 0.5 \
  --energy-weight 1.0 \
  --force-weight 10.0 \
  --stress-weight 0.0 \
  --optimizer adamw \
  --lr 0.001 \
  --min-lr 0.000001 \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --lr-scheduler cosine \
  --warmup-batches 1000 \
  --warmup-start-ratio 0.1 \
  --ema-decay 0.0 \
  --swa-start-epoch -1 \
  --checkpoint-state-source raw \
  --device cuda \
  --dtype float64 \
  --checkpoint model_bridge_u.pth
```

Complete high-performance training command:

```bash
python -m mace_ictd.cli.train \
  --data-dir DATA \
  --train-prefix train \
  --val-prefix val \
  --seed 123 \
  --channels 64 \
  --lmax 2 \
  --max-ell 2 \
  --num-interaction 2 \
  --correlation 2 \
  --function-type bessel \
  --product-backend cueq \
  --angular-basis e3nn \
  --train-makefx-compile \
  --makefx-buckets 6 \
  --makefx-max-slots 8 \
  --pad-nodes-to-max \
  --pad-edges-to-max \
  --scaling rms_forces_scaling \
  --epochs 300 \
  --max-steps 200000 \
  --batch-size 8 \
  --loss smooth_l1 \
  --loss-beta 0.5 \
  --energy-weight 1.0 \
  --force-weight 10.0 \
  --stress-weight 0.0 \
  --optimizer adamw \
  --lr 0.001 \
  --min-lr 0.000001 \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-eps 1e-8 \
  --lr-scheduler cosine \
  --warmup-batches 1000 \
  --warmup-start-ratio 0.1 \
  --ema-decay 0.999 \
  --ema-start-step 1000 \
  --swa-start-epoch -1 \
  --checkpoint-state-source auto \
  --device cuda \
  --dtype float32 \
  --checkpoint model_cueq_e3nn_makefx.pth
```

The numeric values above are safe starting points, not chemistry-independent optimum
hyperparameters. For strict comparison against a native MACE run, match the dataset
split, seed, loss weights, optimizer, scheduler, batch construction, dtype, and
ScaleShift/E0 settings.

## 6. Data Pipeline

The trainer consumes preprocessed H5 files:

```text
DATA/
  processed_train.h5
  processed_val.h5       # optional
  processed_train.counts.npz / bucket sidecars, when generated
```

The parser in `mace_ictd.data.preprocessing` supports extended XYZ-style data with:

- atomic species / atomic numbers,
- Cartesian positions,
- forces,
- total energy,
- cell and PBC flags,
- optional stress or virial.

Programmatic preprocessing entry point:

```python
from mace_ictd.data.preprocessing import save_to_h5_parallel

save_to_h5_parallel(
    prefix="train",
    max_radius=5.0,
    num_workers=8,
    data_dir="DATA",
)
```

The dataset loader is `mace_ictd.data.datasets.H5Dataset`; batching uses `mace_ictd.data.collate.collate_fn_h5`.

For `make_fx` training, prefer size bucketing:

```bash
--train-makefx-compile --makefx-buckets 6
```

This groups similar atom/edge counts so Inductor compiles once per bucket instead of once per raw shape.

## 7. Training Details

Training CLI:

```bash
python -m mace_ictd.cli.train --help
```

Key architecture arguments:

| Argument | Meaning |
|---|---|
| `--channels` | Hidden channel count. |
| `--lmax` | Hidden feature maximum angular order. |
| `--max-ell` | Edge spherical-harmonics cutoff. Defaults to `--lmax`. |
| `--num-interaction` | Number of MACE interaction/product blocks. |
| `--correlation` | MACE product correlation order, also called `save_contraction_order`. |
| `--function-type` | Radial basis type; use `bessel` for MACE-like parity. |
| `--product-backend` | Product backend: usually `ictd-bridge-u` or `cueq`. |
| `--angular-basis` | `ictd` or `e3nn`; use `e3nn` with `cueq` for the performance path. |
| `--use-reduced-cg` | Enable reduced-CG product layout. Must match source MACE when converting. |

Optimization, step, and seed controls:

| Argument | Meaning |
|---|---|
| `--seed` | Seeds Python, NumPy, PyTorch, DataLoader shuffle, and bucket sampling. It improves reproducibility but does not force deterministic CUDA kernels. |
| `--epochs` | Maximum epoch count. |
| `--max-steps` | Optional optimizer-step cap. If set, training stops once this global step is reached, even mid-epoch. |
| `--batch-size` | Graphs per batch or per bucketed batch. |
| `--optimizer` | `adamw` or `adam`. |
| `--lr`, `--min-lr`, `--weight-decay` | Learning rate, cosine minimum LR, and AdamW weight decay. |
| `--adam-beta1`, `--adam-beta2`, `--adam-eps`, `--amsgrad` | Adam/AdamW numerical parameters. |
| `--lr-scheduler` | `cosine`, `step`, or `none`. |
| `--warmup-batches`, `--warmup-start-ratio` | Linear warmup length and initial LR multiplier. |
| `--lr-decay-step`, `--lr-decay-factor` | StepLR parameters when `--lr-scheduler step` is selected. |
| `--max-grad-norm` | Optional gradient clipping threshold. |

Energy/force/stress loss:

```text
total = energy_weight * loss(E)
      + force_weight  * loss(F)
      + stress_weight * loss(stress)
```

| Argument | Meaning |
|---|---|
| `--loss` | `smooth_l1` (default) or `mse`. |
| `--loss-beta` | SmoothL1 beta for energy, force, and stress when `--loss smooth_l1`. |
| `--energy-weight`, `--force-weight`, `--stress-weight` | Weights in the total loss. Stress is disabled by default (`--stress-weight 0`). |
| `--force-shift-value` | Multiplies the reference force before the force loss. Keep at `1.0` unless reproducing a legacy run. |

When stress is enabled, stress is computed from the strain derivative.

EMA and SWA:

| Argument | Meaning |
|---|---|
| `--ema-decay` | Enables exponential moving average when greater than `0`, for example `0.999`. Saved as `e3trans_ema_state_dict`. |
| `--ema-start-step` | First global optimizer step eligible for EMA updates. |
| `--swa-start-epoch` | `-1` disables SWA; otherwise starts arithmetic weight averaging from that epoch. |
| `--swa-start-step` | `-1` disables SWA; otherwise starts arithmetic weight averaging from that global step. |
| `--checkpoint-state-source` | `auto`, `raw`, `ema`, or `swa`. Deploy loaders use `default_state_source`; `auto` prefers EMA, then SWA, then raw. |

The SWA implementation here is weight averaging and checkpoint selection. It does
not by itself add a native MACE-style SWA training phase with a special SWA LR or
changed loss weights; set those explicitly with the optimizer/loss flags if you need
to reproduce such a schedule.

ScaleShift behavior:

- Default: `--scaling rms_forces_scaling`.
- Also available: `std_scaling`, `no_scaling`.
- Override scale/shift explicitly with `--atomic-inter-scale` and `--atomic-inter-shift`.
- Use `--no-atomic-inter-shift` to keep scaling but force zero interaction-energy shift.

E0 behavior:

- Pass `--atomic-energy-keys` and `--atomic-energy-values` for explicit reference energies.
- If omitted, the training CLI uses its built-in H/C/N/O defaults.
- Export can embed E0 into the deployed core with `--embed-e0`.

## 8. Native MACE Conversion

Use this path for an existing `mace-torch` `ScaleShiftMACE` checkpoint:

```bash
mff-convert-mace \
  --mace-model mace.model \
  --out mace_ictd.pth \
  --product-backend ictd-bridge-u \
  --dtype float64 \
  --device cpu
```

Then export:

```bash
mff-export-aoti \
  --checkpoint mace_ictd.pth \
  --elements H,C,N,O \
  --out mace_ictd.pt2 \
  --dynamic \
  --embed-e0
```

For faster cuEq product inference from a converted checkpoint:

```bash
mff-export-aoti \
  --checkpoint mace_ictd.pth \
  --elements H,C,N,O \
  --out mace_ictd_cueq_e3nn.pt2 \
  --dynamic \
  --embed-e0 \
  --cueq-product \
  --angular-basis e3nn
```

Conversion constraints are intentionally strict. Unsupported variants are rejected rather than silently converted. Current supported assumptions include:

- `ScaleShiftMACE`,
- Bessel radial basis,
- uniform correlation across layers,
- no pair repulsion or distance transform,
- MACE-style scalar readout,
- `max_ell >= hidden_irreps.lmax`,
- `num_interactions >= 2`.

The converter reads `mace_model.use_reduced_cg` and rebuilds MACE-ICTD with the same reduced-CG setting.

## 9. AOTInductor Export

Basic export:

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model.pt2 \
  --dynamic \
  --embed-e0
```

Performance-oriented export:

```bash
mff-export-aoti \
  --checkpoint model.pth \
  --elements H,C,N,O \
  --out model_cueq_e3nn.pt2 \
  --dynamic \
  --embed-e0 \
  --cueq-product \
  --angular-basis e3nn \
  --assume-cutoff-edges \
  --preserve-edge-order \
  --fuse-selector-message-linear \
  --inductor-max-autotune
```

Important export options:

| Option | Meaning |
|---|---|
| `--dynamic` | Export with dynamic atom/edge dimensions where supported. |
| `--static-n` | Keep atom count static. Useful for fixed-N MD when dynamic export is problematic. |
| `--embed-e0` | Add atomic reference energies into the exported energy. |
| `--cueq-product` | Replace product blocks with cuEq product blocks during export. |
| `--angular-basis e3nn` | Fold fixed angular operators to e3nn basis for fold-capable products. |
| `--assume-cutoff-edges` | Assume caller already filtered edges inside cutoff; skips model-side edge mask. |
| `--preserve-edge-order` | Assume caller passes stable edge order; skips model-side destination sort. |
| `--fuse-selector-message-linear` | Fuse selected message linears where supported. |
| `--inductor-max-autotune` | Slower compile, potentially faster kernel choices. Benchmark before relying on it. |

`strict=False` export fallback:

- The exporter first tries strict export when appropriate.
- If strict export fails due to exporter limitations, it can retry non-strict export.
- Correctness is still checked by compiling/loading the `.pt2` and comparing numerical outputs.

## 10. ASE and Python Inference

The ASE wrapper is `mace_ictd.evaluation.calculator.MyE3NNCalculator`.

Typical use:

```python
import torch
from mace_ictd.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from mace_ictd.evaluation.calculator import MyE3NNCalculator

wrapper = LAMMPS_MLIAP_MFF.from_checkpoint(
    "model.pth",
    element_types=["H", "C", "N", "O"],
    device="cuda",
)

atoms.calc = MyE3NNCalculator(
    model=wrapper.wrapper.model,
    atomic_energies_dict={1: 0.0, 6: 0.0, 7: 0.0, 8: 0.0},
    device=torch.device("cuda"),
    max_radius=5.0,
)
```

For production MD, prefer exported AOTI/LAMMPS paths after numerical validation.

## 11. LAMMPS Deployment

LAMMPS support lives in:

```text
lammps_user_mfftorch/
```

Read:

- `lammps_user_mfftorch/README.md`
- `lammps_user_mfftorch/docs/BUILD_AND_RUN.md`

The package provides:

- `pair_style mff/torch`
- `pair_style mff/torch/kk`
- `compute ... mff/torch/phys`

General workflow:

1. Train or convert a checkpoint.
2. Export an AOTI `.pt2` or TorchScript core depending on the target LAMMPS integration.
3. Build LAMMPS with `USER-MFFTORCH` and LibTorch.
4. Use the exported model in a LAMMPS input script.

Minimal example:

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data
neighbor 1.0 bin

pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/model.pt2 H C N O

fix 1 all nve
run 100
```

The element order in `pair_coeff` must match the export/load order.

## 12. Benchmarking

Main benchmark harness:

```bash
python -m mace_ictd.bench.bench_mace_ictd_vs_mace \
  --device cuda \
  --dtype float32 \
  --channels 64 \
  --atoms-list 256,1024,4096 \
  --configs 1:1,2:2,2:3 \
  --train-iters 5 \
  --infer-iters 20 \
  --out-dir /tmp/mace_ictd_bench
```

The benchmark reports rows for training and inference modes where supported. Treat it as a kernel/backend throughput harness, not a chemistry validation benchmark.

Recommended comparisons:

- Native `mace-torch` e3nn backend.
- Native `mace-torch` cuEq backend.
- MACE-ICTD bridge-U eager/make_fx/AOTI.
- MACE-ICTD cuEq product eager/make_fx/AOTI.
- Optional pure-U diagnostic path.

Always separate:

- first compile time,
- steady-state step time,
- ASE/Python overhead,
- neighbor-list overhead,
- LAMMPS throughput.

## 13. Tests and Validation

Core smoke tests:

```bash
python -m mace_ictd.test.test_training_smoke
python -m pytest mace_ictd/test/test_angular_basis.py -q
python -m pytest mace_ictd/test/test_export_aoti_core.py -q
```

MACE converter validation:

```bash
python -m mace_ictd.test.test_mace_converter
```

cuEq product tests:

```bash
python -m pytest mace_ictd/test/test_cueq_product_backend.py -q
python -m mace_ictd.test.test_cueq_makefx_training
```

Use a CUDA machine for the cuEq and make_fx tests.

Expected parity levels depend on dtype and backend:

- Float64 bridge-U conversion should reach near machine precision for energy/forces.
- Float32 cuEq paths should be judged with float32 tolerances.
- AOTI and make_fx paths must be compared against eager outputs after compile/load.

## 14. Common Pitfalls

### Bridge-U and `angular_basis=e3nn`

Bridge-U does not have an e3nn fold path. Use:

```bash
--product-backend ictd-bridge-u --angular-basis ictd
```

For e3nn-folded product inference, use:

```bash
--cueq-product --angular-basis e3nn
```

### cuEq Product Replacement

When replacing bridge-U products with cuEq products, only learnable MACE contraction weights should be copied. Fixed bridge-U `U_matrix_*` buffers already contain the ICTD/e3nn basis fold and must not be copied into cuEq.

The export path handles this.

### `use_reduced_cg`

This is a model-structure choice, not a harmless speed flag. If converting native MACE, follow the source checkpoint. If training from scratch, choose intentionally and keep it in checkpoint metadata.

### ScaleShift and E0

MACE-style ScaleShift affects interaction energy. E0 reference energies are added separately. To compare against native MACE or deploy absolute energies, make sure:

- atomic energy keys/values match,
- scale/shift match,
- `avg_num_neighbors` matches,
- export uses `--embed-e0` when the deployment expects absolute energy.

### `max_ell` Versus `lmax`

- `lmax`: hidden feature angular cutoff.
- `max_ell`: edge spherical-harmonics cutoff.

Native MACE often allows `max_ell >= hidden_lmax`. Higher `max_ell` can be much more expensive, especially for product contractions and force training.

### Dynamic Shapes

Dynamic AOTI and make_fx paths are sensitive to PyTorch/Inductor version. If a dynamic export fails, try:

- fixed-N export with `--static-n`,
- fewer dynamic dimensions,
- smaller buckets,
- PyTorch 2.7+,
- disabling optional fusion/autotune flags.

## 15. Development Notes

Local code style is intentionally conservative:

- Prefer existing model/backend abstractions.
- Keep MACE parity tests before changing angular basis or product code.
- Do not treat passing smoke tests as proof of MACE parity; run direct MACE-vs-ICTD converter tests for parity claims.
- When touching checkpoint metadata, verify `LAMMPS_MLIAP_MFF.from_checkpoint` strict reload.
- When touching `angular_basis=e3nn`, verify both eager forward and checkpoint reload to avoid double-folding fixed buffers.

Useful files:

| File | Why it matters |
|---|---|
| `mace_ictd/models/pure_cartesian_ictd_fix.py` | Main model and product backends. |
| `mace_ictd/mace_basis.py` | Orthogonal ICTD/e3nn basis conversion. |
| `mace_ictd/interfaces/mace_converter.py` | Native MACE to MACE-ICTD weight conversion. |
| `mace_ictd/cli/export_aoti_core.py` | AOTI export, cuEq product replacement, angular-basis export logic. |
| `mace_ictd/training/makefx_compile.py` | `make_fx` force-step compilation. |
| `mace_ictd/training/train_loop.py` | Trainer, checkpoint metadata, ScaleShift/E0 loss handling. |
| `mace_ictd/interfaces/lammps_mliap.py` | Deployment checkpoint reload and wrapper logic. |
| `docs/MACE_correspondence.md` | MACE-vs-ICTD mathematical correspondence. |
