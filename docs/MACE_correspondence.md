# MACE-ICTD vs original MACE — a component-by-component correspondence

This document establishes, block by block, that the baseline `PureCartesianICTDFix` is **MACE
expressed in an irreducible Cartesian tensor basis** (the ICTD basis). Every learnable operation
is either *numerically identical* to MACE's or related to MACE's by an *exact orthogonal change
of basis*; the Clebsch–Gordan algebra is preserved exactly and precomputed into dense `U` matrices.

All numbers below are reproducible with [`compare_to_mace.py`](compare_to_mace.py)
(`PYTHONPATH=<repo-root> python docs/compare_to_mace.py`, needs `mace-torch` installed).
Checked on torch 2.5.1 / e3nn 0.5.9 / mace-torch, float64.

---

## 0. Pipeline-level correspondence

| stage | original MACE (`ScaleShiftMACE`) | baseline ICTD-MACE (`PureCartesianICTDFix`) |
|---|---|---|
| node embedding | `LinearNodeEmbeddingBlock` (one-hot Z → `C×0e`) | `node_embedding = nn.Linear(num_elements, C, bias=False)` |
| radial | `RadialEmbeddingBlock` = Bessel × `PolynomialCutoff` | `mace_radial_embedding` = e3nn Bessel × `mace_polynomial_cutoff` |
| angular | `o3.SphericalHarmonics(0..L)` of edge dirs | ICTD `direction_harmonics(n, 0..L)` (irreducible Cartesian) |
| message passing ×N | `InteractionBlock` (TP of node ⊗ Yₗ, radial-weighted) + residual self-connection | `ICTDResidualInteractionBlock` (same TP via `U`-matrices) + residual `sc` |
| many-body | `EquivariantProductBasisBlock` (element-wise symmetric contraction, order ν) | `ICTDPureUProductBasisBlockSO3` = `ICTDPureUSymmetricContractionSO3` + `o3.Linear` |
| readout | `LinearReadoutBlock` / `NonLinearReadoutBlock` per layer | `EquivariantScalarReadoutSO3` / `MACEStyleScalarReadoutSO3` (`C×0e → 16×0e → 1×0e`) |
| energy assembly | Σ layer energies, `ScaleShiftBlock`, `AtomicEnergiesBlock` (E0) | Σ layer energies, `energy_output_scale` / `energy_output_shift`, E0 wrapper (`mff-export-aoti --embed-e0`) |

The mapping is **1:1**. The only structural choices that differ are basis-representation choices
(dense `(C, 2l+1)` Cartesian blocks instead of `e3nn` spherical irreps), not model changes.

---

## 1. Radial embedding — *byte-literal identical*

MACE uses `BesselBasis(r_max, num_basis) × PolynomialCutoff(r_max, p)`. ICTD's
`mace_radial_embedding` uses the same sinc-Bessel functions (via e3nn) and a polynomial cutoff
whose envelope formula **mirrors MACE's `PolynomialCutoff`** verbatim. With the default
`radial_sqrt_num_basis=False` it is byte-literal — **no scale adjustment**:

```
cutoff envelope   max |ICTD − MACE| = 0.0e+00      (exactly identical)
full radial       max abs err       = 5.0e-16      (machine precision, no scale fudge)
```

A legacy `√num_basis` constant (which the first radial linear absorbs, so it is numerically inert
for a trained model) is available via `radial_sqrt_num_basis=True`; `from_checkpoint` sets it `True`
so FSCETP checkpoints trained with that scale still deploy correctly.

---

## 2. Angular basis — *exact orthogonal change of basis*

ICTD stores degree-`l` features as the `2l+1` components of an **irreducible Cartesian tensor**
rather than e3nn real spherical harmonics. The two bases are related by a fixed orthogonal matrix
`Q_l` (`SO3ToE3NNBasisBridge._q(l)`). Measured per degree:

| l | `‖QₗᵀQₗ − I‖∞` (orthogonality) | `ICTD·Qₗ` vs e3nn SH (max rel err) |
|---|---|---|
| 0 | 0.0e+00 | 2.2e-15 |
| 1 | 1.3e-15 | 6.8e-16 |
| 2 | 1.3e-15 | 1.2e-15 |
| 3 | 1.3e-15 | 1.6e-15 |

So `direction_harmonics(n, l) · Qₗ = Yₗ(n)` to machine precision: the ICTD angular basis **is** the
e3nn/MACE real-spherical basis, rotated by an exact orthogonal `Qₗ`. Because `Qₗ` is orthogonal,
every SO(3)-equivariant contraction (the Clebsch–Gordan algebra) is preserved exactly.

---

## 3. Many-body product basis — *the same symmetric contraction*

This is the heart of MACE (the `EquivariantProductBasisBlock` / symmetric contraction up to
correlation order ν). The ICTD model ships interchangeable backends:

- `native-mace` — instantiates MACE's own `MaceSymmetricContraction` (the real
  `mace.modules.symmetric_contraction` code), bridged into the ICTD component basis by `Q`.
- `ictd-bridge-u` — the default exact-MACE backend; it folds MACE's `U` tensors into the ICTD
  basis once, then runs in the ICTD component basis.
- `ictd-pure-u` — a dense-`U` ablation whose CG coefficients are generated directly from ICTD
  CG tensors (precomputed, cuBLAS-friendly).

Built with identical initialization (same seed) and run on the same 40-atom graph (channels=16,
lmax=2, ν=3):

```
backend=native-mace  uses MaceSymmetricContraction=True   E=-0.006864315863539
backend=ictd-pure-u  uses MaceSymmetricContraction=False  E=-0.006864316371404
   |dE| = 5.1e-10     max|dF| = 1.6e-9        (~9 significant figures)
```

The two **independent implementations of the MACE symmetric contraction agree to ~9 significant
figures**. The residual is float64 accumulation between the cached-`U` path and the live
MACE-CG + `Q`-fold path — a numerical, not structural, difference. Both backends are exactly
SO(3)-equivariant (force-equivariance error ~1e-20).

> This is the decisive granular check: ICTD's dense-`U` product basis computes the *same function*
> as MACE's symmetric contraction.

---

## 4. Per-component parameter correspondence

Small config (channels=16, lmax=2, 2 layers, ν=3), baseline ICTD-MACE:

| MACE block → ICTD-MACE block | #params |
|---|---|
| `LinearNodeEmbeddingBlock` → `node_embedding` | 64 |
| `InteractionBlock ×2` → `ICTDResidualInteractionBlock ×2` | 41,280 |
| `EquivariantProductBasisBlock ×2` → `ICTDPureUProductBasisBlockSO3 ×2` | 6,400 |
| **total** | **48,050** |

The parameter budget sits in the same places as MACE (interactions ≫ products), confirming the
backbone is MACE, not a different architecture.

---

## 5. Deliberate, documented differences

These are representation / engineering choices, **not** changes to the equivariant function class:

1. **Dense Cartesian storage** `(C, 2l+1)` instead of e3nn spherical irreps → `(L+1)²` polynomial
   scaling and dense GEMMs, the reason ICTD-MACE is AOTInductor / cuBLAS friendly.
2. **Last-layer contraction to `l=0`** (`product_target_lmax[-1]=0`), exactly as MACE does.
3. **Readout** uses the MACE-standard `C×0e → 16×0e → 1×0e` MLP shape (`MACEStyleScalarReadoutSO3`).
4. **`energy_output_scale` / `energy_output_shift`** play the role of MACE's `ScaleShiftBlock`:
   `E_inter_atom = scale * readout + shift`; E0 is added separately.
5. **Long-range module** (`long_range.py`) is an *additive* extension with no MACE counterpart;
   it is off / zero-initialized by default, so it does not perturb the MACE-equivalent core.

---

## 6. Putting features in the original-MACE basis (the orthogonal `Q`)

If you want any **equivariant** quantity numerically in the original-MACE / e3nn convention, right-
multiply it by the fixed block-diagonal orthogonal `Q = diag(Q_0, …, Q_lmax)` from §2. This is
available across the whole pipeline:

- **Python** — `mace_ictd.mace_basis`:
  ```python
  from mace_ictd.mace_basis import orthogonal_Q, to_mace_basis, to_ictd_basis
  x_mace = to_mace_basis(x_ictd, lmax)     # x has the (lmax+1)**2 angular components in its last axis
  x_ictd = to_ictd_basis(x_mace, lmax)     # exact inverse (Q is orthogonal)
  ```
  The model also exposes `model.to_mace_basis(x)` / `model.to_ictd_basis(x)` (use its own `lmax`).
- **LAMMPS / C++** — `lammps_user_mfftorch/src/USER-MFFTORCH/mff_mace_basis.h` ships the **same**
  `Q` constants (lmax 0–4, bit-identical to the Python `orthogonal_Q`) plus
  `mff_mace_basis::to_mace_basis(x, out, rows, lmax)`. A LAMMPS compute that dumps equivariant
  per-atom tensors can convert them to the MACE convention with this helper.

**It is an orthogonal change of the angular basis, so energy, forces and the virial are unchanged**
(they are SO(3) invariants / physical Cartesian tensors). A standard LAMMPS energy/force/MD run is
therefore byte-identical with or without `Q`; the conversion only matters for equivariant (`l≥1`)
features you choose to expose. Verified: a converted feature transforms under rotation by the e3nn
Wigner-D matrix to `6e-15` (i.e. it is genuinely in the e3nn/MACE convention); Python and C++ agree
to `~1e-12` (float64 matmul summation order; the `Q` constants are bit-identical).

## Conclusion

Radial: identical. Angular: identical up to an exact orthogonal `Q`. Symmetric contraction: the
same operation (two implementations agree to ~9 sig figs; one *is* MACE's code). Readout / energy
assembly: MACE-standard. **The baseline ICTD-MACE is MACE in the irreducible Cartesian tensor
basis** — any accuracy gap to a reference MACE is a training/framework matter, not architecture.
