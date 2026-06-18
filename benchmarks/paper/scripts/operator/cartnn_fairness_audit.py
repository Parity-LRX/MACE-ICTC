#!/usr/bin/env python
"""Build operator-level fairness audit tables for the cartnn comparison.

The timing plots alone do not show the algebraic workload.  This script records
the matched path set, irreducible-vs-Cartesian storage dimensions, pathwise
coupling tensor sizes, per-edge weight counts, and representative timing cells
for the operator benchmark used in the paper.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from e3nn import o3 as e3o3

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mace_ictd.models.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
from mace_ictd.models.pure_cartesian_ictd_fix import _tp_allowed_paths_from_target_lmax


def parity(l: int) -> int:
    return 1 if l % 2 == 0 else -1


def irrep_dim(l: int) -> int:
    return 2 * l + 1


def cart_dim(l: int) -> int:
    return 3**l


def build_paths(hidden_lmax: int, max_ell: int, target_lmax: int) -> list[tuple[int, int, int]]:
    return [tuple(p) for p in _tp_allowed_paths_from_target_lmax(hidden_lmax, max_ell, target_lmax)]


def build_tp(o3mod, paths: list[tuple[int, int, int]], hidden_lmax: int, max_ell: int, channels: int):
    in1 = o3mod.Irreps([(channels, (l, parity(l))) for l in range(hidden_lmax + 1)])
    in2 = o3mod.Irreps([(1, (l, parity(l))) for l in range(max_ell + 1)])

    def idx(irreps, degree: int) -> int:
        for i, (_mul, ir) in enumerate(irreps):
            if ir.l == degree:
                return i
        raise KeyError(degree)

    irreps_out = []
    instructions = []
    for l1, l2, l3 in paths:
        out_index = len(irreps_out)
        irreps_out.append((channels, (l3, parity(l3))))
        instructions.append((idx(in1, l1), idx(in2, l2), out_index, "uvu", True))
    out = o3mod.Irreps(irreps_out)
    tp = o3mod.TensorProduct(in1, in2, out, instructions, shared_weights=False, internal_weights=False)
    return tp, in1, in2, out


def read_timings(results_dir: Path, channels: int, edges: int) -> pd.DataFrame:
    sources = [
        results_dir / "operator_compile_fwbw_flat.csv",
        results_dir / "operator_cartnn_vs_ictd.csv",
        results_dir / "operator_ictd_compiled.csv",
        results_dir / "operator_aoti_fwd.csv",
    ]
    frames = []
    for source in sources:
        if source.exists():
            frame = pd.read_csv(source)
            frame["source_file"] = source.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[
        (df["status"].eq("ok"))
        & (df["channels"].astype(int).eq(channels))
        & (df["edges"].astype(int).eq(edges))
        & (df["dtype"].eq("float32"))
    ].copy()
    return df


def timing_cell(df: pd.DataFrame, hidden_lmax: int, max_ell: int, backend: str, mode: str) -> dict[str, str]:
    if df.empty:
        return {}
    sub = df[
        (df["hidden_lmax"].astype(int).eq(hidden_lmax))
        & (df["max_ell"].astype(int).eq(max_ell))
        & (df["backend"].eq(backend))
        & (df["mode"].eq(mode))
    ].copy()
    if sub.empty:
        return {}
    # Prefer the flat compile-fwbw rerun for forward+backward because it puts
    # e3nn, cartnn, and compiled ICTD in one script and one launch environment.
    if mode == "forward_backward" and "operator_compile_fwbw_flat.csv" in set(sub["source_file"]):
        sub = sub[sub["source_file"].eq("operator_compile_fwbw_flat.csv")]
    row = sub.iloc[-1]
    return {
        "forward_ms": row.get("forward_ms", ""),
        "backward_ms": row.get("backward_ms", ""),
        "total_ms": row.get("total_ms", ""),
        "peak_mem_gb": row.get("peak_mem_gb", ""),
        "source_file": row.get("source_file", ""),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/paper/results/operator"))
    parser.add_argument("--configs", default="1:1,1:2,2:2,2:3,3:3")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--edges", type=int, default=100000)
    args = parser.parse_args()

    configs = [tuple(int(v) for v in item.split(":")) for item in args.configs.split(",")]
    timing = read_timings(args.results_dir, args.channels, args.edges)

    summary_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []

    for hidden_lmax, max_ell in configs:
        target_lmax = hidden_lmax
        paths = build_paths(hidden_lmax, max_ell, target_lmax)
        e3_tp, e3_in1, e3_in2, e3_out = build_tp(e3o3, paths, hidden_lmax, max_ell, args.channels)
        ictd_tp = EdgeWeightedPathPreservingTensorProduct(
            channels=args.channels,
            lmax=max(hidden_lmax, max_ell, target_lmax),
            allowed_paths=paths,
            path_policy="full",
        )

        hidden_irrep = args.channels * sum(irrep_dim(l) for l in range(hidden_lmax + 1))
        edge_irrep = sum(irrep_dim(l) for l in range(max_ell + 1))
        hidden_cart = args.channels * sum(cart_dim(l) for l in range(hidden_lmax + 1))
        edge_cart = sum(cart_dim(l) for l in range(max_ell + 1))
        out_irrep_pathwise = args.channels * sum(irrep_dim(l3) for _l1, _l2, l3 in paths)
        out_cart_pathwise = args.channels * sum(cart_dim(l3) for _l1, _l2, l3 in paths)
        coupling_irrep = sum(irrep_dim(l1) * irrep_dim(l2) * irrep_dim(l3) for l1, l2, l3 in paths)
        coupling_cart = sum(cart_dim(l1) * cart_dim(l2) * cart_dim(l3) for l1, l2, l3 in paths)

        summary_rows.append(
            {
                "hidden_lmax": hidden_lmax,
                "max_ell": max_ell,
                "target_lmax": target_lmax,
                "channels": args.channels,
                "num_paths": len(paths),
                "path_set": " ".join(f"({l1},{l2},{l3})" for l1, l2, l3 in paths),
                "e3nn_hidden_dim": e3_in1.dim,
                "e3nn_edge_dim": e3_in2.dim,
                "e3nn_out_dim_pathwise": e3_out.dim,
                "cartnn_hidden_dim_formula": hidden_cart,
                "cartnn_edge_dim_formula": edge_cart,
                "cartnn_out_dim_pathwise_formula": out_cart_pathwise,
                "formula_hidden_irrep_dim": hidden_irrep,
                "formula_edge_irrep_dim": edge_irrep,
                "formula_hidden_cart_dim": hidden_cart,
                "formula_edge_cart_dim": edge_cart,
                "formula_out_irrep_pathwise_dim": out_irrep_pathwise,
                "formula_out_cart_pathwise_dim": out_cart_pathwise,
                "hidden_cart_over_irrep": hidden_cart / hidden_irrep,
                "edge_cart_over_irrep": edge_cart / edge_irrep,
                "out_cart_over_irrep": out_cart_pathwise / out_irrep_pathwise,
                "coupling_tensor_elements_irrep_sum": coupling_irrep,
                "coupling_tensor_elements_cart_sum": coupling_cart,
                "coupling_cart_over_irrep": coupling_cart / coupling_irrep,
                "largest_path_irrep_elements": max(irrep_dim(l1) * irrep_dim(l2) * irrep_dim(l3) for l1, l2, l3 in paths),
                "largest_path_cart_elements": max(cart_dim(l1) * cart_dim(l2) * cart_dim(l3) for l1, l2, l3 in paths),
                "e3nn_weight_numel": e3_tp.weight_numel,
                "cartnn_weight_numel_formula": len(paths) * args.channels,
                "ictd_weight_numel": ictd_tp.num_paths * args.channels,
                "e3nn_fusion_level": "e3nn TensorProduct opt_einsum/codegen",
                "cartnn_fusion_level": "cartnn TensorProduct opt_einsum/codegen with cartesian_3j",
                "ictd_fusion_level": "ICTD eager or torch.compile/AOTI depending on row timing",
            }
        )

        for path_index, (l1, l2, l3) in enumerate(paths):
            irrep_elements = irrep_dim(l1) * irrep_dim(l2) * irrep_dim(l3)
            cart_elements = cart_dim(l1) * cart_dim(l2) * cart_dim(l3)
            path_rows.append(
                {
                    "hidden_lmax": hidden_lmax,
                    "max_ell": max_ell,
                    "path_index": path_index,
                    "l1": l1,
                    "l2": l2,
                    "l3": l3,
                    "irrep_dims": f"{irrep_dim(l1)}x{irrep_dim(l2)}x{irrep_dim(l3)}",
                    "cartnn_dims": f"{cart_dim(l1)}x{cart_dim(l2)}x{cart_dim(l3)}",
                    "irrep_coupling_elements": irrep_elements,
                    "cartnn_coupling_elements": cart_elements,
                    "cart_over_irrep": cart_elements / irrep_elements,
                }
            )

        for backend, label in [
            ("e3nn", "MACE e3nn TensorProduct"),
            ("cartnn", "cartnn Cartesian-3j TensorProduct"),
            ("ictd_compile_fwbw", "ICTD torch.compile fused product"),
        ]:
            cell = timing_cell(timing, hidden_lmax, max_ell, backend, "forward_backward")
            if not cell:
                continue
            timing_rows.append(
                {
                    "hidden_lmax": hidden_lmax,
                    "max_ell": max_ell,
                    "channels": args.channels,
                    "edges": args.edges,
                    "dtype": "float32",
                    "mode": "forward_backward",
                    "backend": backend,
                    "backend_label": label,
                    **cell,
                }
            )

    write_csv(args.out_dir / "cartnn_fairness_operator_structure.csv", summary_rows)
    write_csv(args.out_dir / "cartnn_fairness_path_details.csv", path_rows)
    write_csv(args.out_dir / "cartnn_fairness_timing_c64_e100k_fwbw.csv", timing_rows)
    print(f"wrote audit tables to {args.out_dir}")


if __name__ == "__main__":
    main()
