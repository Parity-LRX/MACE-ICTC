#!/usr/bin/env python
"""Matched isolated-operator benchmark for Zaverkin et al.'s official ICTP.

The official ICTP ``WeightedTensorProduct`` is evaluated without modifying its
source.  The workload matches the paper's existing isolated convolution test:

* the same natural-parity ``(l1, l2, l3)`` path set;
* ``C`` hidden channels and one edge-angular channel;
* one externally supplied per-edge/per-path/per-channel weight; and
* path-preserving output multiplicities.

ICTP stores each rank-l irreducible Cartesian tensor in the constrained 3**l
ambient layout.  Its per-output-rank blocks concatenate, rather than sum, the
individual paths, so the output multiplicities agree with the e3nn/cartnn/ICTC
operator benchmark after accounting for ICTP's native path ordering.

The ICTP source is an external non-commercial research dependency.  Pass its
checkout with ``--ictp-root``; this script records the pinned upstream commit
but does not vendor or modify ICTP code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mace_ictc.models.pure_cartesian_ictd_fix import _tp_allowed_paths_from_target_lmax


ICTP_URL = "https://github.com/nec-research/ictp"
ICTP_COMMIT = "f40592a5687ec1d03219300ee557b2660f7d0369"
CSV_COLUMNS = [
    "backend", "package_url", "package_commit", "op_name", "semantic_equivalence",
    "hidden_lmax", "max_ell", "correlation", "channels", "edges", "dtype", "mode",
    "warmup", "measured", "forward_ms", "backward_ms", "total_ms", "edges_per_s",
    "peak_mem_gb", "status", "error", "notes",
]


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def free() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_paths(hidden_lmax: int, max_ell: int, target_lmax: int) -> list[tuple[int, int, int]]:
    return [
        tuple(int(v) for v in path)
        for path in _tp_allowed_paths_from_target_lmax(hidden_lmax, max_ell, target_lmax)
    ]


def native_paths(op: torch.nn.Module) -> list[tuple[int, int, int]]:
    return [
        (int(block.l1), int(block.l2), int(block.l3))
        for output_module in op.cps
        for block in output_module.blocks
    ]


def build_ictp(
    weighted_tensor_product,
    hidden_lmax: int,
    max_ell: int,
    target_lmax: int,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
):
    op = weighted_tensor_product(
        in1_l_max=hidden_lmax,
        in2_l_max=max_ell,
        out_l_max=target_lmax,
        in1_features=channels,
        in2_features=1,
        out_features=channels,
        symmetric_product=False,
        connection_mode="uvu",
        internal_weights=False,
        shared_weights=False,
    ).to(device=device, dtype=dtype)
    expected = build_paths(hidden_lmax, max_ell, target_lmax)
    actual = native_paths(op)
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise RuntimeError(f"ICTP path-set mismatch: expected={expected}, actual={actual}")
    return op, expected, actual


def _random_stf_block(
    edges: int,
    features: int,
    rank: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    shape = [edges] + [3] * rank + [features]
    x = torch.randn(*shape, dtype=dtype, device=device)
    if rank < 2:
        return x
    if rank == 2:
        x = 0.5 * (x + x.transpose(1, 2))
        trace = torch.einsum("eiif->ef", x)
        eye = torch.eye(3, dtype=dtype, device=device)
        return x - torch.einsum("ij,ef->eijf", eye, trace) / 3.0
    if rank == 3:
        x = (
            x
            + x.permute(0, 1, 3, 2, 4)
            + x.permute(0, 2, 1, 3, 4)
            + x.permute(0, 2, 3, 1, 4)
            + x.permute(0, 3, 1, 2, 4)
            + x.permute(0, 3, 2, 1, 4)
        ) / 6.0
        trace = torch.einsum("eiikf->ekf", x)
        eye = torch.eye(3, dtype=dtype, device=device)
        correction = (
            torch.einsum("ij,ekf->eijkf", eye, trace)
            + torch.einsum("ik,ejf->eijkf", eye, trace)
            + torch.einsum("jk,eif->eijkf", eye, trace)
        ) / 5.0
        return x - correction
    raise ValueError(f"STF input generation is implemented only through rank 3, got {rank}")


def make_flat_stf(
    lmax: int,
    features: int,
    edges: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    blocks = [
        _random_stf_block(edges, features, rank, dtype=dtype, device=device).flatten(1)
        for rank in range(lmax + 1)
    ]
    return torch.cat(blocks, dim=-1)


def make_inputs(
    op: torch.nn.Module,
    hidden_lmax: int,
    max_ell: int,
    channels: int,
    edges: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
):
    x1 = make_flat_stf(hidden_lmax, channels, edges, dtype, device).requires_grad_(requires_grad)
    x2 = make_flat_stf(max_ell, 1, edges, dtype, device)
    weights = torch.randn(
        edges,
        int(op.n_total_paths),
        channels,
        1,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return (x1, x2, weights), ([x1, weights] if requires_grad else [])


def _split_input(x: torch.Tensor, lmax: int, features: int) -> list[torch.Tensor]:
    blocks = []
    offset = 0
    for rank in range(lmax + 1):
        width = (3**rank) * features
        blocks.append(x[:, offset : offset + width].reshape([x.shape[0]] + [3] * rank + [features]))
        offset += width
    if offset != x.shape[-1]:
        raise RuntimeError(f"input split consumed {offset} of {x.shape[-1]} values")
    return blocks


def _rotate_block(x: torch.Tensor, rotation: torch.Tensor, rank: int) -> torch.Tensor:
    if rank == 0:
        return x
    if rank == 1:
        return torch.einsum("ia,eaf->eif", rotation, x)
    if rank == 2:
        return torch.einsum("ia,jb,eabf->eijf", rotation, rotation, x)
    if rank == 3:
        return torch.einsum("ia,jb,kc,eabcf->eijkf", rotation, rotation, rotation, x)
    raise ValueError(rank)


def rotate_input(x: torch.Tensor, lmax: int, features: int, rotation: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            _rotate_block(block, rotation, rank).flatten(1)
            for rank, block in enumerate(_split_input(x, lmax, features))
        ],
        dim=-1,
    )


def split_output(op: torch.nn.Module, out: torch.Tensor, channels: int) -> list[tuple[int, torch.Tensor]]:
    pieces: list[tuple[int, torch.Tensor]] = []
    offset = 0
    for rank, n_paths in enumerate(op.n_paths):
        width = (3**rank) * channels * int(n_paths)
        block = out[:, offset : offset + width].reshape(
            [out.shape[0]] + [3] * rank + [channels * int(n_paths)]
        )
        for path_index in range(int(n_paths)):
            pieces.append(
                (
                    rank,
                    block[..., path_index * channels : (path_index + 1) * channels],
                )
            )
        offset += width
    if offset != out.shape[-1]:
        raise RuntimeError(f"output split consumed {offset} of {out.shape[-1]} values")
    return pieces


def rotate_output(op: torch.nn.Module, out: torch.Tensor, channels: int, rotation: torch.Tensor) -> torch.Tensor:
    blocks = []
    offset = 0
    for rank, n_paths in enumerate(op.n_paths):
        width = (3**rank) * channels * int(n_paths)
        block = out[:, offset : offset + width].reshape(
            [out.shape[0]] + [3] * rank + [channels * int(n_paths)]
        )
        blocks.append(_rotate_block(block, rotation, rank).flatten(1))
        offset += width
    return torch.cat(blocks, dim=-1)


def tensor_constraint_residuals(op: torch.nn.Module, out: torch.Tensor, channels: int) -> tuple[float, float]:
    symmetry = out.new_tensor(0.0)
    trace = out.new_tensor(0.0)
    for rank, block in split_output(op, out, channels):
        scale = block.abs().max().clamp_min(1.0e-30)
        if rank == 2:
            symmetry = torch.maximum(symmetry, (block - block.transpose(1, 2)).abs().max() / scale)
            trace = torch.maximum(trace, torch.einsum("eiif->ef", block).abs().max() / scale)
        elif rank == 3:
            for perm in ((0, 1, 3, 2, 4), (0, 2, 1, 3, 4), (0, 3, 2, 1, 4)):
                symmetry = torch.maximum(symmetry, (block - block.permute(*perm)).abs().max() / scale)
            trace = torch.maximum(trace, torch.einsum("eiikf->ekf", block).abs().max() / scale)
    return float(symmetry.item()), float(trace.item())


def random_rotation(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype, device=device))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def validate_operator(
    op: torch.nn.Module,
    expected_paths: list[tuple[int, int, int]],
    actual_paths: list[tuple[int, int, int]],
    hidden_lmax: int,
    max_ell: int,
    channels: int,
    device: torch.device,
) -> dict[str, object]:
    dtype = torch.float64
    op = op.to(device=device, dtype=dtype)
    inputs, _ = make_inputs(op, hidden_lmax, max_ell, channels, 16, dtype, device, False)
    x1, x2, weights = inputs
    with torch.no_grad():
        reference = op(x1, x2, weights)
        rotation = random_rotation(dtype, device)
        rotated = op(
            rotate_input(x1, hidden_lmax, channels, rotation),
            rotate_input(x2, max_ell, 1, rotation),
            weights,
        )
        expected_rotated = rotate_output(op, reference, channels, rotation)
        covariance_abs = (rotated - expected_rotated).abs().max()
        covariance_rel = covariance_abs / expected_rotated.abs().max().clamp_min(1.0e-30)
        symmetry_rel, trace_rel = tensor_constraint_residuals(op, reference, channels)

    grad_inputs, leaves = make_inputs(op, hidden_lmax, max_ell, channels, 8, dtype, device, True)
    op(*grad_inputs).square().sum().backward()
    gradients_finite = all(leaf.grad is not None and torch.isfinite(leaf.grad).all() for leaf in leaves)
    expected_width = channels * sum(3**l3 for _l1, _l2, l3 in actual_paths)
    return {
        "expected_paths": [list(path) for path in expected_paths],
        "native_paths": [list(path) for path in actual_paths],
        "path_sets_equal": set(expected_paths) == set(actual_paths),
        "path_count": len(actual_paths),
        "output_width": int(reference.shape[-1]),
        "expected_output_width": int(expected_width),
        "output_width_matches": int(reference.shape[-1]) == int(expected_width),
        "covariance_abs_max": float(covariance_abs.item()),
        "covariance_rel_max": float(covariance_rel.item()),
        "symmetry_rel_max": symmetry_rel,
        "trace_rel_max": trace_rel,
        "gradients_finite": bool(gradients_finite),
    }


def zero_grads(leaves: list[torch.Tensor]) -> None:
    for tensor in leaves:
        tensor.grad = None


def time_fwbw(op, inputs, leaves, warmup: int, measured: int, device: torch.device) -> tuple[float, float]:
    for _ in range(warmup):
        zero_grads(leaves)
        op(*inputs).square().sum().backward()
    cuda_sync(device)

    forward_ms, backward_ms = [], []
    for _ in range(measured):
        zero_grads(leaves)
        start_fwd = torch.cuda.Event(enable_timing=True)
        stop_fwd = torch.cuda.Event(enable_timing=True)
        start_bwd = torch.cuda.Event(enable_timing=True)
        stop_bwd = torch.cuda.Event(enable_timing=True)
        start_fwd.record()
        out = op(*inputs)
        stop_fwd.record()
        loss = out.square().sum()
        start_bwd.record()
        loss.backward()
        stop_bwd.record()
        cuda_sync(device)
        forward_ms.append(start_fwd.elapsed_time(stop_fwd))
        backward_ms.append(start_bwd.elapsed_time(stop_bwd))
    return statistics.median(forward_ms), statistics.median(backward_ms)


def time_forward_only(op, inputs, warmup: int, measured: int, device: torch.device) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            op(*inputs)
    cuda_sync(device)

    forward_ms = []
    with torch.no_grad():
        for _ in range(measured):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            op(*inputs)
            stop.record()
            cuda_sync(device)
            forward_ms.append(start.elapsed_time(stop))
    return statistics.median(forward_ms)


def empty_row() -> dict[str, object]:
    return {column: "" for column in CSV_COLUMNS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ictp-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--configs", default="1:1,1:2,2:2,2:3,3:3")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--edges", type=int, default=100000)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--mode",
        default="forward_backward",
        choices=["forward_only", "forward_backward"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=50)
    args = parser.parse_args()

    sys.path.insert(0, str(args.ictp_root.resolve()))
    from ictp.o3.tensor_product import WeightedTensorProduct

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The archived timing protocol requires a CUDA device")
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    torch.set_default_dtype(dtype)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configs = [tuple(int(value) for value in item.split(":")) for item in args.configs.split(",")]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / (
        "operator_ictp_fwd.csv" if args.mode == "forward_only" else "operator_ictp_fwbw.csv"
    )
    validation_path = args.out / "operator_ictp_validation.json"
    validations: dict[str, object] = {
        "backend": "ictp_official",
        "package_url": ICTP_URL,
        "package_commit": ICTP_COMMIT,
        "acknowledgement": "ICTP software was developed by NEC Laboratories Europe GmbH.",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "configs": {},
    }

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for hidden_lmax, max_ell in configs:
            row = empty_row()
            row.update(
                backend="ictp_official",
                package_url=ICTP_URL,
                package_commit=ICTP_COMMIT,
                op_name="ICTP WeightedTensorProduct external-weight uvu",
                semantic_equivalence=(
                    "matched natural-parity path set and path-preserving multiplicities; "
                    "official constrained 3**l Cartesian layout; native per-block opt_einsum_fx"
                ),
                hidden_lmax=hidden_lmax,
                max_ell=max_ell,
                correlation=2,
                channels=args.channels,
                edges=args.edges,
                dtype=args.dtype,
                mode=args.mode,
                warmup=args.warmup,
                measured=args.measured,
            )
            try:
                op, expected_paths, actual_paths = build_ictp(
                    WeightedTensorProduct,
                    hidden_lmax,
                    max_ell,
                    hidden_lmax,
                    args.channels,
                    dtype,
                    device,
                )
                validations["configs"][f"{hidden_lmax}:{max_ell}"] = validate_operator(
                    op,
                    expected_paths,
                    actual_paths,
                    hidden_lmax,
                    max_ell,
                    args.channels,
                    device,
                )
                op = op.to(device=device, dtype=dtype).train()
                inputs, leaves = make_inputs(
                    op,
                    hidden_lmax,
                    max_ell,
                    args.channels,
                    args.edges,
                    dtype,
                    device,
                    args.mode == "forward_backward",
                )
                torch.cuda.reset_peak_memory_stats(device)
                if args.mode == "forward_only":
                    forward_ms = time_forward_only(
                        op, inputs, args.warmup, args.measured, device
                    )
                    backward_ms = 0.0
                else:
                    forward_ms, backward_ms = time_fwbw(
                        op, inputs, leaves, args.warmup, args.measured, device
                    )
                total_ms = forward_ms + backward_ms
                row.update(
                    forward_ms=round(forward_ms, 5),
                    backward_ms=round(backward_ms, 5),
                    total_ms=round(total_ms, 5),
                    edges_per_s=round(args.edges / (total_ms / 1000.0), 1),
                    peak_mem_gb=round(torch.cuda.max_memory_allocated(device) / 1.0e9, 4),
                    status="ok",
                    notes=(
                        f"paths={len(actual_paths)}; native_order={actual_paths}; "
                        "STF inputs; official opt_einsum_fx blocks; NEC acknowledgement required"
                    ),
                )
                print(
                    f"[ok] ICTP l{hidden_lmax}/{max_ell} "
                    f"fwd={forward_ms:.5f} bwd={backward_ms:.5f} total={total_ms:.5f} "
                    f"mem={row['peak_mem_gb']} GB",
                    flush=True,
                )
                del inputs, leaves, op
                free()
            except RuntimeError as exc:
                status = "oom" if "out of memory" in str(exc).lower() else "error"
                row.update(status=status, error=f"{type(exc).__name__}:{exc}"[:300])
                print(f"[{status}] ICTP l{hidden_lmax}/{max_ell}: {exc}", flush=True)
                free()
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}:{exc}"[:300])
                print(f"[error] ICTP l{hidden_lmax}/{max_ell}: {exc}", flush=True)
                free()
            writer.writerow(row)
            handle.flush()

    with validation_path.open("w") as handle:
        json.dump(validations, handle, indent=2)
        handle.write("\n")
    print(f"DONE -> {csv_path}")
    print(f"VALIDATION -> {validation_path}")


if __name__ == "__main__":
    main()
