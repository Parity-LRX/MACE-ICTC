#!/usr/bin/env python
"""Operator-level throughput benchmark: MACE-ICTD ICTD product vs cartnn Cartesian
tensor product (and e3nn spherical TP as the MACE-native reference).

The *matched operator* is the equivariant tensor product that couples a hidden node
feature (degrees 0..hidden_lmax, C channels) with the edge angular embedding
(degrees 0..max_ell), per-edge weighted, over a batch of E directed edges -- i.e. the
MACE convolution tensor product, evaluated on identical (l1,l2,l3) natural-parity paths
on all three backends:

  backend=ictd   : mace_ictd EdgeWeightedPathPreservingTensorProduct  (ICTD 2l+1 basis)
  backend=cartnn : cartnn.o3.TensorProduct driven by cartesian_3j     (full 3**l Cartesian)
  backend=e3nn   : e3nn.o3.TensorProduct driven by wigner_3j          (spherical 2l+1; reference)

NOTE on semantics: cartnn stores a degree-l irreducible Cartesian tensor in its full
3**l component layout (not 2l+1), and ships NO symmetric-contraction operator (the
authors declined to implement ICTC). Therefore this is an *operator-level comparable
workload* (same path set, same edge batch, same per-edge weight count) and NOT an exact
apples-to-apples comparison; the symmetric contraction is intentionally out of scope.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import sys
import time
import traceback

import torch

import e3nn
from e3nn import o3 as e3o3
import cartnn
from cartnn import o3 as co3

from mace_ictd.models.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
from mace_ictd.models.pure_cartesian_ictd_fix import _tp_allowed_paths_from_target_lmax

ICTD_URL = "local /home/ylzhang/lrx/MACE-ICTD (== github MACE-ICTD); local git 414aa25"
ICTD_COMMIT = "414aa25"
CARTNN_URL = "https://github.com/xvzemin/cartnn"
CARTNN_COMMIT = "4d0dc381ffe76d62ccddb5cf8ab5030b270a5869"
E3NN_URL = "https://github.com/e3nn/e3nn"
E3NN_COMMIT = f"e3nn=={getattr(e3nn,'__version__','?')}"

CSV_COLUMNS = [
    "backend", "package_url", "package_commit", "op_name", "semantic_equivalence",
    "hidden_lmax", "max_ell", "correlation", "channels", "edges", "dtype", "mode",
    "warmup", "measured", "forward_ms", "backward_ms", "total_ms", "edges_per_s",
    "peak_mem_gb", "status", "error", "notes",
]

SEMEQ_ICTD = ("operator-level comparable workload: ICTD conv tensor product in the 2l+1 "
              "irreducible-Cartesian (ICTD) basis; same (l1,l2,l3) natural-parity path set "
              "and per-edge weight count as cartnn/e3nn")
SEMEQ_CARTNN = ("operator-level comparable workload: cartnn ICTP via cartesian_3j in the FULL "
                "3**l Cartesian layout; same (l1,l2,l3) path set as ictd/e3nn but more numbers "
                "per degree-l (3**l vs 2l+1); NOT exact apples-to-apples; no contraction op")
SEMEQ_E3NN = ("reference: e3nn spherical o3.TensorProduct (wigner_3j, 2l+1) == MACE conv_tp core; "
              "same (l1,l2,l3) path set")


def parity(l: int) -> int:
    return 1 if l % 2 == 0 else -1


def build_paths(hidden_lmax: int, max_ell: int, target_lmax: int):
    return [tuple(p) for p in _tp_allowed_paths_from_target_lmax(hidden_lmax, max_ell, target_lmax)]


# ---------------------------------------------------------------- ICTD operator
def build_ictd(hidden_lmax, max_ell, target_lmax, C, dtype, device):
    paths = build_paths(hidden_lmax, max_ell, target_lmax)
    L = max(hidden_lmax, max_ell, target_lmax)
    tp = EdgeWeightedPathPreservingTensorProduct(
        channels=C, lmax=L, allowed_paths=paths, path_policy="full",
    ).to(device=device, dtype=dtype)
    return tp, paths


def ictd_make_inputs(tp, hidden_lmax, max_ell, C, E, dtype, device, rg):
    x1 = {l: torch.randn(E, C, 2 * l + 1, device=device, dtype=dtype, requires_grad=rg)
          for l in range(hidden_lmax + 1)}
    edge = {l: torch.randn(E, 1, 2 * l + 1, device=device, dtype=dtype)
            for l in range(max_ell + 1)}
    gates = torch.randn(E, tp.num_paths * C, device=device, dtype=dtype, requires_grad=rg)
    leaves = list(x1.values()) + ([gates] if rg else [])
    return (x1, edge, gates), leaves


def ictd_forward(tp, inp):
    x1, edge, gates = inp
    return tp(x1, edge, gates)


def ictd_loss(out):
    return sum(v.pow(2).sum() for v in out.values())


# ------------------------------------------------------ e3nn / cartnn operators
def build_eo3(o3mod, hidden_lmax, max_ell, target_lmax, C, dtype, device):
    paths = build_paths(hidden_lmax, max_ell, target_lmax)
    in1 = o3mod.Irreps([(C, (l, parity(l))) for l in range(hidden_lmax + 1)])
    in2 = o3mod.Irreps([(1, (l, parity(l))) for l in range(max_ell + 1)])

    def idx(irreps, l):
        for i, (m, ir) in enumerate(irreps):
            if ir.l == l:
                return i
        raise KeyError(l)

    irreps_out = []
    instr = []
    for (l1, l2, l3) in paths:
        k = len(irreps_out)
        irreps_out.append((C, (l3, parity(l3))))
        instr.append((idx(in1, l1), idx(in2, l2), k, "uvu", True))
    out = o3mod.Irreps(irreps_out)
    tp = o3mod.TensorProduct(in1, in2, out, instr,
                             shared_weights=False, internal_weights=False)
    tp = tp.to(device=device, dtype=dtype)
    return tp, paths, in1, in2, out


def eo3_make_inputs(tp, in1, in2, E, dtype, device, rg):
    x1 = torch.randn(E, in1.dim, device=device, dtype=dtype, requires_grad=rg)
    x2 = torch.randn(E, in2.dim, device=device, dtype=dtype)
    w = torch.randn(E, tp.weight_numel, device=device, dtype=dtype, requires_grad=rg)
    leaves = [x1, w] if rg else []
    return (x1, x2, w), leaves


def eo3_forward(tp, inp):
    x1, x2, w = inp
    return tp(x1, x2, w)


def eo3_loss(out):
    return out.pow(2).sum()


# ----------------------------------------------------------------------- timing
def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_config(forward_fn, loss_fn, inp, leaves, mode, warmup, measured, device):
    """Return (forward_ms_median, backward_ms_median). CUDA-event timed."""
    use_events = device.type == "cuda"

    def zero_grads():
        for t in leaves:
            t.grad = None

    # warmup
    for _ in range(warmup):
        if mode == "forward_only":
            with torch.no_grad():
                _ = forward_fn(inp)
        else:
            zero_grads()
            out = forward_fn(inp)
            loss_fn(out).backward()
    cuda_sync(device)

    fwd, bwd = [], []
    for _ in range(measured):
        if mode == "forward_only":
            if use_events:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                with torch.no_grad():
                    e0.record(); _ = forward_fn(inp); e1.record()
                torch.cuda.synchronize(device)
                fwd.append(e0.elapsed_time(e1))
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = forward_fn(inp)
                fwd.append((time.perf_counter() - t0) * 1e3)
            bwd.append(0.0)
        else:
            zero_grads()
            if use_events:
                ef0 = torch.cuda.Event(enable_timing=True); ef1 = torch.cuda.Event(enable_timing=True)
                eb0 = torch.cuda.Event(enable_timing=True); eb1 = torch.cuda.Event(enable_timing=True)
                ef0.record(); out = forward_fn(inp); ef1.record()
                loss = loss_fn(out)
                eb0.record(); loss.backward(); eb1.record()
                torch.cuda.synchronize(device)
                fwd.append(ef0.elapsed_time(ef1)); bwd.append(eb0.elapsed_time(eb1))
            else:
                t0 = time.perf_counter(); out = forward_fn(inp); cuda_sync(device)
                t1 = time.perf_counter(); loss_fn(out).backward(); cuda_sync(device)
                t2 = time.perf_counter()
                fwd.append((t1 - t0) * 1e3); bwd.append((t2 - t1) * 1e3)
    return statistics.median(fwd), statistics.median(bwd)


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", default="1:1,1:2,2:2,2:3,3:3")
    ap.add_argument("--channels", default="32,64,128")
    ap.add_argument("--edges", default="10000,50000,100000,500000")
    ap.add_argument("--dtypes", default="float32,float64")
    ap.add_argument("--backends", default="e3nn,cartnn,ictd")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measured", type=int, default=100)
    ap.add_argument("--slow-ms", type=float, default=15.0,
                    help="if a warmup forward exceeds this, drop measured to --slow-measured")
    ap.add_argument("--slow-measured", type=int, default=30)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    configs = [tuple(int(x) for x in c.split(":")) for c in args.configs.split(",")]
    channels = [int(x) for x in args.channels.split(",")]
    edges_list = [int(x) for x in args.edges.split(",")]
    dtypes = [{"float32": torch.float32, "float64": torch.float64}[d] for d in args.dtypes.split(",")]
    backends = args.backends.split(",")

    rows = []
    csv_path = os.path.join(args.out, "operator_cartnn_vs_ictd.csv")
    f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    writer.writeheader(); f.flush()

    def emit(**kw):
        row = {c: kw.get(c, "") for c in CSV_COLUMNS}
        writer.writerow(row); f.flush()
        rows.append(row)
        print(f"[{row['status']:5s}] {row['backend']:6s} "
              f"l{row['hidden_lmax']}/{row['max_ell']} C{row['channels']} E{row['edges']} "
              f"{row['dtype']:7s} {row['mode']:16s} "
              f"fwd={row['forward_ms']} bwd={row['backward_ms']} eps={row['edges_per_s']} "
              f"mem={row['peak_mem_gb']} {row['error']}", flush=True)

    meta = dict(
        ictd=(ICTD_URL, ICTD_COMMIT, "ictd_edge_weighted_path_tp", SEMEQ_ICTD),
        cartnn=(CARTNN_URL, CARTNN_COMMIT, "cartnn_cartesian_tensor_product", SEMEQ_CARTNN),
        e3nn=(E3NN_URL, E3NN_COMMIT, "e3nn_channelwise_tensor_product", SEMEQ_E3NN),
    )

    for (hidden_lmax, max_ell) in configs:
        target_lmax = hidden_lmax  # residual interaction keeps hidden lmax
        for C in channels:
            for dtype in dtypes:
                dname = "float32" if dtype == torch.float32 else "float64"
                torch.set_default_dtype(dtype)
                paths = build_paths(hidden_lmax, max_ell, target_lmax)
                npaths = len(paths)
                base_note = f"paths={npaths}; target_lmax={target_lmax}; binary product (body order 2)"

                # build operators once per (cfg, C, dtype); reuse across edges/modes
                built = {}
                for be in backends:
                    try:
                        if be == "ictd":
                            tp, _ = build_ictd(hidden_lmax, max_ell, target_lmax, C, dtype, device)
                            built[be] = ("ictd", tp, None, None)
                        else:
                            o3mod = e3o3 if be == "e3nn" else co3
                            tp, _, in1, in2, _o = build_eo3(o3mod, hidden_lmax, max_ell, target_lmax, C, dtype, device)
                            built[be] = ("eo3", tp, in1, in2)
                    except Exception as exc:  # noqa
                        url, commit, op, sem = meta[be]
                        emit(backend=be, package_url=url, package_commit=commit, op_name=op,
                             semantic_equivalence=sem, hidden_lmax=hidden_lmax, max_ell=max_ell,
                             correlation=2, channels=C, edges="", dtype=dname, mode="build",
                             warmup=args.warmup, measured=0, status="error",
                             error=f"build:{type(exc).__name__}:{exc}"[:300], notes=base_note)

                for E in edges_list:
                    for mode in ("forward_only", "forward_backward"):
                        rg = (mode == "forward_backward")
                        for be in backends:
                            if be not in built:
                                continue
                            url, commit, op, sem = meta[be]
                            kind, tp = built[be][0], built[be][1]
                            try:
                                if device.type == "cuda":
                                    torch.cuda.reset_peak_memory_stats(device)
                                if kind == "ictd":
                                    inp, leaves = ictd_make_inputs(tp, hidden_lmax, max_ell, C, E, dtype, device, rg)
                                    fwd_fn, loss_fn = (lambda i: ictd_forward(tp, i)), ictd_loss
                                else:
                                    in1, in2 = built[be][2], built[be][3]
                                    inp, leaves = eo3_make_inputs(tp, in1, in2, E, dtype, device, rg)
                                    fwd_fn, loss_fn = (lambda i: eo3_forward(tp, i)), eo3_loss

                                # adaptive measured: one timed warmup forward to gauge cost
                                meas = args.measured
                                note = base_note
                                cuda_sync(device)
                                t0 = time.perf_counter()
                                with torch.no_grad():
                                    _ = fwd_fn(inp)
                                cuda_sync(device)
                                if (time.perf_counter() - t0) * 1e3 > args.slow_ms:
                                    meas = args.slow_measured
                                    note = base_note + f"; measured reduced to {meas} (slow op)"

                                fwd_ms, bwd_ms = time_config(fwd_fn, loss_fn, inp, leaves, mode,
                                                             args.warmup, meas, device)
                                total = fwd_ms + (bwd_ms if rg else 0.0)
                                eps = E / (total / 1e3) if total > 0 else 0.0
                                peak = (torch.cuda.max_memory_allocated(device) / 1e9
                                        if device.type == "cuda" else 0.0)
                                emit(backend=be, package_url=url, package_commit=commit, op_name=op,
                                     semantic_equivalence=sem, hidden_lmax=hidden_lmax, max_ell=max_ell,
                                     correlation=2, channels=C, edges=E, dtype=dname, mode=mode,
                                     warmup=args.warmup, measured=meas,
                                     forward_ms=round(fwd_ms, 5), backward_ms=round(bwd_ms, 5),
                                     total_ms=round(total, 5), edges_per_s=round(eps, 1),
                                     peak_mem_gb=round(peak, 4), status="ok", error="", notes=note)
                                del inp, leaves
                                free()
                            except RuntimeError as exc:
                                msg = str(exc)
                                status = "oom" if "out of memory" in msg.lower() else "error"
                                emit(backend=be, package_url=url, package_commit=commit, op_name=op,
                                     semantic_equivalence=sem, hidden_lmax=hidden_lmax, max_ell=max_ell,
                                     correlation=2, channels=C, edges=E, dtype=dname, mode=mode,
                                     warmup=args.warmup, measured=0, status=status,
                                     error=f"{type(exc).__name__}:{msg}"[:300], notes=base_note)
                                free()
                            except Exception as exc:  # noqa
                                emit(backend=be, package_url=url, package_commit=commit, op_name=op,
                                     semantic_equivalence=sem, hidden_lmax=hidden_lmax, max_ell=max_ell,
                                     correlation=2, channels=C, edges=E, dtype=dname, mode=mode,
                                     warmup=args.warmup, measured=0, status="error",
                                     error=f"{type(exc).__name__}:{exc}"[:300], notes=base_note)
                                free()
                # free built ops
                for be in list(built):
                    del built[be]
                free()

    f.close()
    print(f"\nDONE -> {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
