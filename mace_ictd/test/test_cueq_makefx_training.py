"""CUDA smoke for ``--train-makefx-compile`` with ``product_backend='cueq'``.

Run:
    python -m mace_ictd.test.test_cueq_makefx_training
"""

from __future__ import annotations

import os
import tempfile

import torch
from torch.utils.data import DataLoader

from mace_ictd.cli.train import build_baseline_model, _avg_num_neighbors_from_h5
from mace_ictd.data.collate import collate_fn_h5
from mace_ictd.data.datasets import H5Dataset
from mace_ictd.test.test_training_smoke import _make_h5
from mace_ictd.training.train_loop import ForceTrainer
from mace_ictd.utils.config import ModelConfig


def run_case(*, use_reduced_cg: bool) -> dict[str, float | int | bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuEq make_fx training smoke")

    torch.manual_seed(21 + int(use_reduced_cg))
    dtype = torch.float32
    device = torch.device("cuda")
    tmp = tempfile.mkdtemp(prefix="cueq_makefx_")
    train_h5 = os.path.join(tmp, "processed_train.h5")
    _make_h5(train_h5, sizes=[6, 6], seed=8 + int(use_reduced_cg))
    avg_num_neighbors = _avg_num_neighbors_from_h5(train_h5)

    dataset = H5Dataset(prefix="train", data_dir=tmp)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_h5)

    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = 4
    cfg.irreps_output_conv_channels = 4
    cfg.lmax = 1
    cfg.num_layers = 1
    cfg.max_radius = 5.0
    cfg.max_radius_main = 5.0
    cfg.function_type = "gaussian"
    cfg.internal_compute_dtype = dtype

    model = build_baseline_model(
        cfg,
        avg_num_neighbors=avg_num_neighbors,
        num_interaction=2,
        route="baseline",
        product_backend="cueq",
        correlation=2,
        use_reduced_cg=use_reduced_cg,
        radial_sqrt_num_basis=False,
        edge_lmax=None,
        attn_heads=0,
        atomic_numbers=[1, 6, 7, 8],
        ictd_save_tp_mode="fully-connected",
        invariant_channels=4,
        device=device,
        dtype=dtype,
    )
    trainer = ForceTrainer(
        model,
        loader,
        device=device,
        config=cfg,
        dtype=dtype,
        max_radius=5.0,
        learning_rate=1e-3,
        lr_scheduler="none",
        epochs=1,
        train_makefx_compile=True,
        require_train_makefx_compile=True,
        makefx_max_slots=2,
        extra_hparams={
            "ictd_fix_product_backend": "cueq",
            "ictd_fix_use_reduced_cg": bool(use_reduced_cg),
        },
    )
    out = trainer.train_epoch(0)
    cache_size = 0 if trainer._makefx_cache is None else len(trainer._makefx_cache._cache)
    if trainer._makefx_disabled or cache_size < 1:
        raise AssertionError("cuEq make_fx training fell back to eager")
    return {
        "use_reduced_cg": bool(use_reduced_cg),
        "loss": float(out["total_loss"]),
        "cache_size": int(cache_size),
    }


def main() -> None:
    if not torch.cuda.is_available():
        print("cueq makefx training SKIP CUDA is not available")
        return
    for use_reduced_cg in (False, True):
        result = run_case(use_reduced_cg=use_reduced_cg)
        print(
            "cueq makefx training PASS "
            f"reduced={result['use_reduced_cg']} "
            f"loss={result['loss']:.6g} "
            f"cache={result['cache_size']}"
        )


if __name__ == "__main__":
    main()
