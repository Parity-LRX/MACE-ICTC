import math
from pathlib import Path

import pytest
import torch

from mace_ictc.training.train_loop import ForceTrainer
from mace_ictc.data.preprocessing import extract_data_blocks


def _trainer():
    trainer = ForceTrainer.__new__(ForceTrainer)
    trainer.a = 1.0
    trainer.b = 10.0
    trainer.c = 0.5
    return trainer


def _batch_stats(errors):
    diff = torch.as_tensor(errors, dtype=torch.float64)
    count = float(diff.numel())
    return {
        "energy_loss_sum": 0.0,
        "energy_count": 0.0,
        "force_loss_sum": float(diff.square().sum()),
        "force_count": count,
        "stress_loss_sum": 0.0,
        "stress_count": 0.0,
        "energy_sq_sum": 0.0,
        "energy_abs_sum": 0.0,
        "force_sq_sum": float(diff.square().sum()),
        "force_abs_sum": float(diff.abs().sum()),
        "stress_sq_sum": 0.0,
        "stress_abs_sum": 0.0,
    }


def test_dataset_metrics_are_weighted_by_element_count():
    trainer = _trainer()
    sums = trainer._new_metric_sums()
    for errors in ([0.0], [2.0, 2.0, 2.0]):
        trainer._accumulate_metric_sums(sums, {"_metric_sums": _batch_stats(errors)})

    metrics = trainer._finalize_metric_sums(sums, require_data=True)
    assert metrics["force_rmse"] == pytest.approx(math.sqrt(3.0))
    assert metrics["force_mae"] == pytest.approx(1.5)
    assert metrics["force_loss"] == pytest.approx(3.0)


def test_dataset_metrics_do_not_depend_on_batch_partition():
    trainer = _trainer()

    def aggregate(partition):
        sums = trainer._new_metric_sums()
        for errors in partition:
            trainer._accumulate_metric_sums(sums, {"_metric_sums": _batch_stats(errors)})
        return trainer._finalize_metric_sums(sums, require_data=True)

    split = aggregate(([0.0], [2.0, 2.0, 2.0]))
    single = aggregate(([0.0, 2.0, 2.0, 2.0],))
    for key in ("total_loss", "force_loss", "force_rmse", "force_mae"):
        assert split[key] == pytest.approx(single[key])


def test_empty_validation_metrics_raise():
    trainer = _trainer()
    with pytest.raises(ValueError, match="no batches"):
        trainer._finalize_metric_sums(trainer._new_metric_sums(), require_data=True)


def test_preprocessing_distinguishes_missing_from_zero_stress(tmp_path: Path):
    xyz = tmp_path / "mixed_stress.xyz"
    xyz.write_text(
        "1\nenergy=0 stress=\"0 0 0 0 0 0\"\nH 0 0 0 0 0 0\n"
        "1\nenergy=0\nH 0 0 0 0 0 0\n"
    )
    result = extract_data_blocks(xyz, return_stress_mask=True)
    assert result[-1] == [True, False]


class _LinearEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs):
        del A, batch, edge_src, edge_dst, edge_shifts, cell, kwargs
        return self.weight * pos[:, :1]


def test_force_metrics_use_physical_target_when_loss_target_is_shifted():
    trainer = ForceTrainer(
        _LinearEnergy(), [], device="cpu", dtype=torch.float64,
        atomic_energy_keys=[1], atomic_energy_values=[0.0],
        force_shift_value=2.0, lr_scheduler="none",
    )
    batch = (
        torch.zeros((1, 3), dtype=torch.float64),
        torch.ones(1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty((0, 3), dtype=torch.float64),
        torch.eye(3, dtype=torch.float64).unsqueeze(0),
        torch.zeros((1, 3, 3), dtype=torch.float64),
    )
    out = trainer._compute(batch, training=False)
    assert float(out["force_loss"]) > 0.0
    assert float(out["force_rmse"]) == pytest.approx(0.0)
    assert float(out["force_mae"]) == pytest.approx(0.0)
