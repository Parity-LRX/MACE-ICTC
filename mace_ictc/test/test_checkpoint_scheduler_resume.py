import pytest
import torch

from mace_ictc.training.train_loop import ForceTrainer


def _make_trainer(*, warmup_batches=0, epochs=5, lr_scheduler="cosine"):
    return ForceTrainer(
        torch.nn.Linear(1, 1, dtype=torch.float64),
        train_loader=[None] * 10,
        device="cpu",
        dtype=torch.float64,
        learning_rate=1.0e-3,
        min_learning_rate=1.0e-6,
        lr_scheduler=lr_scheduler,
        warmup_batches=warmup_batches,
        epochs=epochs,
    )


def _advance_scheduler(trainer, steps):
    for _ in range(steps):
        trainer.global_step += 1
        trainer._step_scheduler_after_batch()


@pytest.mark.parametrize("legacy", [False, True], ids=["new", "legacy"])
@pytest.mark.parametrize(
    ("warmup_batches", "completed_steps"),
    [(0, 17), (5, 2), (5, 17), (5, 55)],
    ids=["no-warmup", "during-warmup", "after-warmup", "finished"],
)
def test_cosine_progress_survives_resume(tmp_path, legacy, warmup_batches, completed_steps):
    uninterrupted = _make_trainer(warmup_batches=warmup_batches)
    _advance_scheduler(uninterrupted, completed_steps)
    checkpoint = tmp_path / "checkpoint.pth"
    uninterrupted.save_checkpoint(checkpoint, epoch=1)
    if legacy:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload.pop("scheduler_progress_state", None)
        torch.save(payload, checkpoint)

    resumed = _make_trainer(warmup_batches=warmup_batches)
    assert resumed.load_checkpoint(checkpoint, training_state=True) == 2
    assert resumed.global_step == uninterrupted.global_step
    assert resumed._post_warmup_step == uninterrupted._post_warmup_step
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(
        uninterrupted.optimizer.param_groups[0]["lr"]
    )
    for _ in range(3):
        _advance_scheduler(uninterrupted, 1)
        _advance_scheduler(resumed, 1)
        assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(
            uninterrupted.optimizer.param_groups[0]["lr"]
        )


def test_cosine_resume_preserves_saved_horizon(tmp_path):
    uninterrupted = _make_trainer()
    _advance_scheduler(uninterrupted, 17)
    checkpoint = tmp_path / "checkpoint.pth"
    uninterrupted.save_checkpoint(checkpoint, epoch=1)

    resumed = _make_trainer(epochs=9)
    resumed.load_checkpoint(checkpoint, training_state=True)
    assert resumed._cosine_total_steps == uninterrupted._cosine_total_steps
    _advance_scheduler(uninterrupted, 1)
    _advance_scheduler(resumed, 1)
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(
        uninterrupted.optimizer.param_groups[0]["lr"]
    )


def test_training_resume_rejects_scheduler_kind_change(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    _make_trainer().save_checkpoint(checkpoint)
    resumed = _make_trainer(lr_scheduler="step")
    with pytest.raises(ValueError, match="scheduler kind"):
        resumed.load_checkpoint(checkpoint, training_state=True)


def test_weight_only_load_keeps_fresh_scheduler(tmp_path):
    original = _make_trainer()
    _advance_scheduler(original, 17)
    checkpoint = tmp_path / "checkpoint.pth"
    original.save_checkpoint(checkpoint)

    resumed = _make_trainer(lr_scheduler="step")
    assert resumed.load_checkpoint(checkpoint, training_state=False) == 0
    assert resumed.global_step == 0
    assert resumed._post_warmup_step == 0
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
