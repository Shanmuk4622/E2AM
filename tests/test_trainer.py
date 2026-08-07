"""Trainer integration tests on tiny synthetic data (fast, deterministic)."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from e2am.exceptions import TrainerError  # noqa: E402
from e2am.trainer import Callback, EarlyStopping, Trainer, TrainingResult  # noqa: E402

pytestmark = pytest.mark.torch


def _make_loaders(n: int = 128, batch_size: int = 32) -> tuple[DataLoader, DataLoader]:
    """Linearly separable 2-class blobs -> a linear model must learn them."""
    generator = torch.Generator().manual_seed(7)
    x0 = torch.randn(n // 2, 4, generator=generator) - 2.0
    x1 = torch.randn(n // 2, 4, generator=generator) + 2.0
    x = torch.cat([x0, x1])
    y = torch.cat([torch.zeros(n // 2, dtype=torch.long), torch.ones(n // 2, dtype=torch.long)])
    dataset = TensorDataset(x, y)
    return (
        DataLoader(dataset, batch_size=batch_size, shuffle=True),
        DataLoader(dataset, batch_size=batch_size),
    )


def _make_trainer(tmp_path: Path, **kwargs) -> Trainer:
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    defaults = {
        "model": model,
        "optimizer": torch.optim.SGD(model.parameters(), lr=0.1),
        "epochs": 3,
        "device": "cpu",
        "progress": False,
    }
    defaults.update(kwargs)
    trainer = Trainer(**defaults)
    trainer.config.output.dir = tmp_path
    trainer.config.monitor.sampling_interval_s = 0.05
    return trainer


def test_fit_learns_and_populates_result(tmp_path: Path) -> None:
    train_loader, val_loader = _make_loaders()
    trainer = _make_trainer(tmp_path, train_loader=train_loader, val_loader=val_loader)
    result = trainer.fit()

    assert result.status == "completed"
    assert result.epochs_completed == 3
    assert result.samples_processed == 3 * 128
    assert result.final_val_accuracy is not None and result.final_val_accuracy > 0.9
    assert result.best_val_accuracy >= result.final_val_accuracy - 1e-9
    losses = result.history["train_loss"]["values"]
    assert losses[-1] < losses[0]  # actually learned
    # monitoring + green metrics wired in
    assert result.monitor is not None and result.monitor.total_energy_j > 0
    assert result.green is not None
    assert result.green.energy_per_sample_j is not None
    assert result.green.green_score is not None
    # profiling captured the linear layer
    assert result.profile is not None
    assert result.profile["params"] == 4 * 2 + 2
    # timing stats present
    assert result.avg_batch_time_ms is not None and result.avg_batch_time_ms > 0
    assert result.train_throughput_samples_per_s is not None


def test_artifacts_saved_and_result_reloads(tmp_path: Path) -> None:
    train_loader, val_loader = _make_loaders(n=64)
    trainer = _make_trainer(
        tmp_path, train_loader=train_loader, val_loader=val_loader, run_name="run-a", epochs=1
    )
    trainer.fit()
    run_dir = tmp_path / "run-a"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "summary.yaml").exists()
    assert (run_dir / "config.yaml").exists()
    reloaded = TrainingResult.load(run_dir)
    assert reloaded.run_name == "run-a"
    assert reloaded.to_flat_dict()["epochs_completed"] == 1


def test_callback_lifecycle_order(tmp_path: Path) -> None:
    events: list[str] = []

    class Recorder(Callback):
        def on_fit_start(self, trainer):
            events.append("fit_start")

        def on_epoch_start(self, trainer, epoch):
            events.append(f"epoch_start:{epoch}")

        def on_batch_end(self, trainer, batch_idx, logs):
            events.append("batch_end")

        def on_epoch_end(self, trainer, epoch, logs):
            events.append(f"epoch_end:{epoch}")
            assert "train_loss" in logs and "epoch_time_s" in logs

        def on_fit_end(self, trainer):
            events.append("fit_end")

    train_loader, _ = _make_loaders(n=64, batch_size=32)
    trainer = _make_trainer(
        tmp_path,
        train_loader=train_loader,
        epochs=2,
        callbacks=[Recorder()],
        monitor_enabled=False,
        profile_enabled=False,
        save_artifacts=False,
    )
    trainer.fit()
    assert events[0] == "fit_start"
    assert events[-1] == "fit_end"
    assert events.count("epoch_start:0") == 1
    assert events.count("epoch_end:1") == 1
    assert events.count("batch_end") == 2 * 2  # 2 epochs x 2 batches


def test_broken_callback_does_not_kill_training(tmp_path: Path) -> None:
    class Broken(Callback):
        def on_epoch_end(self, trainer, epoch, logs):
            raise RuntimeError("callback exploded")

    train_loader, _ = _make_loaders(n=32)
    trainer = _make_trainer(
        tmp_path,
        train_loader=train_loader,
        epochs=1,
        callbacks=[Broken()],
        monitor_enabled=False,
        save_artifacts=False,
    )
    result = trainer.fit()
    assert result.status == "completed"


def test_early_stopping_stops(tmp_path: Path) -> None:
    train_loader, val_loader = _make_loaders(n=64)
    stopper = EarlyStopping(monitor="val_loss", patience=0, min_delta=10.0)  # can't improve
    trainer = _make_trainer(
        tmp_path,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        callbacks=[stopper],
        monitor_enabled=False,
        save_artifacts=False,
    )
    result = trainer.fit()
    assert result.status == "stopped"
    assert result.epochs_completed < 10
    assert stopper.stopped_epoch is not None


def test_exception_marks_failed_and_propagates(tmp_path: Path) -> None:
    class Bomb(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if self.calls > 2:
                raise ValueError("model exploded")
            return self.linear(x)

    train_loader, _ = _make_loaders(n=128)
    model = Bomb()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=train_loader,
        epochs=2,
        device="cpu",
        progress=False,
        profile_enabled=False,
        save_artifacts=False,
    )
    trainer.config.output.dir = tmp_path
    trainer.config.monitor.sampling_interval_s = 0.05
    with pytest.raises(ValueError, match="model exploded"):
        trainer.fit()
    assert trainer.result is not None
    assert trainer.result.status == "failed"
    assert trainer.result.monitor is not None  # monitoring was stopped cleanly


def test_gradient_accumulation_and_clipping(tmp_path: Path) -> None:
    train_loader, val_loader = _make_loaders(n=96, batch_size=16)
    trainer = _make_trainer(
        tmp_path,
        train_loader=train_loader,
        val_loader=val_loader,
        monitor_enabled=False,
        save_artifacts=False,
    )
    trainer._accum_steps = 2
    trainer._max_grad_norm = 1.0
    result = trainer.fit()
    losses = result.history["train_loss"]["values"]
    assert losses[-1] < losses[0]


def test_clipping_applied_to_trailing_accumulation_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial final window must be clipped like every other window.

    3 batches with accum_steps=2 means one in-loop step (batches 0-1) and one
    trailing flush (batch 2). Both must clip; the flush path used to step the
    optimizer with unscaled, unclipped gradients.
    """
    calls: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        calls.append(max_norm)
        return real_clip(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

    # 96 samples / batch 32 = 3 batches per epoch, accum_steps=2 -> one partial.
    train_loader, _ = _make_loaders(n=96, batch_size=32)
    trainer = _make_trainer(
        tmp_path,
        train_loader=train_loader,
        epochs=1,
        monitor_enabled=False,
        profile_enabled=False,
        save_artifacts=False,
    )
    trainer._accum_steps = 2
    trainer._max_grad_norm = 1.0
    trainer.fit()

    assert len(calls) == 2, f"expected clipping on both windows, got {len(calls)}"
    assert all(norm == 1.0 for norm in calls)


def test_invalid_task_raises(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    with pytest.raises(TrainerError, match="task"):
        Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            train_loader=[],
            task="segmentation",
        )


def test_bad_batch_format_raises(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[torch.randn(4, 2)],  # not (inputs, targets)
        epochs=1,
        device="cpu",
        progress=False,
        monitor_enabled=False,
        profile_enabled=False,
        save_artifacts=False,
    )
    with pytest.raises(TrainerError, match="inputs, targets"):
        trainer.fit()


def test_evaluate_without_loader_raises(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_loader=[],
        device="cpu",
        progress=False,
        monitor_enabled=False,
        save_artifacts=False,
    )
    with pytest.raises(TrainerError, match="loader"):
        trainer.evaluate()


@pytest.mark.gpu
def test_gpu_training_with_amp(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    train_loader, val_loader = _make_loaders(n=128)
    trainer = _make_trainer(
        tmp_path,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda",
        mixed_precision=True,
        epochs=2,
    )
    result = trainer.fit()
    assert result.status == "completed"
    assert result.final_val_accuracy is not None and result.final_val_accuracy > 0.9
    assert result.monitor is not None
    gpu_devices = [d for d in result.monitor.devices if d.name.startswith("gpu")]
    assert gpu_devices, "GPU sampler should be active during CUDA training"
