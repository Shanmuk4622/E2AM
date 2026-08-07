"""The E2AM Trainer: a drop-in training loop with automatic Green AI telemetry.

Design: the Trainer is an *orchestrator*. The training loop itself is
deliberately standard PyTorch (forward → loss → backward → step, with AMP,
gradient accumulation, and clipping); everything E2AM-specific — energy
monitoring, model profiling, quality metrics, green metrics — is delegated
to the modules that own those concerns. Extension happens exclusively
through :class:`~e2am.trainer.callbacks.Callback`.

Example:
    >>> from e2am import Trainer
    >>> trainer = Trainer(model=model, optimizer=optimizer,
    ...                   train_loader=train_loader, val_loader=val_loader)
    >>> result = trainer.fit()
    >>> print(result.green.green_score, result.total_energy_wh)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn

from e2am.config.settings import ExperimentConfig, MonitorConfig, TrainerConfig
from e2am.exceptions import TrainerError
from e2am.metrics.classification import classification_metrics
from e2am.metrics.green import compute_green_metrics
from e2am.metrics.tracker import MetricsTracker
from e2am.monitoring.session import MonitorSession
from e2am.trainer.callbacks import Callback, CallbackRunner, LoggingCallback
from e2am.trainer.result import TrainingResult
from e2am.utils.logging import get_logger

logger = get_logger("trainer")


def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors (in tuples/lists/dicts) to a device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    return obj


class Trainer:
    """Train a PyTorch model with automatic energy/carbon/quality tracking.

    Args:
        model: The model to train.
        optimizer: A configured optimizer over ``model.parameters()``.
        train_loader: Iterable of ``(inputs, targets)`` batches.
        val_loader: Optional validation loader of ``(inputs, targets)``.
        loss_fn: Loss callable ``(outputs, targets) -> scalar tensor``.
            Defaults to :class:`torch.nn.CrossEntropyLoss`.
        config: Full :class:`ExperimentConfig`; individual keyword arguments
            below override its fields when given.
        project: Project name grouping related runs.
        run_name: Run identifier; auto-generated when omitted.
        epochs: Number of training epochs.
        device: Target device (``"cuda"``, ``"cuda:1"``, ``"cpu"``);
            auto-detected when omitted.
        mixed_precision: Enable AMP (autocast + GradScaler) on CUDA.
        scheduler: Optional LR scheduler stepped once per epoch.
        callbacks: Additional :class:`Callback` instances.
        task: ``"classification"`` computes accuracy/precision/recall/F1 on
            validation; ``"custom"`` computes only losses.
        monitor_enabled: Record energy/carbon/utilization during training.
        profile_enabled: Profile params/MACs/FLOPs from the first batch.
        progress: Log an epoch summary line after each epoch.
        save_artifacts: Write ``metrics.json``/``summary.yaml``/``config.yaml``
            into ``results/<run_name>/`` after training.

    Raises:
        TrainerError: On invalid configuration (e.g. unknown task).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable,
        val_loader: Iterable | None = None,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        config: ExperimentConfig | None = None,
        project: str | None = None,
        run_name: str | None = None,
        epochs: int | None = None,
        device: str | torch.device | None = None,
        mixed_precision: bool | None = None,
        scheduler: Any | None = None,
        callbacks: list[Callback] | None = None,
        task: str = "classification",
        monitor_enabled: bool = True,
        profile_enabled: bool = True,
        progress: bool = True,
        save_artifacts: bool = True,
    ) -> None:
        if task not in ("classification", "custom"):
            raise TrainerError(f"task must be 'classification' or 'custom', got {task!r}.")

        self.config = config or ExperimentConfig()
        if project is not None:
            self.config.project = project
        if run_name is not None:
            self.config.run_name = run_name
        trainer_cfg: TrainerConfig = self.config.trainer
        if epochs is not None:
            trainer_cfg.epochs = epochs
        if device is not None:
            trainer_cfg.device = str(device)
        if mixed_precision is not None:
            trainer_cfg.mixed_precision = mixed_precision

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.task = task
        self.monitor_enabled = monitor_enabled
        self.profile_enabled = profile_enabled
        self.save_artifacts = save_artifacts

        self.device = self._resolve_device(trainer_cfg.device)
        self.epochs = trainer_cfg.epochs
        self.run_name = self.config.resolved_run_name()
        self.config.run_name = self.run_name

        user_callbacks = list(callbacks) if callbacks else []
        if progress and not any(isinstance(c, LoggingCallback) for c in user_callbacks):
            user_callbacks.append(LoggingCallback())
        self._runner = CallbackRunner(user_callbacks)

        #: Set to ``True`` (e.g. by EarlyStopping) to stop after this epoch.
        self.should_stop = False
        #: Per-epoch metric series; callbacks may read or extend it.
        self.tracker = MetricsTracker()
        #: Live monitoring session while ``fit()`` runs (else ``None``).
        self.monitor_session: MonitorSession | None = None
        self.result: TrainingResult | None = None

        use_amp = trainer_cfg.mixed_precision and self.device.type == "cuda"
        if trainer_cfg.mixed_precision and not use_amp:
            logger.warning("mixed_precision requested but device is %s; disabled.", self.device)
        self._use_amp = use_amp
        self._scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self._accum_steps = trainer_cfg.gradient_accumulation_steps
        self._max_grad_norm = trainer_cfg.max_grad_norm

    @staticmethod
    def _resolve_device(requested: str | None) -> torch.device:
        if requested:
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> TrainingResult:
        """Run the full training loop and return the assembled result."""
        self.model.to(self.device)
        self.should_stop = False
        run_dir = self.config.output.dir / self.run_name

        if self.monitor_enabled:
            self.monitor_session = MonitorSession(
                config=self.config.monitor or MonitorConfig(),
                project=self.config.project,
                run_name=self.run_name,
            )
            self.monitor_session.start()

        logger.info(
            "Training %s for %d epoch(s) on %s (AMP=%s).",
            type(self.model).__name__,
            self.epochs,
            self.device,
            self._use_amp,
        )

        profile_dump: dict[str, Any] | None = None
        status = "completed"
        epochs_completed = 0
        samples_processed = 0
        batch_times_ms: list[float] = []
        epoch_times_s: list[float] = []
        val_accuracies: list[float] = []
        cumulative_energy_wh: list[float] = []
        best_val_acc: float | None = None
        best_epoch: int | None = None
        last_val_metrics: dict[str, float] = {}

        self._runner.fire("on_fit_start", self)
        try:
            for epoch in range(self.epochs):
                self._runner.fire("on_epoch_start", self, epoch)
                epoch_start = time.perf_counter()

                train_loss, epoch_samples, epoch_batch_times = self._train_epoch(epoch)
                if profile_dump is None and self.profile_enabled:
                    profile_dump = self._profile_from_loader()
                samples_processed += epoch_samples
                batch_times_ms.extend(epoch_batch_times)

                logs: dict[str, float] = {"train_loss": train_loss}
                if self.val_loader is not None:
                    val_logs = self.evaluate(self.val_loader)
                    logs.update(val_logs)
                    last_val_metrics = val_logs
                    accuracy = val_logs.get("val_accuracy")
                    if accuracy is not None:
                        val_accuracies.append(accuracy)
                        if best_val_acc is None or accuracy > best_val_acc:
                            best_val_acc, best_epoch = accuracy, epoch

                epoch_time = time.perf_counter() - epoch_start
                epoch_times_s.append(epoch_time)
                logs["epoch_time_s"] = epoch_time
                if self.monitor_session is not None:
                    snapshot = self.monitor_session.snapshot()
                    energy_wh = float(snapshot["total_energy_j"]) / 3600.0
                    cumulative_energy_wh.append(energy_wh)
                    logs["cumulative_energy_wh"] = energy_wh
                if self.scheduler is not None:
                    logs["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
                    self.scheduler.step()

                self.tracker.log(logs, step=epoch)
                self._runner.fire("on_epoch_end", self, epoch, logs)
                epochs_completed = epoch + 1

                if self.should_stop:
                    status = "stopped"
                    break
        except BaseException as exc:
            status = "failed"
            self._runner.fire("on_exception", self, exc)
            raise
        finally:
            monitor_result = self.monitor_session.stop() if self.monitor_session else None
            self.monitor_session = None
            if monitor_result is not None:
                monitor_result.status = "completed" if status != "failed" else "failed"

            green = None
            if monitor_result is not None:
                green = compute_green_metrics(
                    energy_j=monitor_result.total_energy_j,
                    num_samples=samples_processed or None,
                    accuracy=best_val_acc,
                    emissions_g=monitor_result.carbon.emissions_g,
                    epoch_accuracies=val_accuracies or None,
                    epoch_cumulative_energy_wh=cumulative_energy_wh or None,
                )

            total_train_time = sum(epoch_times_s)
            self.result = TrainingResult(
                project=self.config.project,
                run_name=self.run_name,
                status=status,
                epochs_requested=self.epochs,
                epochs_completed=epochs_completed,
                device=str(self.device),
                mixed_precision=self._use_amp,
                samples_processed=samples_processed,
                best_val_accuracy=best_val_acc,
                best_epoch=best_epoch,
                final_train_loss=self.tracker.latest("train_loss"),
                final_val_loss=last_val_metrics.get("val_loss"),
                final_val_accuracy=last_val_metrics.get("val_accuracy"),
                final_val_f1_macro=last_val_metrics.get("val_f1_macro"),
                avg_batch_time_ms=(
                    sum(batch_times_ms) / len(batch_times_ms) if batch_times_ms else None
                ),
                avg_epoch_time_s=(
                    sum(epoch_times_s) / len(epoch_times_s) if epoch_times_s else None
                ),
                train_throughput_samples_per_s=(
                    samples_processed / total_train_time if total_train_time > 0 else None
                ),
                monitor=monitor_result,
                profile=profile_dump,
                green=green,
                history=self.tracker.to_dict(),
            )
            if self.save_artifacts:
                try:
                    self.result.save(run_dir)
                    self.config.to_yaml(run_dir / "config.yaml")
                    from e2am.reports.generate import generate_run_artifacts

                    generate_run_artifacts(self.result, run_dir, self.config.output)
                    logger.info("Run artifacts saved to %s", run_dir)
                except Exception as save_exc:
                    logger.warning("Could not save run artifacts: %s", save_exc)
            self._runner.fire("on_fit_end", self)

        return self.result

    def evaluate(self, loader: Iterable | None = None) -> dict[str, float]:
        """Evaluate the model on a loader (default: the validation loader).

        Returns:
            ``{"val_loss": ...}`` plus, for classification tasks,
            ``val_accuracy``, ``val_precision_macro``, ``val_recall_macro``,
            and ``val_f1_macro``.

        Raises:
            TrainerError: If no loader is available.
        """
        loader = loader if loader is not None else self.val_loader
        if loader is None:
            raise TrainerError("evaluate() needs a loader (none was provided).")

        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions: list[torch.Tensor] = []
        targets_seen: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                for batch in loader:
                    inputs, targets = self._split_batch(batch)
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    batch_size = int(targets.shape[0]) if targets.dim() else 1
                    total_loss += float(loss.detach()) * batch_size
                    total_samples += batch_size
                    if self.task == "classification":
                        predictions.append(outputs.detach().cpu())
                        targets_seen.append(targets.detach().cpu())
        finally:
            self.model.train(was_training)

        if total_samples == 0:
            raise TrainerError("evaluate() received an empty loader.")
        logs = {"val_loss": total_loss / total_samples}
        if self.task == "classification" and predictions:
            metrics = classification_metrics(torch.cat(predictions), torch.cat(targets_seen))
            logs.update(
                val_accuracy=metrics.accuracy,
                val_precision_macro=metrics.precision_macro,
                val_recall_macro=metrics.recall_macro,
                val_f1_macro=metrics.f1_macro,
            )
        return logs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _split_batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise TrainerError(
                "Batches must be (inputs, targets) pairs; got "
                f"{type(batch).__name__}. Use task='custom' with your own "
                "loop and e2am.monitor() for exotic batch formats."
            )
        moved = _move_to_device(batch, self.device)
        return moved[0], moved[1]

    def _optimizer_step(self) -> None:
        """Clip, step, and reset gradients for one accumulation window.

        Shared by the in-loop path and the trailing-window flush so gradient
        clipping cannot be skipped when the batch count is not a multiple of
        ``gradient_accumulation_steps``.
        """
        if self._max_grad_norm is not None:
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
        self._scaler.step(self.optimizer)
        self._scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def _train_epoch(self, epoch: int) -> tuple[float, int, list[float]]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        batch_times_ms: list[float] = []
        batch_idx = -1
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(self.train_loader):
            self._runner.fire("on_batch_start", self, batch_idx)
            batch_start = time.perf_counter()

            inputs, targets = self._split_batch(batch)
            with torch.amp.autocast(self.device.type, enabled=self._use_amp):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
            scaled = loss / self._accum_steps
            self._scaler.scale(scaled).backward()

            if (batch_idx + 1) % self._accum_steps == 0:
                self._optimizer_step()

            batch_size = int(targets.shape[0]) if targets.dim() else 1
            batch_loss = float(loss.detach())
            total_loss += batch_loss * batch_size
            total_samples += batch_size
            batch_time_ms = (time.perf_counter() - batch_start) * 1000.0
            batch_times_ms.append(batch_time_ms)
            self._runner.fire(
                "on_batch_end",
                self,
                batch_idx,
                {"loss": batch_loss, "batch_time_ms": batch_time_ms},
            )

        # Flush a trailing partial accumulation window (same clipping path).
        if total_samples and (batch_idx + 1) % self._accum_steps != 0:
            self._optimizer_step()

        if total_samples == 0:
            raise TrainerError("train_loader yielded no batches.")
        return total_loss / total_samples, total_samples, batch_times_ms

    def _profile_from_loader(self) -> dict[str, Any] | None:
        """Profile params/MACs from the first training batch shape."""
        try:
            first_batch = next(iter(self.train_loader))
            inputs, _ = self._split_batch(first_batch)
            if not isinstance(inputs, torch.Tensor):
                return None
            from e2am.profiler.flops import profile_model

            profile = profile_model(self.model, sample_input=inputs)
            logger.info(
                "Model profile: %.2fM params, %.3f GMACs, coverage %.0f%%.",
                profile.params / 1e6,
                profile.gmacs,
                profile.coverage * 100,
            )
            return profile.model_dump()
        except Exception as exc:
            logger.warning("Model profiling skipped: %s", exc)
            return None
