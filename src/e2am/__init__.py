"""E2AM — Energy Efficient AI Models.

Automatic energy, carbon, and performance profiling for AI training with
almost zero code changes.

Example:
    >>> from e2am import monitor
    >>> with monitor(project="ResNet50"):
    ...     train()

    >>> from e2am import Trainer
    >>> trainer = Trainer(model=model, optimizer=optimizer,
    ...                   train_loader=train_loader, val_loader=val_loader)
    >>> trainer.fit()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

__all__ = ["Trainer", "__version__", "monitor"]

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from e2am.monitoring import monitor
    from e2am.trainer import Trainer


def __getattr__(name: str) -> Any:
    """Lazily resolve the public API.

    Keeps ``import e2am`` fast and torch-free: the ``Trainer`` (which needs
    PyTorch) is only imported when actually accessed, so monitoring-only
    installs work on machines without PyTorch.
    """
    if name == "monitor":
        from e2am.monitoring import monitor

        return monitor
    if name == "Trainer":
        from e2am.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
