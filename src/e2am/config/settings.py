"""Experiment configuration models.

Every knob in E2AM lives in a pydantic model here, giving three things at
once: validation with helpful errors, YAML round-tripping for reproducible
runs (``config.yaml`` is written into every results directory), and a single
documented source of truth for defaults.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from e2am.exceptions import ConfigError
from e2am.schema import SCHEMA_VERSION, migrate

#: Global average grid carbon intensity in gCO2eq/kWh (IEA 2023 estimate).
WORLD_AVG_CARBON_INTENSITY = 475.0


class CarbonConfig(BaseModel):
    """How carbon emissions are computed from energy."""

    country_iso_code: str | None = Field(
        default=None,
        description="ISO 3166 country code (e.g. 'IND', 'USA') for region-aware intensity.",
    )
    carbon_intensity_g_per_kwh: float | None = Field(
        default=None,
        gt=0,
        description="Explicit grid carbon intensity in gCO2eq/kWh. When set it "
        "wins over the country lookup; None means 'not specified', so the "
        "country code (else the world average) decides.",
    )


class MonitorConfig(BaseModel):
    """Background monitoring behavior."""

    sampling_interval_s: float = Field(
        default=1.0,
        ge=0.05,
        le=60.0,
        description="Seconds between hardware samples.",
    )
    gpu_indices: list[int] | None = Field(
        default=None,
        description="GPUs to monitor (None = all detected GPUs).",
    )
    cpu_tdp_w: float | None = Field(
        default=None,
        gt=0,
        description="CPU TDP override in watts for energy estimation.",
    )
    carbon: CarbonConfig = Field(default_factory=CarbonConfig)


class OutputConfig(BaseModel):
    """What gets written to disk after a run."""

    dir: Path = Field(default=Path("results"), description="Root results directory.")
    save_plots: bool = True
    save_html: bool = True
    save_pdf: bool = False
    save_json: bool = True
    save_csv: bool = True

    @field_validator("dir", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        return Path(value)


class TrainerConfig(BaseModel):
    """Training loop behavior for :class:`e2am.Trainer`."""

    epochs: int = Field(default=10, ge=1)
    device: str | None = Field(
        default=None,
        description="Target device ('cuda', 'cuda:1', 'cpu'). None = auto-detect.",
    )
    mixed_precision: bool = Field(
        default=False,
        description="Enable automatic mixed precision (torch.autocast + GradScaler).",
    )
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_grad_norm: float | None = Field(
        default=None,
        gt=0,
        description="Clip gradients to this norm when set.",
    )
    log_every_n_steps: int = Field(default=50, ge=1)


class ExperimentConfig(BaseModel):
    """Top-level configuration for one experiment run."""

    schema_version: int = Field(
        default=SCHEMA_VERSION,
        description="Artifact schema version; see e2am.schema for migrations.",
    )
    project: str = Field(default="e2am", min_length=1)
    run_name: str | None = Field(
        default=None,
        description="Run identifier. None = auto ('<project>-<timestamp>').",
    )
    seed: int | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)

    def resolved_run_name(self) -> str:
        """Return the explicit run name or generate a timestamped one."""
        if self.run_name:
            return self.run_name
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.project}-{stamp}"

    def run_dir(self) -> Path:
        """Directory where this run's artifacts are written."""
        return self.output.dir / self.resolved_run_name()

    # ------------------------------------------------------------------
    # YAML round-tripping
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a configuration from a YAML file.

        Args:
            path: Path to a YAML file produced by :meth:`to_yaml` or written
                by hand.

        Configs written by older E2AM releases are migrated forward on load
        (see :mod:`e2am.schema`), preserving the meaning they were written
        with.

        Raises:
            ConfigError: If the file is missing, unparsable, invalid, or was
                written by a newer E2AM than this one.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Invalid configuration in {path}: expected a YAML mapping.")
        data = migrate(data, "config")
        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise ConfigError(f"Invalid configuration in {path}: {exc}") from exc

    def to_yaml(self, path: str | Path) -> Path:
        """Write this configuration to a YAML file and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        return path


def load_config(path: str | Path) -> ExperimentConfig:
    """Convenience wrapper for :meth:`ExperimentConfig.from_yaml`."""
    return ExperimentConfig.from_yaml(path)
