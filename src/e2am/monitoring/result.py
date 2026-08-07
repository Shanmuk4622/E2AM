"""Result models produced by a monitoring session.

Everything is a pydantic model so a full run — including per-device energy,
utilization statistics, and raw time series — serializes losslessly to
``metrics.json`` and back, which the reports/visualization modules rely on.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from e2am.monitoring.carbon import CarbonResult
from e2am.monitoring.energy import joules_to_kwh, joules_to_wh
from e2am.schema import SCHEMA_VERSION
from e2am.utils.hardware import SystemInfo


class DeviceEnergy(BaseModel):
    """Energy and power statistics for one monitored device."""

    name: str
    energy_j: float = 0.0
    avg_power_w: float = 0.0
    max_power_w: float = 0.0
    power_estimated: bool = Field(
        default=True,
        description="True when power was estimated (TDP x utilization) "
        "rather than read from a hardware sensor.",
    )

    @property
    def energy_wh(self) -> float:
        """Energy in watt-hours."""
        return joules_to_wh(self.energy_j)


class UtilizationStats(BaseModel):
    """Utilization summary for one device."""

    avg_pct: float = 0.0
    max_pct: float = 0.0


class TimeSeriesData(BaseModel):
    """Raw sampled series, aligned on a shared relative-time axis."""

    timestamps_s: list[float] = Field(default_factory=list)
    power_w: dict[str, list[float | None]] = Field(default_factory=dict)
    utilization_pct: dict[str, list[float | None]] = Field(default_factory=dict)
    memory_used_mb: dict[str, list[float | None]] = Field(default_factory=dict)
    temperature_c: dict[str, list[float | None]] = Field(default_factory=dict)


class MonitorResult(BaseModel):
    """Complete outcome of one monitoring session."""

    schema_version: int = Field(
        default=SCHEMA_VERSION,
        description="Artifact schema version; see e2am.schema for migrations.",
    )
    project: str = "e2am"
    run_name: str = ""
    status: str = Field(default="completed", description="'completed' or 'failed'.")
    started_at: datetime
    ended_at: datetime
    duration_s: float = 0.0
    sampling_interval_s: float = 1.0
    num_samples: int = 0
    system: SystemInfo = Field(default_factory=SystemInfo)
    devices: list[DeviceEnergy] = Field(default_factory=list)
    utilization: dict[str, UtilizationStats] = Field(default_factory=dict)
    peak_memory_mb: dict[str, float] = Field(default_factory=dict)
    carbon: CarbonResult = Field(default_factory=CarbonResult)
    timeseries: TimeSeriesData = Field(default_factory=TimeSeriesData)

    # ------------------------------------------------------------------
    # Convenience aggregations
    # ------------------------------------------------------------------

    @property
    def total_energy_j(self) -> float:
        """Total energy across all devices in joules."""
        return sum(d.energy_j for d in self.devices)

    @property
    def total_energy_wh(self) -> float:
        """Total energy across all devices in watt-hours."""
        return joules_to_wh(self.total_energy_j)

    @property
    def total_energy_kwh(self) -> float:
        """Total energy across all devices in kilowatt-hours."""
        return joules_to_kwh(self.total_energy_j)

    def _energy_wh_for(self, prefix: str) -> float:
        return sum(d.energy_wh for d in self.devices if d.name.startswith(prefix))

    @property
    def gpu_energy_wh(self) -> float:
        """Energy of all GPUs combined, in watt-hours."""
        return self._energy_wh_for("gpu")

    @property
    def cpu_energy_wh(self) -> float:
        """CPU energy in watt-hours."""
        return self._energy_wh_for("cpu")

    @property
    def ram_energy_wh(self) -> float:
        """RAM energy in watt-hours."""
        return self._energy_wh_for("ram")

    @property
    def avg_total_power_w(self) -> float:
        """Average combined power draw over the run, in watts."""
        if self.duration_s <= 0:
            return 0.0
        return self.total_energy_j / self.duration_s

    def to_flat_dict(self) -> dict[str, object]:
        """Flatten headline numbers for leaderboards and CSV export."""
        flat: dict[str, object] = {
            "project": self.project,
            "run_name": self.run_name,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "duration_s": round(self.duration_s, 3),
            "total_energy_wh": round(self.total_energy_wh, 6),
            "gpu_energy_wh": round(self.gpu_energy_wh, 6),
            "cpu_energy_wh": round(self.cpu_energy_wh, 6),
            "ram_energy_wh": round(self.ram_energy_wh, 6),
            "avg_power_w": round(self.avg_total_power_w, 3),
            "emissions_g": round(self.carbon.emissions_g, 6),
            "carbon_intensity_g_per_kwh": self.carbon.intensity_g_per_kwh,
        }
        for name, stats in sorted(self.utilization.items()):
            flat[f"avg_{name}_util_pct"] = round(stats.avg_pct, 2)
        for name, peak in sorted(self.peak_memory_mb.items()):
            flat[f"peak_{name}_memory_mb"] = round(peak, 2)
        return flat
