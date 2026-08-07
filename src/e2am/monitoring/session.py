"""Background monitoring session.

A :class:`MonitorSession` polls all available samplers from a daemon thread
at a fixed interval, integrates power into energy on the fly, and assembles a
:class:`~e2am.monitoring.result.MonitorResult` on stop. It is designed to sit
underneath a training loop for hours: sampler failures are swallowed and
logged, the thread wakes promptly on stop, and a live :meth:`snapshot` lets
callbacks read running totals (e.g. per-epoch energy) without stopping.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from e2am.config.settings import MonitorConfig
from e2am.exceptions import MonitorError
from e2am.monitoring.carbon import CarbonEstimator
from e2am.monitoring.energy import EnergyIntegrator, joules_to_kwh
from e2am.monitoring.result import (
    DeviceEnergy,
    MonitorResult,
    TimeSeriesData,
    UtilizationStats,
)
from e2am.monitoring.samplers import Sampler, create_samplers
from e2am.utils.hardware import SystemInfo, detect_hardware
from e2am.utils.logging import get_logger

logger = get_logger("monitor.session")


class MonitorSession:
    """Continuously samples hardware in the background.

    Args:
        config: Monitoring configuration. Defaults to :class:`MonitorConfig`.
        project: Project name recorded in the result.
        run_name: Run identifier recorded in the result.
        samplers: Explicit samplers (dependency injection for tests). When
            ``None``, samplers are auto-created from detected hardware.

    Example:
        >>> session = MonitorSession(project="demo", run_name="run-1")
        >>> session.start()
        >>> train()
        >>> result = session.stop()
        >>> print(result.total_energy_wh)
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        project: str = "e2am",
        run_name: str = "",
        samplers: list[Sampler] | None = None,
    ) -> None:
        self.config = config or MonitorConfig()
        self.project = project
        self.run_name = run_name
        self._injected_samplers = samplers
        self._samplers: list[Sampler] = []
        self._integrators: dict[str, EnergyIntegrator] = {}
        self._series = TimeSeriesData()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0_monotonic = 0.0
        self._started_at: datetime | None = None
        self._running = False
        self.system = SystemInfo()

    @property
    def is_running(self) -> bool:
        """Whether the background sampling thread is active."""
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> MonitorSession:
        """Detect hardware, set up samplers, and launch the sampling thread.

        Returns:
            The session itself, for chaining.

        Raises:
            MonitorError: If the session is already running or no sampler
                could be initialized.
        """
        if self._running:
            raise MonitorError("MonitorSession is already running.")

        self.system = detect_hardware(self.config.cpu_tdp_w)
        if self._injected_samplers is not None:
            self._samplers = list(self._injected_samplers)
        else:
            self._samplers = create_samplers(self.config)
        if not self._samplers:
            raise MonitorError("No hardware sampler could be initialized on this machine.")

        for sampler in self._samplers:
            self._integrators[sampler.name] = EnergyIntegrator()
            self._series.power_w[sampler.name] = []
            self._series.utilization_pct[sampler.name] = []
            self._series.memory_used_mb[sampler.name] = []
            self._series.temperature_c[sampler.name] = []

        self._started_at = datetime.now(timezone.utc)
        self._t0_monotonic = time.monotonic()
        self._stop_event.clear()
        self._tick()  # record the t=0 baseline sample
        self._thread = threading.Thread(target=self._loop, name="e2am-monitor", daemon=True)
        self._running = True
        self._thread.start()
        logger.info(
            "Monitoring started (%s) at %.1fs interval.",
            ", ".join(s.name for s in self._samplers),
            self.config.sampling_interval_s,
        )
        return self

    def stop(self) -> MonitorResult:
        """Stop sampling and return the assembled result.

        Raises:
            MonitorError: If the session was never started.
        """
        if self._started_at is None:
            raise MonitorError("MonitorSession was never started.")
        if self._running:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=self.config.sampling_interval_s + 5.0)
            self._tick()  # final sample so integration covers the full span
            self._running = False
        for sampler in self._samplers:
            try:
                sampler.teardown()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Sampler %s teardown failed: %s", sampler.name, exc)
        result = self._build_result()
        logger.info(
            "Monitoring stopped: %.2f Wh over %.1f s (%.2f g CO2eq).",
            result.total_energy_wh,
            result.duration_s,
            result.carbon.emissions_g,
        )
        return result

    # ------------------------------------------------------------------
    # Sampling loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.wait(self.config.sampling_interval_s):
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Monitor tick failed: %s", exc)

    def _tick(self) -> None:
        t_rel = time.monotonic() - self._t0_monotonic
        with self._lock:
            self._series.timestamps_s.append(t_rel)
            for sampler in self._samplers:
                try:
                    sample = sampler.sample()
                except Exception as exc:  # samplers shouldn't raise, but never trust
                    logger.debug("Sampler %s raised: %s", sampler.name, exc)
                    sample = None
                name = sampler.name
                power = sample.power_w if sample else None
                self._series.power_w[name].append(power)
                self._series.utilization_pct[name].append(
                    sample.utilization_pct if sample else None
                )
                self._series.memory_used_mb[name].append(sample.memory_used_mb if sample else None)
                self._series.temperature_c[name].append(sample.temperature_c if sample else None)
                self._integrators[name].add(t_rel, power)

    # ------------------------------------------------------------------
    # Live access
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Thread-safe view of running totals for live displays/callbacks.

        Returns:
            A dict with ``elapsed_s``, ``total_energy_j``, per-device
            ``energy_j``, and the most recent ``power_w`` per device.
        """
        with self._lock:
            energy = {name: integ.energy_j for name, integ in self._integrators.items()}
            last_power = {
                name: next((p for p in reversed(series) if p is not None), None)
                for name, series in self._series.power_w.items()
            }
        return {
            "elapsed_s": time.monotonic() - self._t0_monotonic,
            "total_energy_j": sum(energy.values()),
            "energy_j": energy,
            "power_w": last_power,
        }

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _stats(values: list[float | None]) -> tuple:
        present = [v for v in values if v is not None]
        if not present:
            return 0.0, 0.0
        return sum(present) / len(present), max(present)

    def _build_result(self) -> MonitorResult:
        assert self._started_at is not None
        ended_at = datetime.now(timezone.utc)
        # Duration is measured on the same monotonic clock the integrator
        # integrates against — never wall-clock. A DST or NTP step mid-run
        # would otherwise desync duration_s from the energy time base and
        # silently corrupt avg_total_power_w. The last recorded timestamp is
        # exactly the span energy was integrated over.
        with self._lock:
            last_t = self._series.timestamps_s[-1] if self._series.timestamps_s else None
        duration = last_t if last_t is not None else time.monotonic() - self._t0_monotonic

        devices: list[DeviceEnergy] = []
        utilization: dict[str, UtilizationStats] = {}
        peak_memory: dict[str, float] = {}
        for sampler in self._samplers:
            name = sampler.name
            avg_p, max_p = self._stats(self._series.power_w[name])
            devices.append(
                DeviceEnergy(
                    name=name,
                    energy_j=self._integrators[name].energy_j,
                    avg_power_w=avg_p,
                    max_power_w=max_p,
                    power_estimated=sampler.estimates_power,
                )
            )
            avg_u, max_u = self._stats(self._series.utilization_pct[name])
            utilization[name] = UtilizationStats(avg_pct=avg_u, max_pct=max_u)
            _, peak_mem = self._stats(self._series.memory_used_mb[name])
            if peak_mem > 0:
                peak_memory[name] = peak_mem

        total_j = sum(d.energy_j for d in devices)
        carbon = CarbonEstimator(self.config.carbon).estimate(joules_to_kwh(total_j))

        return MonitorResult(
            project=self.project,
            run_name=self.run_name,
            started_at=self._started_at,
            ended_at=ended_at,
            duration_s=duration,
            sampling_interval_s=self.config.sampling_interval_s,
            num_samples=len(self._series.timestamps_s),
            system=self.system,
            devices=devices,
            utilization=utilization,
            peak_memory_mb=peak_memory,
            carbon=carbon,
            timeseries=self._series,
        )
