"""MonitorSession tests using injected fake samplers for exact arithmetic."""

import time
from datetime import timedelta

import pytest

from e2am.config.settings import MonitorConfig
from e2am.exceptions import MonitorError
from e2am.monitoring.samplers import Sample, Sampler
from e2am.monitoring.session import MonitorSession


class ConstantSampler(Sampler):
    """Deterministic sampler emitting a fixed power/utilization."""

    def __init__(self, name: str = "fake", power_w: float = 100.0) -> None:
        super().__init__(name)
        self.power = power_w
        self.torn_down = False

    @property
    def estimates_power(self) -> bool:
        return True

    def setup(self) -> bool:
        return True

    def sample(self) -> Sample:
        return Sample(power_w=self.power, utilization_pct=50.0, memory_used_mb=123.0)

    def teardown(self) -> None:
        self.torn_down = True


class ExplodingSampler(Sampler):
    """Misbehaving sampler whose sample() raises — must not kill the session."""

    @property
    def estimates_power(self) -> bool:
        return True

    def setup(self) -> bool:
        return True

    def sample(self) -> Sample:
        raise RuntimeError("sensor exploded")


def _fast_config() -> MonitorConfig:
    return MonitorConfig(sampling_interval_s=0.05)


def test_energy_matches_constant_power() -> None:
    sampler = ConstantSampler(power_w=100.0)
    session = MonitorSession(config=_fast_config(), samplers=[sampler])
    session.start()
    time.sleep(0.4)
    result = session.stop()
    expected = 100.0 * result.duration_s
    assert result.total_energy_j == pytest.approx(expected, rel=0.15)
    assert result.num_samples >= 5
    assert sampler.torn_down


def test_result_statistics_and_flat_dict() -> None:
    session = MonitorSession(
        config=_fast_config(),
        project="proj",
        run_name="run-x",
        samplers=[ConstantSampler("fake", 60.0)],
    )
    session.start()
    time.sleep(0.2)
    result = session.stop()
    assert result.project == "proj"
    assert result.run_name == "run-x"
    assert result.utilization["fake"].avg_pct == pytest.approx(50.0)
    assert result.peak_memory_mb["fake"] == pytest.approx(123.0)
    flat = result.to_flat_dict()
    assert flat["run_name"] == "run-x"
    assert flat["total_energy_wh"] > 0
    assert "avg_fake_util_pct" in flat


def test_snapshot_while_running() -> None:
    session = MonitorSession(config=_fast_config(), samplers=[ConstantSampler()])
    session.start()
    time.sleep(0.15)
    snap = session.snapshot()
    assert snap["elapsed_s"] > 0
    assert snap["power_w"]["fake"] == pytest.approx(100.0)
    assert snap["total_energy_j"] >= 0
    session.stop()


def test_exploding_sampler_does_not_crash_session() -> None:
    session = MonitorSession(
        config=_fast_config(),
        samplers=[ConstantSampler("ok", 10.0), ExplodingSampler("bad")],
    )
    session.start()
    time.sleep(0.15)
    result = session.stop()
    assert result.status == "completed"
    ok = next(d for d in result.devices if d.name == "ok")
    bad = next(d for d in result.devices if d.name == "bad")
    assert ok.energy_j > 0
    assert bad.energy_j == 0.0


def test_duration_uses_monotonic_not_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wall-clock jump mid-run must not distort duration_s or avg power.

    duration_s anchors avg_total_power_w, so if it came from datetime.now()
    an NTP/DST step during a long run would silently corrupt the headline
    watt number while the energy integral (monotonic) stayed correct.
    """
    import e2am.monitoring.session as session_mod

    real_datetime = session_mod.datetime

    class JumpingDatetime:
        """Wall clock that leaps an hour forward after the run starts."""

        calls = 0

        @classmethod
        def now(cls, tz=None):
            cls.calls += 1
            base = real_datetime.now(tz)
            return base if cls.calls == 1 else base + timedelta(hours=1)

    monkeypatch.setattr(session_mod, "datetime", JumpingDatetime)

    session = MonitorSession(config=_fast_config(), samplers=[ConstantSampler(power_w=100.0)])
    session.start()
    time.sleep(0.2)
    result = session.stop()

    # Real elapsed time is a fraction of a second, not the hour the wall clock
    # claims to have passed.
    assert result.duration_s < 5.0
    # Energy and duration share a time base, so average power stays truthful.
    assert result.avg_total_power_w == pytest.approx(100.0, rel=0.2)


def test_double_start_raises() -> None:
    session = MonitorSession(config=_fast_config(), samplers=[ConstantSampler()])
    session.start()
    with pytest.raises(MonitorError, match="already running"):
        session.start()
    session.stop()


def test_stop_without_start_raises() -> None:
    session = MonitorSession(config=_fast_config(), samplers=[ConstantSampler()])
    with pytest.raises(MonitorError, match="never started"):
        session.stop()


def test_result_round_trips_through_json() -> None:
    from e2am.monitoring.result import MonitorResult

    session = MonitorSession(config=_fast_config(), samplers=[ConstantSampler()])
    session.start()
    time.sleep(0.1)
    result = session.stop()
    restored = MonitorResult.model_validate_json(result.model_dump_json())
    assert restored.total_energy_j == pytest.approx(result.total_energy_j)
    assert restored.timeseries.timestamps_s == result.timeseries.timestamps_s
