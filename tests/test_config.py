"""Configuration model and YAML round-trip tests."""

from pathlib import Path

import pytest

from e2am.config import ExperimentConfig, MonitorConfig, load_config
from e2am.exceptions import ConfigError


def test_defaults_are_sensible() -> None:
    cfg = ExperimentConfig()
    assert cfg.project == "e2am"
    assert cfg.monitor.sampling_interval_s == 1.0
    # None means "not specified" — the estimator resolves it to the country
    # table or the world average. A concrete default here would be a sentinel
    # indistinguishable from a user pinning that same number.
    assert cfg.monitor.carbon.carbon_intensity_g_per_kwh is None
    assert cfg.output.dir == Path("results")
    assert cfg.trainer.epochs == 10


def test_run_name_resolution() -> None:
    named = ExperimentConfig(project="demo", run_name="exp-1")
    assert named.resolved_run_name() == "exp-1"
    auto = ExperimentConfig(project="demo")
    assert auto.resolved_run_name().startswith("demo-")


def test_run_dir_combines_output_and_run_name(tmp_path: Path) -> None:
    cfg = ExperimentConfig(run_name="r1", output={"dir": tmp_path})
    assert cfg.run_dir() == tmp_path / "r1"


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        project="resnet",
        run_name="run-42",
        seed=42,
        tags=["vision", "baseline"],
        monitor=MonitorConfig(sampling_interval_s=0.5, cpu_tdp_w=95.0),
    )
    path = cfg.to_yaml(tmp_path / "config.yaml")
    loaded = load_config(path)
    assert loaded == cfg


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(bad)


def test_invalid_values_raise_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad_values.yaml"
    bad.write_text("monitor:\n  sampling_interval_s: -5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config(bad)


def test_sampling_interval_bounds() -> None:
    with pytest.raises(ValueError):
        MonitorConfig(sampling_interval_s=0.0)
