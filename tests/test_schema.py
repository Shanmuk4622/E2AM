"""Artifact schema versioning and v1 -> v2 migration tests.

The contract under test: an artifact written by E2AM 0.1.x must still load
here, and must load with the *meaning it was written with* — not merely
without raising.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from e2am.config.settings import WORLD_AVG_CARBON_INTENSITY, ExperimentConfig, load_config
from e2am.exceptions import ConfigError
from e2am.monitoring.carbon import CARBON_INTENSITY_BY_COUNTRY, CarbonEstimator
from e2am.monitoring.result import MonitorResult
from e2am.schema import LEGACY_VERSION, SCHEMA_VERSION, detect_version, migrate
from e2am.trainer.result import TrainingResult

# A config.yaml exactly as E2AM 0.1.x wrote it: no schema_version, and the
# world-average intensity written out literally even though the user only set
# a country code.
V1_CONFIG = """
project: legacy
run_name: legacy-run
seed: 7
tags: []
notes: ''
monitor:
  sampling_interval_s: 1.0
  gpu_indices: null
  cpu_tdp_w: null
  carbon:
    country_iso_code: IND
    carbon_intensity_g_per_kwh: 475.0
output:
  dir: results
  save_plots: true
  save_html: true
  save_pdf: false
  save_json: true
  save_csv: true
trainer:
  epochs: 10
  device: null
  mixed_precision: false
  gradient_accumulation_steps: 1
  max_grad_norm: null
  log_every_n_steps: 50
"""


# ---------------------------------------------------------------------------
# version detection
# ---------------------------------------------------------------------------


def test_unversioned_artifact_is_v1() -> None:
    assert detect_version({}) == LEGACY_VERSION
    assert detect_version({"project": "x"}) == LEGACY_VERSION


def test_declared_version_is_read() -> None:
    assert detect_version({"schema_version": 2}) == 2


def test_unreadable_version_falls_back_to_legacy() -> None:
    assert detect_version({"schema_version": "banana"}) == LEGACY_VERSION


def test_current_models_stamp_the_current_version() -> None:
    now = datetime.now(timezone.utc)
    assert MonitorResult(started_at=now, ended_at=now).schema_version == SCHEMA_VERSION
    assert TrainingResult().schema_version == SCHEMA_VERSION
    assert ExperimentConfig().schema_version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# migration mechanics
# ---------------------------------------------------------------------------


def test_migrate_does_not_mutate_input() -> None:
    original = {"monitor": {"carbon": {"carbon_intensity_g_per_kwh": 475.0}}}
    migrate(original, "config")
    assert original["monitor"]["carbon"]["carbon_intensity_g_per_kwh"] == 475.0


def test_migrate_stamps_current_version() -> None:
    assert migrate({}, "result")["schema_version"] == SCHEMA_VERSION


def test_current_version_passes_through_untouched() -> None:
    data = {"schema_version": SCHEMA_VERSION, "project": "x"}
    assert migrate(data, "config") is data


def test_future_schema_is_refused_with_upgrade_hint() -> None:
    with pytest.raises(ConfigError, match="Upgrade E2AM"):
        migrate({"schema_version": SCHEMA_VERSION + 99}, "result")


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown artifact kind"):
        migrate({}, "spreadsheet")


# ---------------------------------------------------------------------------
# the migration that actually carries meaning
# ---------------------------------------------------------------------------


def test_v1_sentinel_intensity_becomes_unset() -> None:
    """475.0 in a v1 config meant 'unset', so it must not become 'user'."""
    migrated = migrate(
        {"monitor": {"carbon": {"country_iso_code": "IND", "carbon_intensity_g_per_kwh": 475.0}}},
        "config",
    )
    assert migrated["monitor"]["carbon"]["carbon_intensity_g_per_kwh"] is None


def test_v1_non_sentinel_intensity_is_preserved() -> None:
    """Any other value was a genuine user choice and must survive."""
    migrated = migrate({"monitor": {"carbon": {"carbon_intensity_g_per_kwh": 120.0}}}, "config")
    assert migrated["monitor"]["carbon"]["carbon_intensity_g_per_kwh"] == 120.0


def test_v1_config_file_keeps_its_original_meaning(tmp_path: Path) -> None:
    """End-to-end: a real 0.1.x config.yaml still resolves to India's grid."""
    path = tmp_path / "config.yaml"
    path.write_text(V1_CONFIG, encoding="utf-8")

    cfg = load_config(path)
    assert cfg.schema_version == SCHEMA_VERSION
    assert cfg.project == "legacy"
    assert cfg.monitor.carbon.country_iso_code == "IND"

    estimator = CarbonEstimator(cfg.monitor.carbon)
    assert estimator.intensity_g_per_kwh == CARBON_INTENSITY_BY_COUNTRY["IND"]
    assert estimator.intensity_source == "country:IND"
    assert estimator.intensity_g_per_kwh != WORLD_AVG_CARBON_INTENSITY


def test_config_round_trip_is_stable_after_migration(tmp_path: Path) -> None:
    """Re-saving a migrated config produces a v2 file that reloads identically."""
    legacy = tmp_path / "old.yaml"
    legacy.write_text(V1_CONFIG, encoding="utf-8")
    migrated = load_config(legacy)

    modern = migrated.to_yaml(tmp_path / "new.yaml")
    written = yaml.safe_load(modern.read_text(encoding="utf-8"))
    assert written["schema_version"] == SCHEMA_VERSION
    assert written["monitor"]["carbon"]["carbon_intensity_g_per_kwh"] is None
    assert load_config(modern) == migrated


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def test_v1_metrics_json_still_loads(tmp_path: Path) -> None:
    """A 0.1.x metrics.json (no schema_version) loads and is stamped v2."""
    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir()
    payload = TrainingResult(project="p", run_name="legacy-run", epochs_completed=3).model_dump(
        mode="json"
    )
    payload.pop("schema_version")  # as 0.1.x wrote it
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = TrainingResult.load(run_dir)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.run_name == "legacy-run"
    assert loaded.epochs_completed == 3


def test_recorded_carbon_intensity_is_not_sentinel_translated() -> None:
    """CarbonResult records what was *used* — a measurement, not a setting.

    475.0 there is a real recorded number and must survive migration intact,
    unlike the same value in a CarbonConfig.
    """
    migrated = migrate({"carbon": {"emissions_g": 1.0, "intensity_g_per_kwh": 475.0}}, "result")
    assert migrated["carbon"]["intensity_g_per_kwh"] == 475.0
