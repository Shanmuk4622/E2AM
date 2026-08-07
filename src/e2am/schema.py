"""Artifact schema versioning and migration.

Every artifact E2AM writes — ``metrics.json`` and ``config.yaml`` — carries a
``schema_version``. Artifacts written by E2AM 0.1.x predate the field; they
are treated as version 1 and migrated forward on load, so a run recorded with
0.1.x still opens in this version with its original meaning intact.

Migrations are plain dict transforms applied *before* pydantic validation.
They deliberately do not import the models: a migration must be able to
reshape data that the current models would reject outright.

Design rule: a migration preserves the *semantics* the file was written with,
not its literal bytes. Where v1 encoded "unset" as a magic default value, the
migration restores the explicit ``None`` that v2 uses instead.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from e2am.exceptions import ConfigError
from e2am.utils.logging import get_logger

logger = get_logger("schema")

#: Schema version written by this release.
SCHEMA_VERSION = 2

#: Version assumed for artifacts with no ``schema_version`` field (E2AM 0.1.x).
LEGACY_VERSION = 1

#: The world-average carbon intensity that v1 used as a "not set by the user"
#: sentinel in ``CarbonConfig``. v2 uses an explicit ``None`` instead.
_V1_CARBON_SENTINEL = 475.0


def detect_version(data: dict[str, Any]) -> int:
    """Return the schema version of a loaded artifact.

    Args:
        data: Raw parsed JSON/YAML of an E2AM artifact.

    Returns:
        The declared ``schema_version``, or :data:`LEGACY_VERSION` when the
        field is absent (an E2AM 0.1.x artifact).
    """
    version = data.get("schema_version", LEGACY_VERSION)
    try:
        return int(version)
    except (TypeError, ValueError):
        logger.warning("Unreadable schema_version %r; assuming v%d.", version, LEGACY_VERSION)
        return LEGACY_VERSION


def _migrate_config_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Restore explicit carbon-intensity intent lost to the v1 sentinel.

    In v1 ``carbon_intensity_g_per_kwh`` defaulted to 475.0 and "the user set
    it" was inferred from the value differing from that default — so a config
    holding exactly 475.0 meant *unset*, and ``to_yaml`` wrote that literal
    into every file. v2 stores ``None`` for unset, so the sentinel value has
    to be translated back or every legacy config would suddenly claim the
    user pinned 475 g/kWh and override its own country lookup.
    """
    carbon = data.get("monitor", {}).get("carbon")
    if isinstance(carbon, dict) and carbon.get("carbon_intensity_g_per_kwh") == _V1_CARBON_SENTINEL:
        carbon["carbon_intensity_g_per_kwh"] = None
    return data


def _migrate_result_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Bring a v1 result forward.

    v1 and v2 results are structurally compatible: ``CarbonResult`` records
    the intensity that was actually *used*, which is a measurement rather than
    a setting and so needs no sentinel translation. Only the version stamp
    changes.
    """
    return data


#: Migrations per artifact kind, keyed by the version they upgrade *from*.
_MIGRATIONS: dict[str, dict[int, Any]] = {
    "config": {1: _migrate_config_v1_to_v2},
    "result": {1: _migrate_result_v1_to_v2},
}


def migrate(data: dict[str, Any], kind: str) -> dict[str, Any]:
    """Upgrade a loaded artifact to the current schema version.

    Args:
        data: Raw parsed artifact. Not mutated; a copy is returned.
        kind: ``"config"`` for ``config.yaml``, ``"result"`` for
            ``metrics.json``.

    Returns:
        The artifact upgraded to :data:`SCHEMA_VERSION`.

    Raises:
        ConfigError: If the artifact was written by a *newer* E2AM than this
            one, or ``kind`` is unknown.
    """
    if kind not in _MIGRATIONS:
        raise ConfigError(f"Unknown artifact kind {kind!r}; expected 'config' or 'result'.")

    version = detect_version(data)
    if version > SCHEMA_VERSION:
        raise ConfigError(
            f"This artifact uses schema v{version}, but this E2AM understands "
            f"up to v{SCHEMA_VERSION}. Upgrade E2AM: pip install -U e2am"
        )
    if version == SCHEMA_VERSION:
        return data

    migrated = deepcopy(data)
    steps = _MIGRATIONS[kind]
    while version < SCHEMA_VERSION:
        step = steps.get(version)
        if step is None:
            raise ConfigError(f"No migration path for {kind} schema v{version} -> v{version + 1}.")
        migrated = step(migrated)
        version += 1

    migrated["schema_version"] = SCHEMA_VERSION
    logger.debug(
        "Migrated %s artifact from schema v%d to v%d.",
        kind,
        detect_version(data),
        SCHEMA_VERSION,
    )
    return migrated
