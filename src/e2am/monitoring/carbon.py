"""Carbon emission estimation from measured energy.

Emissions are ``energy (kWh) × grid carbon intensity (gCO2eq/kWh)``. The
intensity is resolved in priority order:

1. Explicit ``carbon_intensity_g_per_kwh`` set by the user (differs from the
   world-average default).
2. Country lookup via ``country_iso_code`` from the bundled table.
3. World average (475 gCO2eq/kWh).

The bundled table holds approximate 2023 grid averages (Ember / IEA public
data) — good enough for comparative Green AI reporting. For live regional
data, the optional CodeCarbon integration can be used instead.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from e2am.config.settings import WORLD_AVG_CARBON_INTENSITY, CarbonConfig
from e2am.utils.logging import get_logger

logger = get_logger("monitor.carbon")

#: Approximate 2023 grid carbon intensity in gCO2eq/kWh (Ember/IEA data).
CARBON_INTENSITY_BY_COUNTRY = {
    "AUS": 510.0,
    "BRA": 96.0,
    "CAN": 120.0,
    "CHE": 46.0,
    "CHN": 582.0,
    "DEU": 380.0,
    "DNK": 151.0,
    "ESP": 174.0,
    "FIN": 79.0,
    "FRA": 56.0,
    "GBR": 237.0,
    "IDN": 675.0,
    "IND": 713.0,
    "ITA": 331.0,
    "JPN": 462.0,
    "KOR": 436.0,
    "MEX": 423.0,
    "NLD": 328.0,
    "NOR": 29.0,
    "POL": 662.0,
    "RUS": 322.0,
    "SAU": 706.0,
    "SGP": 471.0,
    "SWE": 41.0,
    "TWN": 561.0,
    "USA": 369.0,
    "ZAF": 708.0,
}


class CarbonResult(BaseModel):
    """Carbon emissions attributed to one run."""

    emissions_g: float = Field(default=0.0, description="Total emissions in gCO2eq.")
    intensity_g_per_kwh: float = Field(
        default=WORLD_AVG_CARBON_INTENSITY,
        description="Grid carbon intensity used for the conversion.",
    )
    intensity_source: str = Field(
        default="world_average",
        description="Where the intensity came from: 'user', 'country:<ISO>', "
        "or 'world_average'.",
    )


class CarbonEstimator:
    """Converts energy to CO2eq emissions using a resolved grid intensity."""

    def __init__(self, config: CarbonConfig | None = None) -> None:
        self.config = config or CarbonConfig()
        self.intensity_g_per_kwh, self.intensity_source = self._resolve_intensity()

    def _resolve_intensity(self) -> tuple[float, str]:
        cfg = self.config
        # Explicit None (not a magic value) means "user did not specify": a
        # user living in a 475 g/kWh grid must be able to pin that number and
        # still have it beat their country code.
        if cfg.carbon_intensity_g_per_kwh is not None:
            return cfg.carbon_intensity_g_per_kwh, "user"
        if cfg.country_iso_code:
            code = cfg.country_iso_code.strip().upper()
            if code in CARBON_INTENSITY_BY_COUNTRY:
                return CARBON_INTENSITY_BY_COUNTRY[code], f"country:{code}"
            logger.warning(
                "Unknown country code %r; falling back to world-average "
                "carbon intensity (%.0f g/kWh).",
                cfg.country_iso_code,
                WORLD_AVG_CARBON_INTENSITY,
            )
        return WORLD_AVG_CARBON_INTENSITY, "world_average"

    def estimate(self, energy_kwh: float) -> CarbonResult:
        """Compute emissions for the given energy.

        Args:
            energy_kwh: Total energy consumed in kilowatt-hours.

        Returns:
            A :class:`CarbonResult` with emissions in gCO2eq.
        """
        return CarbonResult(
            emissions_g=energy_kwh * self.intensity_g_per_kwh,
            intensity_g_per_kwh=self.intensity_g_per_kwh,
            intensity_source=self.intensity_source,
        )
