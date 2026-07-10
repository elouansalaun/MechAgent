"""Stress-life (S-N) data and the component S-N curve.

The material data describe smooth, polished specimens under fully reversed loading (R = -1).
The component curve lowers the knee strength with Marin factors (Shigley, Mechanical
Engineering Design): surface finish, size and reliability.

The curve is a Basquin line in log-log space through (1e3, S_1e3) and (N_knee, S_knee).
Above the knee it is either flat (endurance limit, infinite life below it) or the same line
continued (no endurance limit, e.g. aluminium).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

LOW_CYCLE_LIMIT = 1e3  # below this, stress-life methods are not valid

# Shigley surface factor ka = a * Sut^b, Sut in MPa.
SURFACE_FACTORS = {
    "polished": (1.0, 0.0),
    "ground": (1.58, -0.085),
    "machined": (4.51, -0.265),
    "hot_rolled": (57.7, -0.718),
    "as_forged": (272.0, -0.995),
}
SurfaceFinish = Literal["polished", "ground", "machined", "hot_rolled", "as_forged"]

# Shigley reliability factor ke for a standard deviation of 8 % of the endurance limit.
RELIABILITY_FACTORS = {0.5: 1.0, 0.9: 0.897, 0.95: 0.868, 0.99: 0.814, 0.999: 0.753, 0.9999: 0.702}


class SNData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strength_at_1e3: float = Field(gt=0, description="fully reversed stress amplitude, MPa")
    knee_strength: float = Field(gt=0, description="fully reversed stress amplitude, MPa")
    knee_cycles: float = Field(gt=LOW_CYCLE_LIMIT)
    endurance_limit: bool
    source: str = ""

    @model_validator(mode="after")
    def _check_order(self) -> SNData:
        if self.knee_strength >= self.strength_at_1e3:
            raise ValueError("knee strength must be lower than the strength at 1e3 cycles")
        return self


class FatigueSettings(BaseModel):
    """Corrections from specimen to component, and the analysis options."""

    model_config = ConfigDict(extra="forbid")

    surface_finish: SurfaceFinish = "machined"
    reliability: float = 0.99
    size_factor: float = Field(
        0.9, gt=0, le=1.0, description="Shigley kb, 0.9 is typical for 10 to 50 mm sections"
    )
    mean_stress_correction: Literal["goodman", "gerber", "soderberg", "swt"] = "goodman"

    @model_validator(mode="after")
    def _check_reliability(self) -> FatigueSettings:
        if self.reliability not in RELIABILITY_FACTORS:
            raise ValueError(f"reliability must be one of {sorted(RELIABILITY_FACTORS)}")
        return self


def surface_factor(finish: SurfaceFinish, ultimate_strength: float) -> float:
    a, b = SURFACE_FACTORS[finish]
    return min(1.0, a * ultimate_strength**b)


class SNCurve(BaseModel):
    """Component S-N curve: stress amplitude (MPa) against cycles to failure."""

    strength_at_1e3: float
    knee_strength: float
    knee_cycles: float
    endurance_limit: bool
    marin_factor: float = 1.0  # product of the corrections applied to the knee strength

    @classmethod
    def for_component(
        cls, data: SNData, ultimate_strength: float, settings: FatigueSettings
    ) -> SNCurve:
        factor = (
            surface_factor(settings.surface_finish, ultimate_strength)
            * settings.size_factor
            * RELIABILITY_FACTORS[settings.reliability]
        )
        return cls(
            strength_at_1e3=data.strength_at_1e3,
            knee_strength=data.knee_strength * factor,
            knee_cycles=data.knee_cycles,
            endurance_limit=data.endurance_limit,
            marin_factor=factor,
        )

    @property
    def basquin_exponent(self) -> float:
        """b in S = S_1e3 * (N / 1e3)^b, negative."""
        return np.log(self.knee_strength / self.strength_at_1e3) / np.log(
            self.knee_cycles / LOW_CYCLE_LIMIT
        )

    def cycles_to_failure(self, amplitude: np.ndarray | float) -> np.ndarray:
        """Cycles to failure, inf below the endurance limit and for zero amplitude."""
        amplitude = np.asarray(amplitude, dtype=float)
        with np.errstate(divide="ignore"):
            cycles = LOW_CYCLE_LIMIT * (amplitude / self.strength_at_1e3) ** (
                1.0 / self.basquin_exponent
            )
        cycles = np.where(amplitude <= 0, np.inf, cycles)
        if self.endurance_limit:
            cycles = np.where(amplitude < self.knee_strength, np.inf, cycles)
        return cycles

    def strength_at(self, cycles: float) -> float:
        """Allowable fully reversed amplitude for a given life."""
        if self.endurance_limit and cycles >= self.knee_cycles:
            return self.knee_strength
        return float(self.strength_at_1e3 * (cycles / LOW_CYCLE_LIMIT) ** self.basquin_exponent)
