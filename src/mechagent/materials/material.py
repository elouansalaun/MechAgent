"""Material model used by the simulations.

Units follow the project system: MPa for moduli and strengths, tonne/mm^3 for density.
The material database and selection logic come in Phase 2.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Material(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    youngs_modulus: float = Field(gt=0, description="MPa")
    poisson_ratio: float = Field(gt=-1.0, lt=0.5)
    density: float = Field(gt=0, description="tonne/mm^3, e.g. 7.85e-9 for steel")
    yield_strength: float = Field(gt=0, description="MPa")
    ultimate_strength: float = Field(gt=0, description="MPa")
    source: str = ""

    @model_validator(mode="after")
    def _check_strengths(self) -> Material:
        if self.ultimate_strength < self.yield_strength:
            raise ValueError("ultimate strength must be greater than or equal to yield strength")
        # Catches densities typed in kg/m^3 instead of tonne/mm^3.
        if self.density > 1e-7:
            raise ValueError(f"density {self.density} looks wrong, it must be in tonne/mm^3")
        return self

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
