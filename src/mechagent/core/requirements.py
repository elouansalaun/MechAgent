"""Engineering requirements of a design, in the project units (mm, N, kg, cycles).

In Phase 3 the agent will build these from the user's description. For now they are written
by hand, and every field the verification depends on is explicit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mechagent.core.geometry import Vec3
from mechagent.materials.fatigue import FatigueSettings


class FatigueRequirement(BaseModel):
    """Cyclic loading as load factors of the reference force.

    Either constant amplitude (`cycles` between `load_ratio` * F and F), or a load factor
    `history` repeated `history_repeats` times over the life of the part.
    """

    model_config = ConfigDict(extra="forbid")

    cycles: float | None = Field(None, gt=0)
    load_ratio: float = Field(0.0, ge=-1.0, lt=1.0, description="R = F_min / F_max")
    history: list[float] | None = None
    history_repeats: float | None = Field(None, gt=0)
    min_safety_factor: float = Field(1.0, gt=0, description="on the load, at the required life")
    allowable_damage: float = Field(1.0, gt=0, le=1.0)
    settings: FatigueSettings = FatigueSettings()

    @model_validator(mode="after")
    def _one_loading_mode(self) -> FatigueRequirement:
        constant = self.cycles is not None
        spectrum = self.history is not None or self.history_repeats is not None
        if constant == spectrum:
            raise ValueError("give either cycles (constant amplitude) or history + repeats")
        if spectrum and (not self.history or self.history_repeats is None):
            raise ValueError("a load history needs both history and history_repeats")
        return self

    @property
    def min_load_factor(self) -> float:
        return min(self.history) if self.history else self.load_ratio


class DesignRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force: Vec3 = Field(description="maximum load on the load hole, N")
    min_safety_factor: float = Field(1.5, gt=0, description="static, on yield")
    max_displacement: float | None = Field(None, gt=0, description="mm")
    max_mass: float | None = Field(None, gt=0, description="kg")
    fatigue: FatigueRequirement | None = None
    objective: Literal["mass", "cost"] = "mass"
    candidate_materials: list[str] | None = None  # names or categories, None means all
    manufacturing: Literal["cnc_machining"] = "cnc_machining"
