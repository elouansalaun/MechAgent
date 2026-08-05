"""The engineering request as the user states it, in the user's global frame.

Every field is optional: the request can come half filled from the LLM parser, and the review
step decides what is missing. Field descriptions are also the instructions the LLM gets.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Vector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    z: float

    def array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


class InterfacePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: Vector = Field(description="hole centre in the global frame, mm")
    axis: Vector = Field(description="hole axis direction in the global frame")
    hole_diameter: float | None = Field(None, description="hole diameter, mm")


class EngineeringRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component: str | None = Field(None, description="component type, e.g. mounting bracket")
    mounting_points: list[InterfacePoint] = Field(
        default_factory=list,
        description=(
            "bolt holes fixing the part to its support. Position: where the bolt axis meets "
            "the support surface. Axis: normal of the support surface, pointing into the part"
        ),
    )
    load_point: InterfacePoint | None = Field(
        None,
        description=(
            "pin hole through which the load enters. Position: where the hole axis meets the "
            "face of the part on the support side"
        ),
    )
    force: Vector | None = Field(None, description="maximum load in the global frame, N")
    load_type: Literal["static", "cyclic"] | None = None
    cycles: float | None = Field(None, description="required number of load cycles")
    load_ratio: float | None = Field(None, description="R = minimum load / maximum load")
    min_safety_factor: float | None = Field(None, description="static safety factor on yield")
    max_displacement: float | None = Field(None, description="maximum displacement, mm")
    max_mass: float | None = Field(None, description="maximum mass, kg")
    objective: Literal["mass", "cost"] | None = Field(None, description="what to minimise")
    candidate_materials: list[str] | None = Field(
        None, description="material names or families (aluminium, steel, titanium, ...)"
    )
    manufacturing: str | None = Field(None, description="manufacturing process")
    excitation_frequency: float | None = Field(None, description="vibration excitation, Hz")
    max_width: float | None = Field(None, description="maximum width of the part, mm")
    notes: str | None = Field(None, description="anything stated that fits no other field")
