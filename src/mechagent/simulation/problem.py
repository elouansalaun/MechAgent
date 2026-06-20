"""Description of a static structural problem, independent of the solver."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mechagent.core.geometry import GeometryModel, Vec3
from mechagent.materials.material import Material
from mechagent.simulation.mesh import MeshSettings


class Constraint(BaseModel):
    """Zero displacement on the given degrees of freedom (1=x, 2=y, 3=z) of a face group.

    A "support" stands for the real surroundings of the part and creates artificial stress
    peaks. A "symmetry" plane is exact, stresses on it are kept as design values.
    """

    model_config = ConfigDict(extra="forbid")

    surface: str
    dofs: tuple[int, ...] = (1, 2, 3)
    kind: Literal["support", "symmetry"] = "support"
    description: str = ""

    @field_validator("dofs")
    @classmethod
    def _check_dofs(cls, dofs: tuple[int, ...]) -> tuple[int, ...]:
        if not dofs or any(d not in (1, 2, 3) for d in dofs) or len(set(dofs)) != len(dofs):
            raise ValueError(f"dofs must be distinct values among 1, 2, 3, got {dofs}")
        return tuple(sorted(dofs))


class SurfaceForce(BaseModel):
    """A total force (N) spread as a uniform traction over a face group.

    With `bearing_axis` and `bearing_center` set, the face group is a hole loaded by a pin:
    only the half of the hole the pin pushes against carries the force.
    """

    model_config = ConfigDict(extra="forbid")

    surface: str
    force: Vec3
    bearing_axis: Vec3 | None = None
    bearing_center: Vec3 | None = None
    description: str = ""


class StaticProblem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    geometry: GeometryModel
    material: Material
    constraints: list[Constraint] = Field(min_length=1)
    loads: list[SurfaceForce] = Field(min_length=1)
    mesh: MeshSettings = MeshSettings()
    # Stresses on and closer than this to support nodes are not used as design values: rigid
    # supports create stress singularities that grow without bound as the mesh is refined.
    stress_exclusion_distance: float = Field(0.0, ge=0, description="mm")
    assumptions: list[str] = []
