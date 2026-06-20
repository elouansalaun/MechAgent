"""Description of a static structural problem, independent of the solver."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mechagent.core.geometry import GeometryModel, Vec3
from mechagent.materials.material import Material
from mechagent.simulation.mesh import MeshSettings


class Constraint(BaseModel):
    """Zero displacement on the given degrees of freedom (1=x, 2=y, 3=z) of a face group."""

    model_config = ConfigDict(extra="forbid")

    surface: str
    dofs: tuple[int, ...] = (1, 2, 3)
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
    assumptions: list[str] = []
