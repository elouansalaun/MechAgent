"""Static load case of the L bracket: bolted base, pin load in the flange hole."""

from __future__ import annotations

from mechagent.cad.bracket import BracketParameters
from mechagent.core.geometry import GeometryModel, Vec3
from mechagent.materials.material import Material
from mechagent.simulation.mesh import MeshSettings
from mechagent.simulation.problem import Constraint, StaticProblem, SurfaceForce

BRACKET_ASSUMPTIONS = [
    "Bolt holes are fully fixed: rigid bolts, no preload, no contact or friction with the "
    "support surface. This is stiffer than a real bolted joint and gives a stress "
    "singularity at the hole edges, so stresses within a quarter of the bolt hole diameter "
    "of the holes are excluded from the design values.",
    "The load is introduced by a rigid pin: the total force is applied as a uniform traction "
    "on the half of the load hole facing the force direction (no contact analysis).",
]


def default_bracket_mesh(params: BracketParameters) -> MeshSettings:
    """About two quadratic elements through the thinnest wall, finer around the holes."""
    size = min(params.base_thickness, params.flange_thickness) / 2
    refinements = {"bolt_holes": size / 2, "load_hole": size / 2}
    if params.rib_thickness > 0 and params.rib_fillet_radius > 0:
        refinements["rib_fillets"] = size / 2
    return MeshSettings(element_size=size, refinements=refinements)


def bracket_static_problem(
    geometry: GeometryModel,
    params: BracketParameters,
    material: Material,
    force: Vec3,
    mesh: MeshSettings | None = None,
) -> StaticProblem:
    hole = geometry.hole("load_hole")
    return StaticProblem(
        name=geometry.name,
        geometry=geometry,
        material=material,
        constraints=[Constraint(surface="bolt_holes", description="bolted to a rigid support")],
        loads=[
            SurfaceForce(
                surface="load_hole",
                force=force,
                bearing_axis=hole.axis,
                bearing_center=hole.center,
                description="pin load in the flange hole",
            )
        ],
        mesh=mesh or default_bracket_mesh(params),
        stress_exclusion_distance=0.25 * params.bolt_hole_diameter,
        assumptions=BRACKET_ASSUMPTIONS,
    )
