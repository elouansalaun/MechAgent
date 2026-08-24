"""Parametric L-shaped mounting bracket.

Local frame of the bracket (mm):

        z
        |  ___
        | |   |  <- vertical flange, load hole along x at height `load_hole_height`
        | |   |
        | |   |__________________
        | |  (fillet)             |  <- base plate, bolt holes along z at x = `bolt_hole_position`
        |_|_______________________|____ x

- The back face of the flange is the plane x = 0, the bottom of the base is z = 0.
- The bracket is centred on y = 0, the two bolt holes sit at y = +/- bolt_hole_spacing / 2.
- Optional triangular ribs (gussets) join the base and the flange along both side edges.
  Their concave edges (toes and junctions with the plates) are filleted like a machined part,
  a sharp re-entrant corner would give a stress that never converges with mesh refinement.
"""

from __future__ import annotations

from pathlib import Path

import cadquery as cq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mechagent.cad.common import GeometryError, export_geometry, hole_faces, single_solid
from mechagent.core.geometry import GeometryModel, HoleFeature

# Minimum material left between a hole and any edge, as a fraction of the hole diameter.
MIN_LIGAMENT_RATIO = 0.5


class BracketParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: float = Field(60.0, gt=0, description="extent along y, mm")
    base_length: float = Field(80.0, gt=0, description="extent of the base along x, mm")
    base_thickness: float = Field(8.0, gt=0, description="mm")
    flange_height: float = Field(80.0, gt=0, description="extent of the flange along z, mm")
    flange_thickness: float = Field(8.0, gt=0, description="mm")
    fillet_radius: float = Field(6.0, ge=0, description="inner corner fillet, mm")
    bolt_hole_diameter: float = Field(9.0, gt=0, description="mm, 9 mm is an M8 clearance hole")
    bolt_hole_spacing: float = Field(36.0, gt=0, description="distance between bolt axes along y")
    bolt_hole_position: float = Field(50.0, gt=0, description="x of the bolt axes, mm")
    load_hole_diameter: float = Field(12.0, gt=0, description="mm")
    load_hole_height: float = Field(60.0, gt=0, description="z of the load hole axis, mm")
    rib_thickness: float = Field(0.0, ge=0, description="mm, 0 means no ribs")
    rib_size: float = Field(40.0, gt=0, description="length of both rib legs from the corner, mm")
    rib_fillet_radius: float = Field(
        2.0, ge=0, description="fillet on the rib toes and junctions, typical tool radius, mm"
    )

    @model_validator(mode="after")
    def _check_feasible(self) -> BracketParameters:
        problems = self.feasibility_problems()
        if problems:
            raise ValueError("infeasible bracket: " + "; ".join(problems))
        return self

    def feasibility_problems(self) -> list[str]:
        problems = []
        d_bolt, d_load = self.bolt_hole_diameter, self.load_hole_diameter
        bolt_margin = d_bolt / 2 + MIN_LIGAMENT_RATIO * d_bolt
        load_margin = d_load / 2 + MIN_LIGAMENT_RATIO * d_load
        side_limit = self.width / 2
        if self.rib_thickness > 0:
            side_limit -= self.rib_thickness + self.rib_fillet_radius

        if self.flange_thickness >= self.base_length:
            problems.append("flange thickness must be smaller than base length")
        if self.base_thickness >= self.flange_height:
            problems.append("base thickness must be smaller than flange height")
        if self.fillet_radius > 0.5 * min(
            self.base_length - self.flange_thickness, self.flange_height - self.base_thickness
        ):
            problems.append("fillet radius too large for the bracket legs")

        corner_x = self.flange_thickness + self.fillet_radius
        if self.bolt_hole_position - bolt_margin < corner_x:
            problems.append("bolt holes too close to the flange or fillet")
        if self.bolt_hole_position + bolt_margin > self.base_length:
            problems.append("bolt holes too close to the end of the base")
        if self.bolt_hole_spacing / 2 + bolt_margin > side_limit:
            problems.append("bolt holes too close to the side edges or ribs")
        if self.bolt_hole_spacing < d_bolt + MIN_LIGAMENT_RATIO * d_bolt:
            problems.append("bolt holes too close to each other")

        corner_z = self.base_thickness + self.fillet_radius
        if self.load_hole_height - load_margin < corner_z:
            problems.append("load hole too close to the base or fillet")
        if self.load_hole_height + load_margin > self.flange_height:
            problems.append("load hole too close to the top of the flange")
        if load_margin > side_limit:
            problems.append("load hole too wide for the flange width")

        if self.rib_thickness > 0:
            if 2 * self.rib_thickness > self.width / 2:
                problems.append("ribs too thick for the bracket width")
            # The toe fillet must still have material under it, at the end of the base and at
            # the top of the flange.
            toe = self.rib_size + self.rib_fillet_radius
            if toe > self.base_length - self.flange_thickness:
                problems.append("ribs longer than the base")
            if toe > self.flange_height - self.base_thickness:
                problems.append("ribs taller than the flange")
            if self.rib_size <= self.fillet_radius:
                problems.append("ribs smaller than the fillet")
            if self.rib_fillet_radius > self.rib_size / 4:
                problems.append("rib fillet radius too large for the ribs")
        return problems


def bracket_holes(params: BracketParameters) -> list[HoleFeature]:
    y = params.bolt_hole_spacing / 2
    holes = [
        HoleFeature(
            name=f"bolt_hole_{i + 1}",
            center=(params.bolt_hole_position, side * y, 0.0),
            axis=(0.0, 0.0, 1.0),
            diameter=params.bolt_hole_diameter,
        )
        for i, side in enumerate((-1, 1))
    ]
    holes.append(
        HoleFeature(
            name="load_hole",
            center=(0.0, 0.0, params.load_hole_height),
            axis=(1.0, 0.0, 0.0),
            diameter=params.load_hole_diameter,
        )
    )
    return holes


def _rib_concave_edges(solid: cq.Solid, params: BracketParameters) -> list[cq.Edge]:
    """Edges where the ribs meet the base, the flange and the corner fillet, and the rib toes."""
    tb, tf, s = params.base_thickness, params.flange_thickness, params.rib_size
    inner_y = params.width / 2 - params.rib_thickness
    edges = []
    for face in solid.Faces():
        box = face.BoundingBox()
        on_inner_side = any(
            abs(box.ymin - y) < 1e-6 and abs(box.ymax - y) < 1e-6 for y in (-inner_y, inner_y)
        )
        if face.geomType() == "PLANE" and on_inner_side:
            for edge in face.Edges():
                ends = (edge.startPoint(), edge.endPoint())
                # The sloped free edge x + z = tf + tb + s is convex, it stays sharp.
                if not all(abs(p.x + p.z - (tf + tb + s)) < 1e-6 for p in ends):
                    edges.append(edge)

    toes = [(tf + s, tb), (tf, tb + s)]
    for edge in solid.Edges():
        a, b = edge.startPoint(), edge.endPoint()
        along_y = abs(a.x - b.x) < 1e-9 and abs(a.z - b.z) < 1e-9 and abs(a.y - b.y) > 1e-9
        if along_y and any(abs(a.x - x) < 1e-6 and abs(a.z - z) < 1e-6 for x, z in toes):
            edges.append(edge)
    return edges


def build_bracket_solid(params: BracketParameters) -> cq.Solid:
    """The bracket solid. Any OpenCascade failure is raised as a GeometryError."""
    try:
        return _build_solid(params)
    except GeometryError:
        raise
    except Exception as exc:  # OCP raises Standard_Failure and friends, not Python errors
        raise GeometryError(f"OpenCascade could not build the bracket: {exc}") from exc


def _build_solid(params: BracketParameters) -> cq.Solid:
    tb, tf = params.base_thickness, params.flange_thickness
    length, height, width = params.base_length, params.flange_height, params.width

    # The XZ workplane maps local (u, v) to global (x, z), extruding both ways keeps y centred.
    profile = [(0, 0), (length, 0), (length, tb), (tf, tb), (tf, height), (0, height)]
    body = cq.Workplane("XZ").polyline(profile).close().extrude(width / 2, both=True)

    if params.fillet_radius > 0:
        corner = cq.selectors.NearestToPointSelector((tf, 0, tb))
        body = body.edges("|Y").edges(corner).fillet(params.fillet_radius)

    if params.rib_thickness > 0:
        s, t = params.rib_size, params.rib_thickness
        for y in (-(width - t) / 2, (width - t) / 2):
            rib = (
                cq.Workplane("XZ", origin=(0, y, 0))
                .polyline([(tf, tb), (tf + s, tb), (tf, tb + s)])
                .close()
                .extrude(t / 2, both=True)
            )
            body = body.union(rib)
        if params.rib_fillet_radius > 0:
            solid = single_solid(body)
            edges = _rib_concave_edges(solid, params)
            body = cq.Workplane("XY").add(solid.fillet(params.rib_fillet_radius, edges))

    for hole in bracket_holes(params):
        # Cylinders start 1 mm outside the part so the cut goes cleanly through.
        axis = cq.Vector(*hole.axis)
        start = cq.Vector(*hole.center) - axis
        depth = (tb if hole.axis[2] else tf) + 2
        body = body.cut(cq.Solid.makeCylinder(hole.diameter / 2, depth, start, axis))

    return single_solid(body)


def rib_fillet_faces(solid: cq.Solid, params: BracketParameters) -> list[cq.Face]:
    """Curved faces lying entirely in the rib zones, i.e. the rib fillets."""
    if params.rib_thickness == 0 or params.rib_fillet_radius == 0:
        return []
    limit = params.width / 2 - params.rib_thickness - params.rib_fillet_radius - 1e-6
    faces = []
    for face in solid.Faces():
        box = face.BoundingBox()
        if face.geomType() != "PLANE" and (box.ymin >= limit or box.ymax <= -limit):
            faces.append(face)
    return faces


def build_bracket(params: BracketParameters, out_dir: Path, name: str = "bracket") -> GeometryModel:
    """Build the bracket, export it to BREP and describe its holes and rib fillet faces."""
    solid = build_bracket_solid(params)
    holes = bracket_holes(params)

    groups = {
        "bolt_holes": [f for h in holes if h.name.startswith("bolt") for f in hole_faces(solid, h)],
        "load_hole": hole_faces(solid, next(h for h in holes if h.name == "load_hole")),
    }
    fillets = rib_fillet_faces(solid, params)
    if fillets:
        groups["rib_fillets"] = fillets
    return export_geometry(solid, name, groups, out_dir, holes=holes)
