"""Place the L bracket on the user's mounting points.

The bracket is built in its own frame (see cad/bracket.py): base on the plane z = 0, bolt
axes along z, bolts at x = bolt_hole_position and y = +/- spacing / 2, flange back face on
x = 0, load hole axis along x at height z = load_hole_height.

From the global positions and axes of the two bolt holes and of the load hole we find that
frame, check that the interface really is one of an L bracket, and derive the parameters the
interface fixes. Everything else stays free for the optimiser.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from mechagent.agent.issues import RequirementIssue, inconsistent, missing
from mechagent.core.geometry import Vec3
from mechagent.core.request import EngineeringRequest

ANGLE_TOLERANCE = np.radians(1.0)
POSITION_TOLERANCE = 0.5  # mm
# Material kept beyond a hole along the base or the flange, in hole diameters.
END_MARGIN = 1.5


class BracketInterface(BaseModel):
    origin: Vec3  # global position of the local origin, mm
    x_axis: Vec3  # global directions of the local axes
    y_axis: Vec3
    z_axis: Vec3
    fixed: dict[str, float]  # bracket parameters set by the interface
    force_local: Vec3  # N

    @property
    def rotation(self) -> np.ndarray:
        """Columns are the local axes in global coordinates."""
        return np.column_stack([self.x_axis, self.y_axis, self.z_axis])

    def to_global(self, point: Vec3) -> np.ndarray:
        return np.array(self.origin) + self.rotation @ np.array(point)

    def to_local(self, point: Vec3) -> np.ndarray:
        return self.rotation.T @ (np.array(point) - np.array(self.origin))


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def map_bracket_interface(
    request: EngineeringRequest,
) -> tuple[BracketInterface | None, list[RequirementIssue]]:
    issues = []
    if len(request.mounting_points) != 2:
        issues.append(
            missing(
                "mounting_points",
                f"the bracket needs exactly two bolt holes, {len(request.mounting_points)} given",
            )
        )
    if request.load_point is None:
        issues.append(missing("load_point", "position and axis of the load hole are needed"))
    if request.force is None:
        issues.append(missing("force", "the load vector (N, global frame) is needed"))
    if issues:
        return None, issues

    bolts = request.mounting_points
    axes = [("mounting_points", b.axis) for b in bolts] + [("load_point", request.load_point.axis)]
    if any(np.linalg.norm(axis.array()) < 1e-9 for _, axis in axes):
        return None, [
            missing(field, "a hole axis direction is zero, it must be given")
            for field, axis in axes
            if np.linalg.norm(axis.array()) < 1e-9
        ]
    diameters = [b.hole_diameter for b in bolts]
    if None in diameters:
        issues.append(missing("mounting_points", "bolt hole diameters are needed (9 mm for M8)"))
    elif abs(diameters[0] - diameters[1]) > 1e-9:
        issues.append(inconsistent("mounting_points", "both bolt holes must have one diameter"))
    if request.load_point.hole_diameter is None:
        issues.append(missing("load_point", "the load hole diameter is needed"))

    p1, p2 = (b.position.array() for b in bolts)
    n1, n2 = (_unit(b.axis.array()) for b in bolts)
    load = request.load_point.position.array()
    load_axis = _unit(request.load_point.axis.array())
    force = request.force.array()

    if n1 @ n2 < np.cos(ANGLE_TOLERANCE):
        issues.append(inconsistent("mounting_points", "the two bolt axes are not parallel"))
        return None, issues
    normal = _unit(n1 + n2)
    spacing = float(np.linalg.norm(p2 - p1))
    if spacing < POSITION_TOLERANCE:
        issues.append(inconsistent("mounting_points", "the two bolt holes are at one place"))
        return None, issues
    bolt_line = (p2 - p1) / spacing
    if abs((p2 - p1) @ normal) > POSITION_TOLERANCE:
        issues.append(inconsistent("mounting_points", "the bolt holes are not on one surface"))
    if abs(load_axis @ normal) > np.sin(ANGLE_TOLERANCE):
        issues.append(
            inconsistent("load_point", "the load hole axis must be parallel to the support")
        )

    middle = (p1 + p2) / 2
    height = float((load - middle) @ normal)
    in_plane = (middle - load) + height * normal  # from the load hole foot to the bolts
    offset = float((load - middle) @ bolt_line)
    if height <= 0:
        issues.append(inconsistent("load_point", "the load hole must be above the support"))
    if abs(offset) > POSITION_TOLERANCE:
        issues.append(
            inconsistent(
                "load_point",
                f"the load hole is {offset:.1f} mm off the mid plane of the bolts, "
                "the bracket model is symmetric",
            )
        )
    bolt_distance = float(np.linalg.norm(in_plane))
    if bolt_distance < POSITION_TOLERANCE:
        issues.append(inconsistent("load_point", "the load hole is right above the bolts"))
        return None, issues
    x_axis = in_plane / bolt_distance
    if abs(load_axis @ x_axis) < np.cos(ANGLE_TOLERANCE):
        issues.append(
            inconsistent(
                "load_point",
                "the load hole axis must point from the load hole towards the bolts",
            )
        )
    if abs(bolt_line @ x_axis) > np.sin(ANGLE_TOLERANCE):
        issues.append(
            inconsistent(
                "mounting_points", "the bolt line must be perpendicular to the load hole axis"
            )
        )
    if force @ force == 0:
        issues.append(inconsistent("force", "the load is zero"))
    if issues:
        return None, issues

    z_axis = normal
    y_axis = np.cross(z_axis, x_axis)
    origin = load - height * normal
    d_bolt, d_load = diameters[0], request.load_point.hole_diameter
    fixed = {
        "bolt_hole_diameter": d_bolt,
        "bolt_hole_spacing": spacing,
        "bolt_hole_position": bolt_distance,
        "base_length": bolt_distance + END_MARGIN * d_bolt,
        "load_hole_diameter": d_load,
        "load_hole_height": height,
        "flange_height": height + END_MARGIN * d_load,
    }
    interface = BracketInterface(
        origin=tuple(origin),
        x_axis=tuple(x_axis),
        y_axis=tuple(y_axis),
        z_axis=tuple(z_axis),
        fixed=fixed,
        force_local=tuple(np.array([x_axis, y_axis, z_axis]) @ force),
    )

    # Self check: the bracket holes must land exactly on the requested ones.
    local_bolts = [(bolt_distance, -spacing / 2, 0.0), (bolt_distance, spacing / 2, 0.0)]
    placed = sorted(tuple(np.round(interface.to_global(p), 6)) for p in local_bolts)
    wanted = sorted(tuple(np.round(p, 6)) for p in (p1, p2))
    if not np.allclose(placed, wanted, atol=1e-6) or not np.allclose(
        interface.to_global((0.0, 0.0, height)), load, atol=1e-6
    ):
        raise AssertionError("bracket placement does not reproduce the requested holes")
    return interface, issues
