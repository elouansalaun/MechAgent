import math

import cadquery as cq
import pytest
from pydantic import ValidationError

from mechagent.cad.bracket import (
    BracketParameters,
    build_bracket,
    build_bracket_solid,
)
from mechagent.cad.common import GeometryError


def expected_volume(p: BracketParameters) -> float:
    """Hand calculation of the bracket volume."""
    volume = p.base_length * p.base_thickness * p.width
    volume += p.flange_thickness * (p.flange_height - p.base_thickness) * p.width
    volume += (1 - math.pi / 4) * p.fillet_radius**2 * p.width
    volume -= 2 * math.pi * (p.bolt_hole_diameter / 2) ** 2 * p.base_thickness
    volume -= math.pi * (p.load_hole_diameter / 2) ** 2 * p.flange_thickness
    if p.rib_thickness > 0:
        # The fillet already fills the corner of each triangular rib.
        rib_area = p.rib_size**2 / 2 - (1 - math.pi / 4) * p.fillet_radius**2
        volume += 2 * p.rib_thickness * rib_area
    return volume


@pytest.mark.parametrize(
    "overrides",
    [
        {"fillet_radius": 0.0},
        {},
        {"rib_thickness": 3.0, "rib_fillet_radius": 0.0},
        {
            "width": 80.0,
            "base_thickness": 10.0,
            "flange_thickness": 6.0,
            "rib_thickness": 4.0,
            "rib_fillet_radius": 0.0,
        },
    ],
)
def test_volume_matches_hand_calculation(overrides):
    params = BracketParameters(**overrides)

    solid = build_bracket_solid(params)

    assert solid.Volume() == pytest.approx(expected_volume(params), rel=1e-6)


def test_build_bracket_exports_brep_and_hole_faces(tmp_path):
    params = BracketParameters()

    geometry = build_bracket(params, tmp_path)

    assert geometry.brep_file.exists()
    assert set(geometry.face_groups) == {"bolt_holes", "load_hole"}

    bolt_area = sum(f.area for f in geometry.face_groups["bolt_holes"])
    load_area = sum(f.area for f in geometry.face_groups["load_hole"])
    p = params
    assert bolt_area == pytest.approx(2 * math.pi * p.bolt_hole_diameter * p.base_thickness)
    assert load_area == pytest.approx(math.pi * p.load_hole_diameter * p.flange_thickness)


def test_bounding_box_and_holes(tmp_path):
    params = BracketParameters()

    geometry = build_bracket(params, tmp_path)

    low, high = geometry.bounding_box
    assert low == pytest.approx((0, -30, 0), abs=1e-6)
    assert high == pytest.approx((80, 30, 80), abs=1e-6)
    assert geometry.hole("load_hole").center == (0.0, 0.0, 60.0)
    assert [h.name for h in geometry.holes] == ["bolt_hole_1", "bolt_hole_2", "load_hole"]


def test_rib_fillets_add_a_little_material_and_a_face_group(tmp_path):
    sharp = BracketParameters(rib_thickness=3.0, rib_fillet_radius=0.0)
    rounded = sharp.model_copy(update={"rib_fillet_radius": 2.0})

    added = build_bracket_solid(rounded).Volume() - build_bracket_solid(sharp).Volume()
    geometry = build_bracket(rounded, tmp_path)

    # Each concave edge gains at most (1 - pi/4) r^2 per unit length, the toes even less.
    r, s, t = 2.0, rounded.rib_size, rounded.rib_thickness
    edge_length = 2 * (2 * s + 2 * t)
    assert 0 < added < (1 - math.pi / 4) * r**2 * edge_length
    assert len(geometry.face_groups["rib_fillets"]) >= 10
    for face in geometry.face_groups["rib_fillets"]:
        assert abs(face.center[1]) > rounded.width / 2 - t - r - 1e-6


def test_no_rib_fillet_group_without_ribs(tmp_path):
    assert "rib_fillets" not in build_bracket(BracketParameters(), tmp_path).face_groups


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"bolt_hole_position": 15.0}, "flange or fillet"),
        ({"bolt_hole_position": 78.0}, "end of the base"),
        ({"bolt_hole_spacing": 56.0}, "side edges"),
        ({"load_hole_height": 78.0}, "top of the flange"),
        ({"fillet_radius": 40.0}, "fillet radius"),
        ({"rib_thickness": 5.0, "rib_size": 90.0}, "ribs longer"),
        ({"rib_thickness": 10.0, "bolt_hole_spacing": 36.0}, "side edges or ribs"),
        ({"rib_thickness": 3.0, "rib_size": 20.0, "rib_fillet_radius": 6.0}, "rib fillet"),
        # Found in an agent run: the toe ended 0.1 mm before the end of a 63.5 mm base.
        (
            {
                "base_length": 63.5,
                "flange_thickness": 8.2,
                "rib_thickness": 2.7,
                "rib_size": 55.2,
                "width": 70.0,
            },
            "ribs longer than the base",
        ),
    ],
)
def test_infeasible_parameters_are_rejected(overrides, message):
    with pytest.raises(ValidationError, match=message):
        BracketParameters(**overrides)


def test_opencascade_failures_become_geometry_errors(monkeypatch):
    def broken_fillet(self, *args, **kwargs):
        raise RuntimeError("BRep_API: command not done")

    monkeypatch.setattr(cq.Solid, "fillet", broken_fillet)

    with pytest.raises(GeometryError, match="OpenCascade"):
        build_bracket_solid(BracketParameters(rib_thickness=3.0, width=66.0))
