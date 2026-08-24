import numpy as np
import pytest

from mechagent.agent.interface import map_bracket_interface
from mechagent.agent.review import (
    numbers_in_text,
    review_request,
    unverified_values,
    unverified_vectors,
)
from mechagent.cad.bracket import BracketParameters
from mechagent.core.request import EngineeringRequest, InterfacePoint, Vector
from mechagent.materials.database import load_materials

MATERIALS = load_materials()


def point(position, axis, diameter=None):
    return InterfacePoint(
        position=Vector(x=position[0], y=position[1], z=position[2]),
        axis=Vector(x=axis[0], y=axis[1], z=axis[2]),
        hole_diameter=diameter,
    )


def wall_request(**overrides) -> EngineeringRequest:
    """Shelf bracket on a vertical wall (x = 0), loaded by gravity."""
    data = dict(
        mounting_points=[
            point((0, -18, 200), (1, 0, 0), 9.0),
            point((0, 18, 200), (1, 0, 0), 9.0),
        ],
        load_point=point((60, 0, 150), (0, 0, 1), 12.0),
        force=Vector(x=0, y=0, z=-2000),
        load_type="cyclic",
        cycles=1e6,
        load_ratio=0.0,
        min_safety_factor=2.0,
        max_displacement=0.5,
        objective="mass",
        candidate_materials=["aluminium", "Ti-6Al-4V"],
    )
    data.update(overrides)
    return EngineeringRequest(**data)


def test_wall_bracket_frame_and_fixed_parameters():
    interface, issues = map_bracket_interface(wall_request())

    assert issues == []
    assert interface.z_axis == pytest.approx((1, 0, 0))  # out of the wall
    assert interface.x_axis == pytest.approx((0, 0, 1))  # from the load hole to the bolts
    assert interface.y_axis == pytest.approx((0, -1, 0))
    assert interface.fixed["bolt_hole_spacing"] == pytest.approx(36.0)
    assert interface.fixed["bolt_hole_position"] == pytest.approx(50.0)
    assert interface.fixed["load_hole_height"] == pytest.approx(60.0)
    # Gravity pulls the load hole away from the bolts, along local -x.
    assert interface.force_local == pytest.approx((-2000, 0, 0))


def test_placed_bracket_holes_match_the_request():
    interface, _ = map_bracket_interface(wall_request())
    params = BracketParameters(**interface.fixed)

    bolt = interface.to_global((params.bolt_hole_position, params.bolt_hole_spacing / 2, 0))
    load = interface.to_global((0, 0, params.load_hole_height))

    assert any(np.allclose(bolt, wanted) for wanted in ((0, -18, 200), (0, 18, 200)))
    assert load == pytest.approx((60, 0, 150))
    assert np.allclose(interface.to_local(load), (0, 0, 60))


def test_arbitrary_rotation_round_trip():
    angle = np.radians(30)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    shift = np.array([100.0, -40.0, 25.0])

    def moved(p):
        return tuple(rotation @ np.array(p) + shift)

    request = wall_request(
        mounting_points=[
            point(moved((50, -18, 0)), tuple(rotation @ (0, 0, 1)), 9.0),
            point(moved((50, 18, 0)), tuple(rotation @ (0, 0, 1)), 9.0),
        ],
        load_point=point(moved((0, 0, 60)), tuple(rotation @ (1, 0, 0)), 12.0),
        force=Vector(**dict(zip("xyz", rotation @ (0, 0, -2000), strict=True))),
    )

    interface, issues = map_bracket_interface(request)

    assert issues == []
    assert interface.fixed["bolt_hole_position"] == pytest.approx(50.0)
    assert interface.force_local == pytest.approx((0, 0, -2000))


@pytest.mark.parametrize(
    "overrides, field, text",
    [
        ({"mounting_points": []}, "mounting_points", "exactly two"),
        ({"load_point": None}, "load_point", "needed"),
        (
            {
                "mounting_points": [
                    point((0, -18, 200), (1, 0, 0), 9.0),
                    point((0, 18, 200), (0.9, 0.44, 0), 9.0),
                ]
            },
            "mounting_points",
            "not parallel",
        ),
        ({"load_point": point((60, 5, 150), (0, 0, 1), 12.0)}, "load_point", "mid plane"),
        ({"load_point": point((60, 0, 150), (1, 0, 0), 12.0)}, "load_point", "parallel"),
        ({"load_point": point((-60, 0, 150), (0, 0, 1), 12.0)}, "load_point", "above"),
        (
            {
                "mounting_points": [
                    point((0, -18, 200), (1, 0, 0), 9.0),
                    point((0, 18, 200), (1, 0, 0), 11.0),
                ]
            },
            "mounting_points",
            "one diameter",
        ),
    ],
)
def test_interface_problems_are_reported(overrides, field, text):
    interface, issues = map_bracket_interface(wall_request(**overrides))

    assert interface is None
    assert any(i.field == field and text in i.message and i.blocking for i in issues)


def test_complete_request_gives_requirements_and_space():
    review = review_request(wall_request(), MATERIALS)

    assert review.blocking == []
    requirements = review.requirements
    assert requirements.force == pytest.approx((-2000, 0, 0))
    assert requirements.fatigue.cycles == 1e6
    assert requirements.candidate_materials == ["aluminium", "Ti-6Al-4V"]
    width = next(v for v in review.space.variables if v.name == "width")
    assert width.low == pytest.approx(36 + 2 * 9)
    assert review.space.fixed["load_hole_height"] == pytest.approx(60.0)


def test_defaults_are_recorded_as_assumptions():
    request = wall_request(
        load_ratio=None, min_safety_factor=None, objective=None, candidate_materials=None
    )

    review = review_request(request, MATERIALS)

    assert review.blocking == []
    fields = {i.field for i in review.issues if i.kind == "assumption"}
    assert {"load_ratio", "min_safety_factor", "objective", "candidate_materials"} <= fields
    assert review.requirements.fatigue.load_ratio == 0.0
    assert review.requirements.min_safety_factor == 1.5


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"load_type": None, "cycles": None}, "load_type"),
        ({"cycles": None}, "cycles"),
        ({"candidate_materials": ["unobtainium"]}, "candidate_materials"),
        ({"load_type": "static"}, "load_type"),
        ({"max_width": 40.0}, "max_width"),
    ],
)
def test_blocking_requirement_issues(overrides, field):
    review = review_request(wall_request(**overrides), MATERIALS)

    assert any(i.field == field and i.blocking for i in review.issues)
    assert review.requirements is None


def test_numbers_are_found_in_text_with_their_notations():
    text = "A 2,000 N load, 0,5 mm max, 1 million cycles, 2 kN peak, M8 bolts at 1e3 mm."

    assert {2000.0, 0.5, 1e6, 1000.0} <= numbers_in_text(text)
    assert 8.0 not in numbers_in_text(text)


def test_numbers_not_in_the_text_must_be_confirmed():
    text = (
        "Bolts at (0, -18, 200) and (0, 18, 200), normal (1, 0, 0), 9 mm holes. "
        "Pin hole 12 mm at (60, 0, 150), axis vertical. 2000 N downward, cyclic, "
        "1 million cycles, R = 0, safety factor 2, max deflection 0.5 mm."
    )
    request = wall_request(max_displacement=0.3)  # the parser got this one wrong

    issues = unverified_values(request, text)

    assert [i.field for i in issues] == ["max_displacement"]
    assert issues[0].blocking


EXPLICIT_TEXT = (
    "Global frame: z points up. Two bolt holes of 9 mm at (0, -18, 200) mm and (0, 18, 200) "
    "mm, bolt axis (1, 0, 0). Pin hole 12 mm, axis (0, 0, 1), at (60, 0, 150) mm. Maximum "
    "load (0, 0, -2000) N, cyclic, 1 million cycles, R = 0. Safety factor 2, max deflection "
    "0.5 mm."
)


def test_explicit_vectors_are_verified():
    assert unverified_vectors(wall_request(), EXPLICIT_TEXT) == []


def test_direction_read_from_words_must_be_confirmed():
    text = EXPLICIT_TEXT.replace("axis (0, 0, 1)", "vertical axis")
    wrong = wall_request(load_point=point((60, 0, 150), (0, -1, 0), 12.0))

    issues = unverified_vectors(wrong, text)

    assert [i.field for i in issues] == ["load_point.axis"]
    assert "(0, -1, 0)" in issues[0].message


def test_force_direction_only_needs_the_same_direction():
    text = EXPLICIT_TEXT.replace("(0, 0, -2000) N", "of 2000 N along (0, 0, -1)")

    assert unverified_vectors(wall_request(), text) == []


def test_zero_axis_is_reported_as_missing():
    request = wall_request(
        mounting_points=[point((0, -18, 200), (0, 0, 0), 9.0), point((0, 18, 200), (0, 0, 0), 9.0)]
    )

    interface, issues = map_bracket_interface(request)

    assert interface is None
    assert all(i.kind == "missing" and i.field == "mounting_points" for i in issues)


def test_a_load_ratio_the_text_never_gives_must_be_confirmed():
    text = "Cyclic load of 2000 N, 1 million cycles, safety factor 2, 9 mm and 12 mm holes, 0.5 mm."
    request = wall_request()  # load_ratio 0.0, invented by the parser

    issues = unverified_values(request, text)

    assert [i.field for i in issues] == ["load_ratio"]
