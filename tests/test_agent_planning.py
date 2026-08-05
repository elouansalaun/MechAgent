from mechagent.agent.planning import plan_analyses
from mechagent.agent.review import review_request
from mechagent.core.request import Vector
from mechagent.materials.database import load_materials
from tests.test_agent_review import point, wall_request

MATERIALS = load_materials()


def planned(**overrides):
    request = wall_request(**overrides)
    review = review_request(request, MATERIALS)
    return plan_analyses(request, review.interface, review.requirements)


def by_name(plan):
    return {a.analysis: a for a in plan.analyses}


def test_cyclic_bending_load_needs_static_and_fatigue_only():
    plan, requirements = planned()

    analyses = by_name(plan)
    assert analyses["static"].required and analyses["fatigue"].required
    assert not analyses["buckling"].required and not analyses["modal"].required
    assert "1e+06 cycles" in analyses["fatigue"].reason
    assert plan.unsupported == []
    assert requirements.min_buckling_factor is None


def test_static_load_needs_no_fatigue():
    plan, _ = planned(load_type="static", cycles=None, load_ratio=None)

    assert not by_name(plan)["fatigue"].required


def test_compressed_flange_gets_a_buckling_screening():
    # Bracket on the floor, pushed down through its flange.
    plan, requirements = planned(
        mounting_points=[point((50, -18, 0), (0, 0, 1), 9.0), point((50, 18, 0), (0, 0, 1), 9.0)],
        load_point=point((0, 0, 60), (1, 0, 0), 12.0),
        force=Vector(x=0, y=0, z=-2000),
    )

    assert by_name(plan)["buckling"].required
    assert "2000 N in compression" in by_name(plan)["buckling"].reason
    assert requirements.min_buckling_factor == 3.0


def test_reversed_load_compresses_a_pulled_flange():
    plan, _ = planned(
        mounting_points=[point((50, -18, 0), (0, 0, 1), 9.0), point((50, 18, 0), (0, 0, 1), 9.0)],
        load_point=point((0, 0, 60), (1, 0, 0), 12.0),
        force=Vector(x=0, y=0, z=2000),
        load_ratio=-0.5,
    )

    assert "1000 N in compression" in by_name(plan)["buckling"].reason


def test_vibration_needs_an_unavailable_modal_analysis():
    plan, _ = planned(excitation_frequency=120.0)

    assert [a.analysis for a in plan.unsupported] == ["modal"]
    assert "120 Hz" in by_name(plan)["modal"].reason
