import pytest

from mechagent.cad.bracket import BracketParameters
from mechagent.core.requirements import DesignRequirements, FatigueRequirement
from mechagent.materials.database import load_materials
from mechagent.optimization.evaluation import evaluate_bracket
from mechagent.simulation.mesh import MeshSettings

pytestmark = pytest.mark.external

COARSE = MeshSettings(element_size=6.0, refinements={"bolt_holes": 3.0, "load_hole": 3.0})
STRONG = BracketParameters(base_thickness=12, flange_thickness=12, rib_thickness=3, fillet_radius=8)


@pytest.fixture(scope="module")
def materials():
    return load_materials()


def test_static_only_design_is_valid_when_criteria_hold(materials, tmp_path, require_ccx):
    requirements = DesignRequirements(force=(0, 0, -1000), min_safety_factor=1.5)

    evaluation = evaluate_bracket(STRONG, materials["S355J2"], requirements, tmp_path, COARSE)

    assert evaluation.valid, evaluation.failed_checks
    assert evaluation.fatigue is None and evaluation.static_reversed is None
    assert [c.name for c in evaluation.criteria] == ["static_safety_factor"]
    assert evaluation.cost.total > 0
    assert evaluation.objective_value("mass") == evaluation.mass_kg


def test_failed_criteria_make_the_design_invalid(materials, tmp_path, require_ccx):
    requirements = DesignRequirements(
        force=(0, 0, -2000), min_safety_factor=2.0, max_displacement=0.1, max_mass=0.05
    )

    evaluation = evaluate_bracket(
        BracketParameters(), materials["Al6061-T6"], requirements, tmp_path, COARSE
    )

    assert not evaluation.valid
    assert {"static_safety_factor", "max_displacement", "max_mass"} <= set(evaluation.failed_checks)
    assert all(c.margin < 0 for c in evaluation.criteria if not c.passed)


def test_reversing_pin_load_runs_the_reversed_case(materials, tmp_path, require_ccx):
    requirements = DesignRequirements(
        force=(0, 0, -1000),
        fatigue=FatigueRequirement(cycles=1e6, load_ratio=-1.0),
    )

    evaluation = evaluate_bracket(STRONG, materials["S355J2"], requirements, tmp_path, COARSE)

    assert evaluation.static_reversed is not None
    assert evaluation.static_reversed.reaction_force[2] == pytest.approx(-1000, abs=1e-2)
    assert any(c.name == "reversed_equilibrium" for c in evaluation.verification)
    names = {c.name for c in evaluation.criteria}
    assert {"fatigue_damage", "fatigue_safety_factor"} <= names
    # Fully reversed: the mean stress at the critical point is close to zero.
    fatigue = evaluation.fatigue
    assert abs(fatigue.critical_mean) < 0.05 * fatigue.critical_amplitude


def test_missing_sn_data_blocks_validation(materials, tmp_path, require_ccx):
    no_data = materials["S355J2"].model_copy(update={"fatigue": None})
    requirements = DesignRequirements(
        force=(0, 0, -500), fatigue=FatigueRequirement(cycles=1e6, load_ratio=0.0)
    )

    evaluation = evaluate_bracket(STRONG, no_data, requirements, tmp_path, COARSE)

    assert not evaluation.valid
    assert "fatigue_data" in evaluation.failed_checks
