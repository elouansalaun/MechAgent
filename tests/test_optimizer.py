from pathlib import Path

import optuna
import pytest

from mechagent.cad.bracket import BracketParameters
from mechagent.core.checks import Check, check_above
from mechagent.core.requirements import DesignRequirements, FatigueRequirement
from mechagent.materials.database import load_materials
from mechagent.optimization import optimizer
from mechagent.optimization.evaluation import DesignEvaluation
from mechagent.optimization.optimizer import (
    DEFAULT_BRACKET_SPACE,
    DesignSpace,
    DesignVariable,
    constraint_names,
    constraint_values,
    optimize_bracket_design,
    optimize_material,
)
from mechagent.simulation.convergence import ConvergenceStudy

REQUIREMENTS = DesignRequirements(force=(0, 0, -2000), min_safety_factor=2.0)


def fake_evaluation(params, material, requirements, run_dir, *args, **kwargs):
    """Stands in for the FE evaluation: mass grows with the walls, strength needs 12 mm."""
    walls = params.base_thickness + params.flange_thickness
    safety = 2.0 * walls / 12.0 * (2.0 if material.category == "steel" else 1.0)
    return DesignEvaluation(
        params=params,
        material=material.name,
        run_dir=run_dir,
        mass_kg=walls * (3.0 if material.category == "steel" else 1.0),
        verification=[Check(name="solver_success", passed=True)],
        criteria=[check_above("static_safety_factor", safety, requirements.min_safety_factor)],
    )


@pytest.fixture
def fake_fe(monkeypatch):
    monkeypatch.setattr(optimizer, "evaluate_bracket", fake_evaluation)


def test_design_space_suggestions():
    space = DesignSpace(
        variables=[
            DesignVariable(name="base_thickness", low=4, high=10),
            DesignVariable(name="rib_thickness", low=2, high=6, optional=True),
        ],
        fixed={"width": 70.0},
    )
    trial = optuna.trial.FixedTrial({"base_thickness": 5.0, "rib_thickness_enabled": False})

    assert space.suggest(trial) == {"width": 70.0, "base_thickness": 5.0, "rib_thickness": 0.0}


def test_constraint_names_follow_the_requirements():
    requirements = DesignRequirements(
        force=(0, 0, -1),
        max_displacement=0.5,
        fatigue=FatigueRequirement(cycles=1e6),
    )

    assert constraint_names(REQUIREMENTS) == ["static_safety_factor", "verification"]
    assert constraint_names(requirements) == [
        "static_safety_factor",
        "max_displacement",
        "fatigue_damage",
        "fatigue_safety_factor",
        "verification",
    ]


def test_constraint_values(tmp_path):
    evaluation = DesignEvaluation(
        params=BracketParameters(),
        material="x",
        run_dir=tmp_path,
        verification=[Check(name="a", passed=False), Check(name="b", passed=True)],
        criteria=[check_above("static_safety_factor", 1.5, 2.0)],
    )
    names = ["static_safety_factor", "max_displacement", "verification"]

    assert constraint_values(evaluation, names) == pytest.approx([0.25, 1.0, 1.0])

    evaluation.error = "solver failed"
    assert constraint_values(evaluation, names) == [1.0, 1.0, 1.0]


def test_optimizer_respects_constraints(fake_fe, tmp_path):
    material = load_materials()["Al6061-T6"]

    result = optimize_material(REQUIREMENTS, material, tmp_path, n_trials=40, seed=1)

    best = result.best
    walls = best.params.base_thickness + best.params.flange_thickness
    assert best.valid
    assert 12.0 <= walls < 13.0  # close to the constrained optimum at 12 mm
    assert len(result.trials) == 40
    assert all(t.valid == (max(t.constraints) <= 0) for t in result.trials if t.error is None)


def test_infeasible_geometries_are_recorded_not_evaluated(fake_fe, tmp_path):
    space = DesignSpace(
        variables=[DesignVariable(name="width", low=20.0, high=30.0)],  # always too narrow
    )
    material = load_materials()["Al6061-T6"]

    result = optimize_material(REQUIREMENTS, material, tmp_path, n_trials=3, space=space)

    assert result.best is None
    assert all("infeasible geometry" in t.error for t in result.trials)


def fake_verification(best, material, requirements, work_dir, **kwargs):
    study = ConvergenceStudy(
        levels=[],
        displacement_change=0.0,
        stress_change=0.0,
        checks=[Check(name="mesh_convergence_stress", passed=True)],
    )
    return study, best


def test_material_comparison_and_decision_trace(fake_fe, monkeypatch, tmp_path):
    monkeypatch.setattr(optimizer, "verify_final_design", fake_verification)
    requirements = REQUIREMENTS.model_copy(update={"candidate_materials": ["Al6061-T6", "S355J2"]})

    result = optimize_bracket_design(requirements, tmp_path, n_trials=15, max_materials=2)

    # Steel needs half the wall thickness but is three times heavier per mm in the fake model.
    assert result.chosen_material == "Al6061-T6"
    assert result.valid
    assert any("lowest mass" in d and "S355J2" in d for d in result.decisions)
    assert (tmp_path / "optimization.json").exists()


def test_no_valid_design_is_reported(fake_fe, tmp_path):
    requirements = REQUIREMENTS.model_copy(
        update={"min_safety_factor": 100.0, "candidate_materials": ["Al6061-T6"]}
    )

    result = optimize_bracket_design(requirements, tmp_path, n_trials=5, max_materials=1)

    assert not result.valid
    assert result.final is None and result.chosen_material is None
    assert "No valid design" in result.decisions[-1]


@pytest.mark.external
def test_real_trials_keep_records_and_drop_solver_files(tmp_path, require_ccx):
    material = load_materials()["S355J2"]
    requirements = DesignRequirements(force=(0, 0, -1000), min_safety_factor=1.5)
    space = DEFAULT_BRACKET_SPACE.model_copy(
        update={"variables": [DesignVariable(name="base_thickness", low=8.0, high=12.0)]}
    )

    result = optimize_material(requirements, material, tmp_path, n_trials=2, space=space)

    assert len(result.trials) == 2
    for trial in result.trials:
        static = Path(tmp_path / f"trial_{trial.number:03d}" / "static")
        assert (static / "result.json").exists()
        assert not (static / "job.frd").exists() and not (static / "fields.npz").exists()
        assert trial.objective is not None
