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
    MaterialOptimization,
    _close_alternatives,
    _closest_limit,
    constraint_names,
    constraint_values,
    optimize_bracket_design,
    optimize_material,
    search_mesh,
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
    buckling = REQUIREMENTS.model_copy(update={"min_buckling_factor": 3.0})
    assert constraint_names(buckling) == ["static_safety_factor", "buckling_factor", "verification"]
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

    # With a 10 % search margin the criterion has to pass by 10 %.
    passing = evaluation.model_copy(
        update={"criteria": [check_above("static_safety_factor", 2.1, 2.0)]}
    )
    assert constraint_values(passing, names[:1], margin=0.1) == pytest.approx([0.05])

    evaluation.error = "solver failed"
    assert constraint_values(evaluation, names) == [1.0, 1.0, 1.0]


def test_search_mesh_never_goes_below_the_floor():
    thin = BracketParameters(base_thickness=3.0, flange_thickness=3.2)
    thick = BracketParameters(base_thickness=10.0, flange_thickness=12.0)

    coarse = search_mesh(thin, 2.5)

    assert coarse.element_size == pytest.approx(2.5)
    assert coarse.refinements["load_hole"] == pytest.approx(1.25)
    assert search_mesh(thick, 2.5).element_size == pytest.approx(5.0)


def test_optimizer_respects_constraints(fake_fe, tmp_path):
    material = load_materials()["Al6061-T6"]

    result = optimize_material(REQUIREMENTS, material, tmp_path, n_trials=40, seed=1)

    best = result.best
    walls = best.params.base_thickness + best.params.flange_thickness
    assert best.valid
    assert 12.0 <= walls < 13.0  # close to the constrained optimum at 12 mm
    assert len(result.trials) == 40
    assert (tmp_path / "trials.json").exists()
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
    assert any("lightest valid design" in d and "S355J2" in d for d in result.decisions)
    assert any(d.startswith("Closest requirement: static_safety_factor") for d in result.decisions)
    assert any(
        d.startswith("S355J2: best valid") and "Closest requirement" in d for d in result.decisions
    )
    assert (tmp_path / "optimization.json").exists()


def test_next_candidate_is_verified_when_the_best_fails(fake_fe, monkeypatch, tmp_path):
    calls = []

    def first_fails(best, material, requirements, work_dir, **kwargs):
        calls.append(best.mass_kg)
        study, final = fake_verification(best, material, requirements, work_dir)
        if len(calls) == 1:
            final = final.model_copy(
                update={"criteria": [check_above("static_safety_factor", 1.9, 2.0)]}
            )
        return study, final

    monkeypatch.setattr(optimizer, "verify_final_design", first_fails)
    requirements = REQUIREMENTS.model_copy(update={"candidate_materials": ["Al6061-T6"]})

    result = optimize_bracket_design(requirements, tmp_path, n_trials=15, max_materials=1)

    assert result.valid
    assert len(calls) == 2 and calls[0] <= calls[1]
    assert any("Candidate 1" in d and "static_safety_factor" in d for d in result.decisions)


def test_no_candidate_passing_verification_is_reported(fake_fe, monkeypatch, tmp_path):
    def always_fails(best, material, requirements, work_dir, **kwargs):
        study, final = fake_verification(best, material, requirements, work_dir)
        return study, final.model_copy(update={"error": "mesh failed"})

    monkeypatch.setattr(optimizer, "verify_final_design", always_fails)
    requirements = REQUIREMENTS.model_copy(update={"candidate_materials": ["Al6061-T6"]})

    result = optimize_bracket_design(
        requirements, tmp_path, n_trials=10, max_materials=1, max_verifications=2
    )

    assert not result.valid and result.final is None
    assert "Another iteration is needed" in result.decisions[-1]


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


def test_solver_failure_during_verification_is_not_a_crash(monkeypatch, tmp_path):
    def broken_convergence(*args, **kwargs):
        raise optimizer.SolverError("TIMEOUT after 600 s")

    monkeypatch.setattr(optimizer, "run_convergence_study", broken_convergence)
    best = DesignEvaluation(params=BracketParameters(), material="Al6061-T6", run_dir=tmp_path)

    study, final = optimizer.verify_final_design(
        best, load_materials()["Al6061-T6"], REQUIREMENTS, tmp_path
    )

    assert study is None
    assert not final.valid and "TIMEOUT" in final.error


def test_closest_limit_flags_an_unconverged_search(tmp_path):
    design = DesignEvaluation(
        params=BracketParameters(),
        material="x",
        run_dir=tmp_path,
        criteria=[check_above("static_safety_factor", 3.0, 2.0), check_above("fatigue", 1.3, 1.0)],
    )

    text = _closest_limit(design)

    assert "fatigue with 30 % margin" in text and "not converged" in text
    tight = design.model_copy(update={"criteria": [check_above("fatigue", 1.05, 1.0)]})
    assert "not converged" not in _closest_limit(tight)


def test_close_alternatives_are_reported_with_their_cost(tmp_path):
    def design(material, mass):
        return DesignEvaluation(
            params=BracketParameters(), material=material, run_dir=tmp_path, mass_kg=mass
        )

    chosen = design("Ti-6Al-4V", 0.232)
    materials = [
        MaterialOptimization(material="Ti-6Al-4V", trials=[], valid_designs=[chosen]),
        MaterialOptimization(
            material="Al6061-T6", trials=[], valid_designs=[design("Al6061-T6", 0.234)]
        ),
        MaterialOptimization(material="S355J2", trials=[], valid_designs=[design("S355J2", 0.4)]),
    ]

    notes = _close_alternatives(chosen, materials, "mass")

    assert len(notes) == 1
    assert "Al6061-T6 at 234.0 g (+0.9 %)" in notes[0]


SPACE = DesignSpace(
    variables=[
        DesignVariable(name="base_thickness", low=3.0, high=14.0),
        DesignVariable(name="rib_thickness", low=2.0, high=6.0, optional=True),
    ]
)


def test_narrowed_space_stays_inside_the_original_bounds():
    narrowed = SPACE.narrowed({"base_thickness": 12.0, "rib_thickness": 5.0}, fraction=0.3)

    base, rib = narrowed.variables
    assert (base.low, base.high) == pytest.approx((8.4, 14.0))
    assert (rib.low, rib.high) == pytest.approx((3.5, 6.0))
    assert rib.optional


def test_narrowing_around_a_disabled_feature_keeps_its_range():
    narrowed = SPACE.narrowed({"base_thickness": 5.0, "rib_thickness": 0.0})

    assert narrowed.variables[1] == SPACE.variables[1]


def test_widened_space():
    widened = SPACE.widened(1.5)

    assert [v.high for v in widened.variables] == [21.0, 9.0]
    assert [v.low for v in widened.variables] == [3.0, 2.0]


def test_trial_params_for_optional_features():
    assert SPACE.trial_params({"base_thickness": 6.0, "rib_thickness": 0.0}) == {
        "base_thickness": 6.0,
        "rib_thickness_enabled": False,
    }
    assert SPACE.trial_params({"base_thickness": 6.0, "rib_thickness": 3.0}) == {
        "base_thickness": 6.0,
        "rib_thickness_enabled": True,
        "rib_thickness": 3.0,
    }


def test_warm_start_designs_are_evaluated_first(fake_fe, tmp_path):
    material = load_materials()["Al6061-T6"]
    start = {"base_thickness": 6.0, "flange_thickness": 6.5}
    space = DEFAULT_BRACKET_SPACE.model_copy(
        update={
            "variables": [
                DesignVariable(name="base_thickness", low=3.0, high=14.0),
                DesignVariable(name="flange_thickness", low=3.0, high=14.0),
            ]
        }
    )

    result = optimize_material(
        REQUIREMENTS, material, tmp_path, n_trials=3, space=space, start_from=[start]
    )

    assert result.trials[0].params["base_thickness"] == 6.0
    assert result.trials[0].params["flange_thickness"] == 6.5
    assert result.trials[0].valid  # 12.5 mm of walls meets the fake strength rule
    assert result.space == space
