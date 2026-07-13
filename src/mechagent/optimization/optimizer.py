"""Parametric optimisation of the bracket with Optuna.

For each candidate material we search the design space with a seeded TPE sampler. Every
criterion becomes an Optuna constraint (negative margin, <= 0 when satisfied) and the
verification checks add one more constraint counting the failed ones. Infeasible geometries
and failed analyses are kept as infeasible trials, never given fake results.

The search runs on a mesh whose element size never drops below `min_search_element_size`:
very thin trial designs would otherwise need meshes of several hundred thousand nodes. To
keep a buffer against this coarser mesh, criteria must be met with `search_margin` to spare.

The valid designs are then verified in order of objective, over all materials: a mesh
convergence study on the full resolution mesh and a full evaluation on a finer one. The
first design that passes is the result. Only then is it reported as valid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
from pydantic import BaseModel, ValidationError

from mechagent.cad.bracket import BracketParameters, build_bracket
from mechagent.core.config import SolversConfig
from mechagent.core.logging import get_logger
from mechagent.core.requirements import DesignRequirements
from mechagent.materials.database import load_materials
from mechagent.materials.material import Material
from mechagent.materials.selection import SelectionResult, rank_materials
from mechagent.optimization.evaluation import DesignEvaluation, evaluate_bracket, fatigue_blocks
from mechagent.simulation.bracket import bracket_static_problem, default_bracket_mesh
from mechagent.simulation.convergence import ConvergenceStudy, run_convergence_study
from mechagent.simulation.mesh import MeshSettings
from mechagent.simulation.static import remove_bulky_files

log = get_logger(__name__)

INFEASIBLE_OBJECTIVE = 1e6
MAX_CONSTRAINT = 10.0
MIN_SEARCH_ELEMENT_SIZE = 2.5  # mm
SEARCH_MARGIN = 0.03  # criteria must pass by 3 % during the search


class DesignVariable(BaseModel):
    name: str
    low: float
    high: float
    optional: bool = False  # the optimizer may also switch the feature off (value 0)


class DesignSpace(BaseModel):
    variables: list[DesignVariable]
    fixed: dict[str, float] = {}

    def suggest(self, trial: optuna.Trial) -> dict[str, float]:
        values = dict(self.fixed)
        for variable in self.variables:
            if variable.optional and not trial.suggest_categorical(
                f"{variable.name}_enabled", [True, False]
            ):
                values[variable.name] = 0.0
            else:
                values[variable.name] = trial.suggest_float(
                    variable.name, variable.low, variable.high
                )
        return values


# Hole positions come from the mounting interface, so they stay fixed.
DEFAULT_BRACKET_SPACE = DesignSpace(
    variables=[
        DesignVariable(name="base_thickness", low=3.0, high=14.0),
        DesignVariable(name="flange_thickness", low=3.0, high=14.0),
        DesignVariable(name="width", low=50.0, high=90.0),
        DesignVariable(name="fillet_radius", low=2.0, high=12.0),
        DesignVariable(name="rib_thickness", low=2.0, high=6.0, optional=True),
        DesignVariable(name="rib_size", low=20.0, high=60.0),
    ]
)


class TrialRecord(BaseModel):
    number: int
    params: dict[str, float]
    objective: float | None
    valid: bool  # every check passed, criteria with the search margin to spare
    constraints: list[float]
    failed_checks: list[str] = []
    error: str | None = None


class MaterialOptimization(BaseModel):
    material: str
    trials: list[TrialRecord]
    valid_designs: list[DesignEvaluation] = []  # sorted by objective, best first

    @property
    def best(self) -> DesignEvaluation | None:
        return self.valid_designs[0] if self.valid_designs else None

    @property
    def n_valid(self) -> int:
        return sum(t.valid for t in self.trials)


class OptimizationResult(BaseModel):
    requirements: DesignRequirements
    selection: SelectionResult
    materials: list[MaterialOptimization]
    chosen_material: str | None = None
    final: DesignEvaluation | None = None
    convergence: ConvergenceStudy | None = None
    valid: bool = False
    decisions: list[str] = []


def constraint_names(requirements: DesignRequirements) -> list[str]:
    names = ["static_safety_factor"]
    if requirements.max_displacement is not None:
        names.append("max_displacement")
    if requirements.max_mass is not None:
        names.append("max_mass")
    if requirements.fatigue is not None:
        names += ["fatigue_damage", "fatigue_safety_factor"]
    return names + ["verification"]


def search_mesh(params: BracketParameters, min_element_size: float) -> MeshSettings:
    mesh = default_bracket_mesh(params)
    if mesh.element_size >= min_element_size:
        return mesh
    return mesh.scaled(min_element_size / mesh.element_size)


def constraint_values(
    evaluation: DesignEvaluation, names: list[str], margin: float = 0.0
) -> list[float]:
    """Optuna constraints, <= 0 when every criterion passes with `margin` to spare."""
    if evaluation.error is not None:
        return [1.0] * len(names)
    criteria = {c.name: c for c in evaluation.criteria}
    values = []
    for name in names:
        if name == "verification":
            values.append(float(sum(not c.passed for c in evaluation.verification)))
        elif name in criteria and criteria[name].margin is not None:
            value = margin - criteria[name].margin
            values.append(float(np.clip(value, -MAX_CONSTRAINT, MAX_CONSTRAINT)))
        else:
            values.append(1.0)
    return values


def optimize_material(
    requirements: DesignRequirements,
    material: Material,
    work_dir: Path,
    n_trials: int = 30,
    space: DesignSpace = DEFAULT_BRACKET_SPACE,
    seed: int = 0,
    solvers: SolversConfig | None = None,
    min_search_element_size: float = MIN_SEARCH_ELEMENT_SIZE,
    search_margin: float = SEARCH_MARGIN,
) -> MaterialOptimization:
    names = constraint_names(requirements)
    records: list[TrialRecord] = []
    evaluations: dict[int, DesignEvaluation] = {}

    def set_constraints(trial: optuna.Trial, constraints: list[float]) -> None:
        for name, value in zip(names, constraints, strict=True):
            trial.set_constraint(name, value)

    def objective(trial: optuna.Trial) -> float:
        values = space.suggest(trial)
        try:
            params = BracketParameters(**values)
        except ValidationError as exc:
            constraints = [1.0] * len(names)
            set_constraints(trial, constraints)
            records.append(
                TrialRecord(
                    number=trial.number,
                    params=values,
                    objective=None,
                    valid=False,
                    constraints=constraints,
                    error=f"infeasible geometry: {exc.errors()[0]['msg']}",
                )
            )
            return INFEASIBLE_OBJECTIVE

        evaluation = evaluate_bracket(
            params,
            material,
            requirements,
            work_dir / f"trial_{trial.number:03d}",
            mesh=search_mesh(params, min_search_element_size),
            solvers=solvers,
        )
        for result in (evaluation.static, evaluation.static_reversed):
            if result is not None:
                remove_bulky_files(result.run_dir)

        constraints = constraint_values(evaluation, names, search_margin)
        set_constraints(trial, constraints)
        value = evaluation.objective_value(requirements.objective)
        records.append(
            TrialRecord(
                number=trial.number,
                params=values,
                objective=value,
                valid=evaluation.valid and max(constraints) <= 0,
                constraints=constraints,
                failed_checks=evaluation.failed_checks,
                error=evaluation.error,
            )
        )
        evaluations[trial.number] = evaluation
        return value if value is not None else INFEASIBLE_OBJECTIVE

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(5, n_trials // 3))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    valid = sorted((r for r in records if r.valid), key=lambda r: r.objective)
    optimization = MaterialOptimization(
        material=material.name,
        trials=records,
        valid_designs=[evaluations[r.number] for r in valid],
    )
    log.info(
        "%s: %d valid designs out of %d trials, best %s",
        material.name,
        len(valid),
        n_trials,
        f"{valid[0].objective:.4g}" if valid else "none",
    )
    return optimization


def _required_cycles(requirements: DesignRequirements) -> float | None:
    if requirements.fatigue is None:
        return None
    return sum(block.cycles for block in fatigue_blocks(requirements))


def verify_final_design(
    best: DesignEvaluation,
    material: Material,
    requirements: DesignRequirements,
    work_dir: Path,
    refinement_factor: float = 0.7,
    solvers: SolversConfig | None = None,
) -> tuple[ConvergenceStudy, DesignEvaluation]:
    """Mesh convergence on the chosen design, then a full evaluation on the finer mesh."""
    params = best.params
    mesh = default_bracket_mesh(params)
    geometry = build_bracket(params, work_dir / "geometry")
    problem = bracket_static_problem(geometry, params, material, requirements.force, mesh)
    study, _ = run_convergence_study(
        problem, work_dir / "convergence", refinement_factor, n_levels=2, solvers=solvers
    )
    final = evaluate_bracket(
        params,
        material,
        requirements,
        work_dir / "final",
        mesh=mesh.scaled(refinement_factor),
        solvers=solvers,
    )
    return study, final


def optimize_bracket_design(
    requirements: DesignRequirements,
    work_dir: Path,
    n_trials: int = 30,
    max_materials: int = 3,
    space: DesignSpace = DEFAULT_BRACKET_SPACE,
    seed: int = 0,
    materials: dict[str, Material] | None = None,
    solvers: SolversConfig | None = None,
    max_verifications: int = 3,
) -> OptimizationResult:
    materials = materials or load_materials()
    fatigue_settings = requirements.fatigue.settings if requirements.fatigue else None
    selection = rank_materials(
        materials,
        objective=requirements.objective,
        stiffness=requirements.max_displacement is not None,
        required_cycles=_required_cycles(requirements),
        fatigue_settings=fatigue_settings,
        candidates=requirements.candidate_materials,
    )
    result = OptimizationResult(requirements=requirements, selection=selection, materials=[])
    decisions = result.decisions
    decisions.append(
        f"Ranked {len(selection.ranking)} materials on {', '.join(selection.criteria)} "
        f"for minimum {requirements.objective} (score, limiting index): "
        + ", ".join(
            f"{r.name} ({r.score:.2f}, limited by {r.governing})" for r in selection.ranking
        )
    )
    for rejected in selection.rejected:
        decisions.append(f"Rejected {rejected.name}: {rejected.reason}.")

    shortlist = selection.ranking[:max_materials]
    for ranked in shortlist:
        optimization = optimize_material(
            requirements,
            materials[ranked.name],
            work_dir / ranked.name,
            n_trials,
            space,
            seed,
            solvers,
        )
        result.materials.append(optimization)
        if optimization.best is None:
            decisions.append(
                f"{ranked.name}: no valid design in {n_trials} trials "
                f"({len(optimization.trials) - optimization.n_valid} invalid)."
            )
        else:
            value = optimization.best.objective_value(requirements.objective)
            decisions.append(
                f"{ranked.name}: best valid design {_format(value, requirements.objective)} "
                f"({optimization.n_valid} valid designs out of {n_trials} trials)."
            )

    candidates = sorted(
        (design for m in result.materials for design in m.valid_designs),
        key=lambda design: design.objective_value(requirements.objective),
    )
    if not candidates:
        decisions.append(
            "No valid design was found. The design space, the number of trials or the "
            "requirements have to change before another iteration."
        )
        _save(result, work_dir)
        return result

    for rank, candidate in enumerate(candidates[:max_verifications], start=1):
        value = _format(candidate.objective_value(requirements.objective), requirements.objective)
        convergence, final = verify_final_design(
            candidate,
            materials[candidate.material],
            requirements,
            work_dir / f"final_design_{rank}",
            solvers=solvers,
        )
        failed = final.failed_checks + [c.name for c in convergence.checks if not c.passed]
        if final.error:
            failed.append(final.error)
        if not failed:
            result.chosen_material = candidate.material
            result.final, result.convergence, result.valid = final, convergence, True
            others = sorted({m.material for m in result.materials if m.best} - {candidate.material})
            best_word = "lightest" if requirements.objective == "mass" else "cheapest"
            decision = (
                f"Chose {candidate.material} ({value}), the {best_word} valid design that "
                "passed the final verification: mesh convergence and every check on the "
                "finer mesh."
            )
            if others:
                decision += f" Also optimised: {', '.join(others)}."
            decisions.append(decision)
            break
        decisions.append(
            f"Candidate {rank} ({candidate.material}, {value}) failed the final verification: "
            f"{', '.join(failed)}."
        )
    else:
        decisions.append(
            f"None of the {min(len(candidates), max_verifications)} best designs passed the "
            "final verification. Another iteration is needed."
        )
    _save(result, work_dir)
    return result


def _format(value: float, objective: str) -> str:
    return f"{value * 1000:.1f} g" if objective == "mass" else f"{value:.2f} EUR"


def _save(result: OptimizationResult, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "optimization.json").write_text(result.model_dump_json(indent=2))
