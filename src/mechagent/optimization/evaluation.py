"""Evaluate one bracket design against the requirements.

The evaluation separates two kinds of checks:
- verification: did the analyses run correctly and are their assumptions respected
  (solver, mesh, equilibrium, linearity, fatigue method validity)
- criteria: does the design meet the requirements (safety factors, displacement, mass)

A design is valid only when every check of both kinds passed. A required analysis that could
not run (no S-N data for a cyclic load, solver failure) makes the design invalid, never
silently skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from mechagent.cad.bracket import BracketParameters, build_bracket
from mechagent.cad.common import GeometryError
from mechagent.core.checks import Check, all_passed, check_above, check_below
from mechagent.core.config import SolversConfig
from mechagent.core.logging import get_logger
from mechagent.core.requirements import DesignRequirements
from mechagent.materials.cost import CostEstimate, CostModel, estimate_cost
from mechagent.materials.material import Material
from mechagent.simulation.bracket import bracket_static_problem
from mechagent.simulation.buckling import flange_compression, flange_critical_load
from mechagent.simulation.calculix import SolverError
from mechagent.simulation.fatigue import (
    FatigueResult,
    ReferenceStress,
    blocks_from_history,
    constant_amplitude,
    evaluate_fatigue,
)
from mechagent.simulation.mesh import MeshError, MeshSettings
from mechagent.simulation.static import StaticResult, load_fields, run_static_analysis

log = get_logger(__name__)


class DesignEvaluation(BaseModel):
    params: BracketParameters
    material: str
    run_dir: Path
    mass_kg: float | None = None
    cost: CostEstimate | None = None
    static: StaticResult | None = None
    static_reversed: StaticResult | None = None  # reversed force, only for reversing pin loads
    fatigue: FatigueResult | None = None
    verification: list[Check] = []
    criteria: list[Check] = []
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and all_passed(self.verification) and all_passed(self.criteria)

    @property
    def failed_checks(self) -> list[str]:
        return [c.name for c in self.verification + self.criteria if not c.passed]

    def objective_value(self, objective: str) -> float | None:
        if objective == "mass":
            return self.mass_kg
        return self.cost.total if self.cost else None


def fatigue_blocks(requirements: DesignRequirements):
    fatigue = requirements.fatigue
    if fatigue.cycles is not None:
        return constant_amplitude(fatigue.load_ratio, fatigue.cycles)
    return blocks_from_history(fatigue.history, fatigue.history_repeats)


def _buckling_criterion(
    params: BracketParameters, material: Material, requirements: DesignRequirements
) -> Check:
    min_factor = requirements.fatigue.min_load_factor if requirements.fatigue else 1.0
    compression = flange_compression(requirements.force[2], min_factor)
    critical = flange_critical_load(params, material)
    factor = critical / compression if compression > 0 else float("inf")
    return check_above(
        "buckling_factor",
        factor,
        requirements.min_buckling_factor,
        message="Euler fixed-free column screening of the flange (conservative)",
    )


def _static_criteria(
    results: list[StaticResult], requirements: DesignRequirements, mass: float
) -> list[Check]:
    safety = min(r.yield_safety_factor for r in results)
    displacement = max(r.max_displacement for r in results)
    criteria = [
        check_above(
            "static_safety_factor",
            safety,
            requirements.min_safety_factor,
            message="yield strength / design von Mises stress",
        )
    ]
    if requirements.max_displacement is not None:
        criteria.append(
            check_below("max_displacement", displacement, requirements.max_displacement, "mm")
        )
    if requirements.max_mass is not None:
        criteria.append(check_below("max_mass", mass, requirements.max_mass, "kg"))
    return criteria


def evaluate_bracket(
    params: BracketParameters,
    material: Material,
    requirements: DesignRequirements,
    run_dir: Path,
    mesh: MeshSettings | None = None,
    solvers: SolversConfig | None = None,
    cost_model: CostModel | None = None,
) -> DesignEvaluation:
    evaluation = DesignEvaluation(params=params, material=material.name, run_dir=run_dir)
    try:
        geometry = build_bracket(params, run_dir / "geometry")
    except GeometryError as exc:
        evaluation.error = f"CAD failed: {exc}"
        return evaluation

    evaluation.mass_kg = geometry.volume * material.density * 1000.0
    if material.cost_per_kg is not None and material.removal_rate is not None:
        evaluation.cost = estimate_cost(geometry, material, cost_model)
    elif requirements.objective == "cost":
        evaluation.error = f"material {material.name} has no cost data"
        return evaluation

    fatigue = requirements.fatigue
    reversed_pin = fatigue is not None and fatigue.min_load_factor < 0
    force = np.array(requirements.force, dtype=float)
    try:
        problem = bracket_static_problem(geometry, params, material, tuple(force), mesh)
        evaluation.static = run_static_analysis(problem, run_dir / "static", solvers)
        if reversed_pin:
            reversed_problem = bracket_static_problem(
                geometry, params, material, tuple(-force), mesh
            )
            evaluation.static_reversed = run_static_analysis(
                reversed_problem, run_dir / "static_reversed", solvers
            )
    except (MeshError, SolverError) as exc:
        evaluation.error = f"analysis failed: {exc}"
        log.warning("design evaluation failed in %s: %s", run_dir, exc)
        return evaluation

    statics = [r for r in (evaluation.static, evaluation.static_reversed) if r is not None]
    evaluation.verification = list(evaluation.static.checks)
    if evaluation.static_reversed is not None:
        evaluation.verification += [
            c.model_copy(update={"name": f"reversed_{c.name}"})
            for c in evaluation.static_reversed.checks
        ]
    evaluation.criteria = _static_criteria(statics, requirements, evaluation.mass_kg)
    if requirements.min_buckling_factor is not None:
        evaluation.criteria.append(_buckling_criterion(params, material, requirements))

    if fatigue is not None:
        if material.fatigue is None:
            evaluation.verification.append(
                Check(name="fatigue_data", passed=False, message="no S-N data, cyclic load")
            )
            return evaluation

        fields = load_fields(evaluation.static.run_dir)
        negative = None
        if evaluation.static_reversed is not None:
            reversed_fields = load_fields(evaluation.static_reversed.run_dir)
            if not np.array_equal(reversed_fields.coords, fields.coords):
                evaluation.error = "the reversed load case was meshed differently"
                return evaluation
            negative = reversed_fields.stress

        evaluation.fatigue = evaluate_fatigue(
            ReferenceStress(positive=fields.stress, negative=negative),
            fields.coords,
            fields.evaluated,
            fatigue_blocks(requirements),
            material,
            fatigue.settings,
            fatigue.allowable_damage,
        )
        damage_check = next(c for c in evaluation.fatigue.checks if c.name == "fatigue_damage")
        evaluation.verification += [c for c in evaluation.fatigue.checks if c is not damage_check]
        evaluation.criteria += [
            damage_check,
            check_above(
                "fatigue_safety_factor",
                evaluation.fatigue.safety_factor,
                fatigue.min_safety_factor,
                message="load factor bringing the damage to the allowable value",
            ),
        ]
    return evaluation
