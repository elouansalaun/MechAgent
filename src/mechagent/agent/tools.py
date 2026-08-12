"""The deterministic engineering tools the agent calls. Tests replace them with fakes."""

from __future__ import annotations

from pathlib import Path

from mechagent.core.config import SolversConfig
from mechagent.core.requirements import DesignRequirements
from mechagent.materials.database import load_materials
from mechagent.materials.material import Material
from mechagent.materials.selection import SelectionResult
from mechagent.optimization.optimizer import (
    DesignSpace,
    MaterialOptimization,
    VerificationOutcome,
    choose_and_verify,
    optimize_material,
    select_materials,
)
from mechagent.simulation.validation import ValidationCase, run_validation_suite


class EngineeringTools:
    def __init__(
        self, materials: dict[str, Material] | None = None, solvers: SolversConfig | None = None
    ):
        self.materials = materials or load_materials()
        self.solvers = solvers

    def validate_solver(self, work_dir: Path) -> list[ValidationCase]:
        return run_validation_suite(work_dir, self.solvers)

    def select_materials(
        self, requirements: DesignRequirements
    ) -> tuple[SelectionResult, list[str]]:
        return select_materials(requirements, self.materials)

    def optimize_material(
        self,
        requirements: DesignRequirements,
        material: str,
        work_dir: Path,
        n_trials: int,
        space: DesignSpace,
        start_from: list[dict[str, float]],
    ) -> MaterialOptimization:
        return optimize_material(
            requirements,
            self.materials[material],
            work_dir,
            n_trials,
            space,
            solvers=self.solvers,
            start_from=start_from,
        )

    def choose_and_verify(
        self,
        searches: list[MaterialOptimization],
        requirements: DesignRequirements,
        work_dir: Path,
    ) -> VerificationOutcome:
        return choose_and_verify(searches, requirements, work_dir, self.materials, self.solvers)
