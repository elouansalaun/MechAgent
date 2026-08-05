"""Which analyses a design needs, decided by fixed engineering rules.

| Situation                          | Analysis                                   |
| ---------------------------------- | ------------------------------------------ |
| any load                           | linear static FEM (strength, displacement) |
| cyclic load                        | S-N fatigue with mean stress, Miner        |
| flange in compression              | buckling screening (Euler column)          |
| vibration excitation               | modal analysis (not available)             |

A required analysis that we cannot run makes the design unverifiable: the agent stops
rather than hand over a design that was never checked against it.
"""

from __future__ import annotations

from pydantic import BaseModel

from mechagent.agent.interface import BracketInterface
from mechagent.core.request import EngineeringRequest
from mechagent.core.requirements import DesignRequirements
from mechagent.simulation.buckling import flange_compression

MIN_BUCKLING_FACTOR = 3.0


class AnalysisDecision(BaseModel):
    analysis: str
    required: bool
    available: bool
    method: str
    reason: str


class AnalysisPlan(BaseModel):
    analyses: list[AnalysisDecision]

    @property
    def unsupported(self) -> list[AnalysisDecision]:
        return [a for a in self.analyses if a.required and not a.available]

    def required(self, analysis: str) -> bool:
        return any(a.analysis == analysis and a.required for a in self.analyses)


def plan_analyses(
    request: EngineeringRequest, interface: BracketInterface, requirements: DesignRequirements
) -> tuple[AnalysisPlan, DesignRequirements]:
    """The plan, and the requirements completed with what the plan needs."""
    analyses = [
        AnalysisDecision(
            analysis="static",
            required=True,
            available=True,
            method="linear static FEM, CalculiX C3D10, mesh convergence on the final design",
            reason="every design is checked for yield"
            + (" and for the displacement limit" if requirements.max_displacement else ""),
        )
    ]

    fatigue = requirements.fatigue
    analyses.append(
        AnalysisDecision(
            analysis="fatigue",
            required=fatigue is not None,
            available=True,
            method="S-N curve with Marin factors, Goodman mean stress correction, Miner damage",
            reason=(
                f"the load is cyclic ({fatigue.cycles:g} cycles at R = {fatigue.load_ratio:g}), "
                "a static check alone does not show the part survives repeated loading"
                if fatigue
                else "the load is static"
            ),
        )
    )

    min_factor = fatigue.min_load_factor if fatigue else 1.0
    compression = flange_compression(interface.force_local[2], min_factor)
    analyses.append(
        AnalysisDecision(
            analysis="buckling",
            required=compression > 0,
            available=True,
            method=f"Euler fixed-free column screening of the flange, factor >= "
            f"{MIN_BUCKLING_FACTOR:g}, ribs ignored (conservative)",
            reason=(
                f"the flange carries up to {compression:.0f} N in compression"
                if compression > 0
                else "the flange is never in compression"
            ),
        )
    )

    frequency = request.excitation_frequency
    analyses.append(
        AnalysisDecision(
            analysis="modal",
            required=frequency is not None,
            available=False,
            method="modal analysis (not implemented)",
            reason=(
                f"the part is excited at {frequency:g} Hz, its natural frequencies must be checked"
                if frequency is not None
                else "no vibration excitation was given"
            ),
        )
    )

    plan = AnalysisPlan(analyses=analyses)
    if plan.required("buckling"):
        requirements = requirements.model_copy(update={"min_buckling_factor": MIN_BUCKLING_FACTOR})
    return plan, requirements
