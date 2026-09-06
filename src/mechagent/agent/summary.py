"""A compact, serialisable record of an agent run, read back by the Streamlit app."""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel

from mechagent.agent.graph import TraceEntry
from mechagent.agent.planning import AnalysisPlan
from mechagent.agent.review import ReviewResult
from mechagent.core.request import EngineeringRequest
from mechagent.core.requirements import DesignRequirements
from mechagent.materials.selection import SelectionResult
from mechagent.optimization.optimizer import VerificationOutcome
from mechagent.simulation.validation import ValidationCase

SUMMARY_FILE = "summary.json"
PROGRESS_FILE = "progress.json"


class SearchSummary(BaseModel):
    iteration: int
    material: str
    trials: int
    valid: int
    best_objective: float | None
    smallest_margin: float | None
    best_params: dict[str, float] | None


class RunSummary(BaseModel):
    status: str
    status_reason: str
    text: str | None = None
    request: EngineeringRequest | None = None
    review: ReviewResult | None = None
    plan: AnalysisPlan | None = None
    requirements: DesignRequirements | None = None
    validation: list[ValidationCase] = []
    selection: SelectionResult | None = None
    searches: list[SearchSummary] = []
    best: VerificationOutcome | None = None
    trace: list[TraceEntry] = []
    deliverables: dict[str, str] = {}


def summarise(state) -> RunSummary:
    searches = []
    objective = state["requirements"].objective if state.get("requirements") else "mass"
    for iteration, round_ in enumerate(state.get("history") or [], start=1):
        for name, optimization in round_.items():
            best = optimization.best
            searches.append(
                SearchSummary(
                    iteration=iteration,
                    material=name,
                    trials=len(optimization.trials),
                    valid=optimization.n_valid,
                    best_objective=best.objective_value(objective) if best else None,
                    smallest_margin=min(c.margin for c in best.criteria) if best else None,
                    best_params=best.params.model_dump() if best else None,
                )
            )
    return RunSummary(
        status=state.get("status", "running"),
        status_reason=state.get("status_reason", ""),
        text=state.get("text"),
        request=state.get("request"),
        review=state.get("review"),
        plan=state.get("plan"),
        requirements=state.get("requirements"),
        validation=state.get("validation") or [],
        selection=state.get("selection"),
        searches=searches,
        best=state.get("best"),
        trace=state.get("trace") or [],
        deliverables=state.get("deliverables") or {},
    )


def write_progress(state, work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "status": state.get("status", "running"),
        "iteration": state.get("iteration", 0),
        "updated": time.time(),
        "trace": [entry.model_dump() for entry in state.get("trace") or []],
    }
    (work_dir / PROGRESS_FILE).write_text(json.dumps(progress, indent=1))


def load_summary(work_dir: Path) -> RunSummary | None:
    path = work_dir / SUMMARY_FILE
    return RunSummary.model_validate_json(path.read_text()) if path.exists() else None
