"""The engineering agent as a LangGraph state machine.

    parse -> review -> plan -> validate_solver -> select_materials -> search -> verify -> decide
                                                                         ^                  |
                                                                         +-- refine/widen --+
    accept, stop, or any blocking problem on the way -> report

The LLM reads free text specifications and chooses between the actions the rules allow after
each iteration. Everything else, and every number, comes from deterministic tools. The report
step recomputes the final status from the facts, so no path through the graph can call a
design verified without a verified design, a validated solver and every required analysis.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from mechagent.agent.llm import AgentLLM
from mechagent.agent.planning import AnalysisPlan, plan_analyses
from mechagent.agent.review import ReviewResult, review_request
from mechagent.agent.tools import EngineeringTools
from mechagent.core.logging import get_logger
from mechagent.core.request import EngineeringRequest
from mechagent.core.requirements import DesignRequirements
from mechagent.materials.selection import SelectionResult
from mechagent.optimization.optimizer import (
    CONVERGED_MARGIN,
    DesignSpace,
    MaterialOptimization,
    VerificationOutcome,
    describe_search,
)
from mechagent.simulation.validation import ValidationCase

log = get_logger(__name__)

Status = Literal[
    "running",
    "needs_input",
    "unverifiable",
    "solver_not_validated",
    "verified",
    "not_verified",
    "error",
]


class TraceEntry(BaseModel):
    step: str
    decision: str
    reason: str = ""
    source: Literal["rule", "llm", "tool", "user"]


class AgentConfig(BaseModel):
    n_trials: int = 40
    max_iterations: int = 3
    max_materials: int = 3
    refine_fraction: float = 0.3  # refined bounds: best design +/- 30 %
    keep_within: float = 0.15  # materials more than 15 % behind the best are dropped
    widen_factor: float = 1.5


class AgentState(TypedDict, total=False):
    text: str | None
    request: EngineeringRequest | None
    review: ReviewResult
    plan: AnalysisPlan
    requirements: DesignRequirements
    validation: list[ValidationCase]
    selection: SelectionResult
    shortlist: list[str]
    spaces: dict[str, DesignSpace]
    searches: dict[str, MaterialOptimization]
    history: list[dict[str, MaterialOptimization]]
    verification: VerificationOutcome | None
    best: VerificationOutcome | None
    iteration: int
    n_trials: int
    next_action: str
    status: Status
    status_reason: str
    trace: Annotated[list[TraceEntry], operator.add]
    report: str


def _objective(outcome: VerificationOutcome, objective: str) -> float:
    return outcome.final.objective_value(objective)


def closest_margin(outcome: VerificationOutcome) -> float:
    return min(c.margin for c in outcome.final.criteria)


def describe_outcome(outcome: VerificationOutcome) -> str:
    """Plain statement of the verified design for the LLM: what is met, by how much."""
    final = outcome.final
    lines = [f"Verified design: {final.material}, mass {final.mass_kg * 1000:.1f} g"]
    if final.cost:
        lines[0] += f", indicative cost {final.cost.total:.0f} EUR"
    lines.append("Every requirement is met. Value against limit for each requirement:")
    for check in final.criteria:
        lines.append(
            f"- {check.name}: {check.value:.4g} against a limit of {check.limit:.4g} "
            f"({100 * check.margin:.0f} % spare)"
        )
    return "\n".join(lines)


def iteration_options(state: AgentState, config: AgentConfig) -> tuple[list[str], str]:
    """Allowed next actions, the rule's choice first, and the situation in plain words."""
    best = state.get("best")
    searches = state["searches"]
    budget_left = state["iteration"] < config.max_iterations
    situation = [f"Iteration {state['iteration']} of at most {config.max_iterations}."]
    for optimization in searches.values():
        situation.append(describe_search(optimization, state["requirements"].objective))

    if best is not None and best.valid:
        situation.append(describe_outcome(best))
        if closest_margin(best) <= CONVERGED_MARGIN:
            situation.append("At least one requirement is within 10 % of its limit.")
            return ["accept"], "\n".join(situation)
        if not budget_left:
            return ["accept"], "\n".join(situation)
        situation.append(
            "No requirement is within 10 % of its limit: a lighter design probably exists."
        )
        return ["refine", "accept"], "\n".join(situation)
    if not budget_left:
        return ["stop"], "\n".join(situation)
    if not any(m.best for m in searches.values()):
        return ["widen_search", "stop"], "\n".join(situation)
    situation.append("Valid designs exist but none passed the final verification.")
    return ["refine", "stop"], "\n".join(situation)


def final_status(state: AgentState) -> tuple[Status, str]:
    """The status the facts support, whatever path the graph took."""
    review = state.get("review")
    if review is not None and review.blocking:
        return "needs_input", f"{len(review.blocking)} questions must be answered first"
    plan = state.get("plan")
    if plan is not None and plan.unsupported:
        names = ", ".join(a.analysis for a in plan.unsupported)
        return "unverifiable", f"required analyses not available: {names}"
    validation = state.get("validation")
    if validation is not None and not all(case.passed for case in validation):
        return "solver_not_validated", "an analytical validation case of the solver failed"
    best = state.get("best")
    if validation and best is not None and best.valid:
        return "verified", "a design passed every check, the convergence study and validation"
    if state.get("status") == "error":
        return "error", state.get("status_reason", "")
    return "not_verified", "no design passed every required verification"


def build_agent(
    llm: AgentLLM | None,
    tools: EngineeringTools,
    work_dir: Path,
    config: AgentConfig | None = None,
):
    config = config or AgentConfig()

    def parse(state: AgentState) -> AgentState:
        if state.get("request") is not None:
            return {
                "trace": [
                    TraceEntry(step="parse", decision="structured request given", source="user")
                ]
            }
        if not state.get("text") or llm is None:
            return {
                "status": "error",
                "status_reason": "no request: give a structured request, or text and an LLM",
            }
        try:
            request = llm.extract_request(state["text"])
        except Exception as exc:  # the LLM server can fail in many ways
            return {"status": "error", "status_reason": f"the LLM could not read the text: {exc}"}
        return {
            "request": request,
            "trace": [
                TraceEntry(
                    step="parse",
                    decision="request extracted from the text",
                    reason="every extracted number and vector is checked against the text",
                    source="llm",
                )
            ],
        }

    def review(state: AgentState) -> AgentState:
        result = review_request(state["request"], tools.materials, state.get("text"))
        trace = [
            TraceEntry(
                step="review",
                decision=("question: " if issue.blocking else "assumption: ") + issue.message,
                reason=f"{issue.kind} {issue.field}",
                source="rule",
            )
            for issue in result.issues
        ]
        update: AgentState = {"review": result, "trace": trace}
        if result.blocking:
            update["status"] = "needs_input"
        else:
            update["requirements"] = result.requirements
        return update

    def plan(state: AgentState) -> AgentState:
        review_result = state["review"]
        analysis_plan, requirements = plan_analyses(
            state["request"], review_result.interface, state["requirements"]
        )
        trace = [
            TraceEntry(
                step="plan",
                decision=f"{a.analysis}: {'required' if a.required else 'not required'}"
                + ("" if a.available or not a.required else " but NOT AVAILABLE"),
                reason=f"{a.reason}. Method: {a.method}" if a.required else a.reason,
                source="rule",
            )
            for a in analysis_plan.analyses
        ]
        update: AgentState = {"plan": analysis_plan, "requirements": requirements, "trace": trace}
        if analysis_plan.unsupported:
            update["status"] = "unverifiable"
        return update

    def validate_solver(state: AgentState) -> AgentState:
        cases = tools.validate_solver(work_dir / "validation")
        failed = [c for c in cases if not c.passed]
        trace = [
            TraceEntry(
                step="validate_solver",
                decision=f"{len(cases) - len(failed)} of {len(cases)} analytical reference "
                "cases passed",
                reason="results are only trusted once the solver chain reproduces known solutions",
                source="tool",
            )
        ]
        update: AgentState = {"validation": cases, "trace": trace}
        if failed:
            update["status"] = "solver_not_validated"
        return update

    def select(state: AgentState) -> AgentState:
        selection, decisions = tools.select_materials(state["requirements"])
        shortlist = [r.name for r in selection.ranking[: config.max_materials]]
        space = state["review"].space
        trace = [TraceEntry(step="select_materials", decision=d, source="tool") for d in decisions]
        trace.append(
            TraceEntry(
                step="select_materials",
                decision=f"shortlist: {', '.join(shortlist)}",
                reason=f"the {config.max_materials} best ranked materials are optimised",
                source="rule",
            )
        )
        if not shortlist:
            return {"selection": selection, "status": "not_verified", "trace": trace}
        return {
            "selection": selection,
            "shortlist": shortlist,
            "spaces": {name: space for name in shortlist},
            "searches": {},
            "history": [],
            "iteration": 0,
            "n_trials": config.n_trials,
            "trace": trace,
        }

    def search(state: AgentState) -> AgentState:
        iteration = state["iteration"] + 1
        previous = state["searches"]
        searches = {}
        trace = []
        for name in state["shortlist"]:
            best = previous.get(name).best if name in previous else None
            start = [best.params.model_dump()] if best else []
            optimization = tools.optimize_material(
                state["requirements"],
                name,
                work_dir / f"iteration_{iteration}" / name,
                state["n_trials"],
                state["spaces"][name],
                start,
            )
            searches[name] = optimization
            trace.append(
                TraceEntry(
                    step=f"search {iteration}",
                    decision=describe_search(optimization, state["requirements"].objective),
                    source="tool",
                )
            )
        return {
            "iteration": iteration,
            "searches": searches,
            "history": state["history"] + [searches],
            "trace": trace,
        }

    def verify(state: AgentState) -> AgentState:
        requirements = state["requirements"]
        outcome = tools.choose_and_verify(
            list(state["searches"].values()),
            requirements,
            work_dir / f"iteration_{state['iteration']}",
        )
        trace = [
            TraceEntry(step=f"verify {state['iteration']}", decision=d, source="tool")
            for d in outcome.decisions
        ]
        best = state.get("best")
        if outcome.valid and (
            best is None
            or _objective(outcome, requirements.objective)
            < _objective(best, requirements.objective)
        ):
            best = outcome
        return {"verification": outcome, "best": best, "trace": trace}

    def decide(state: AgentState) -> AgentState:
        options, situation = iteration_options(state, config)
        action, reason, source = options[0], "the only action the rules allow", "rule"
        if len(options) > 1:
            reason, source = "rule default", "rule"
            if llm is not None:
                try:
                    decision = llm.choose_action(situation, options)
                    if decision.action in options:
                        action, reason, source = decision.action, decision.reason, "llm"
                    else:
                        reason = f"the LLM answered '{decision.action}', not allowed: rule default"
                except Exception as exc:
                    reason = f"the LLM failed ({exc}): rule default"
        trace = [
            TraceEntry(
                step=f"decide {state['iteration']}",
                decision=f"{action} (allowed: {', '.join(options)})",
                reason=reason,
                source=source,
            )
        ]
        update: AgentState = {"next_action": action, "trace": trace}

        if action == "refine":
            update.update(_refine(state, config, trace))
        elif action == "widen_search":
            update["spaces"] = {
                n: s.widened(config.widen_factor) for n, s in state["spaces"].items()
            }
            update["n_trials"] = int(state["n_trials"] * config.widen_factor)
            trace.append(
                TraceEntry(
                    step=f"decide {state['iteration']}",
                    decision=f"upper bounds x{config.widen_factor}, {update['n_trials']} trials "
                    "per material",
                    source="rule",
                )
            )
        return update

    def report(state: AgentState) -> AgentState:
        from mechagent.reporting.markdown import render_report

        status, reason = final_status(state)
        final = {**state, "status": status, "status_reason": reason}
        text = render_report(final)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "report.md").write_text(text)
        return {"status": status, "status_reason": reason, "report": text}

    def stop_if_blocked(state: AgentState) -> str:
        return "report" if state.get("status", "running") != "running" else "next"

    def after_decision(state: AgentState) -> str:
        return "search" if state["next_action"] in ("refine", "widen_search") else "report"

    graph = StateGraph(AgentState)
    for name, node in [
        ("parse", parse),
        ("review", review),
        ("plan", plan),
        ("validate_solver", validate_solver),
        ("select_materials", select),
        ("search", search),
        ("verify", verify),
        ("decide", decide),
        ("report", report),
    ]:
        graph.add_node(name, node)
    graph.set_entry_point("parse")
    chain = ["parse", "review", "plan", "validate_solver", "select_materials", "search"]
    for current, following in zip(chain, chain[1:], strict=False):
        graph.add_conditional_edges(
            current, stop_if_blocked, {"report": "report", "next": following}
        )
    graph.add_edge("search", "verify")
    graph.add_edge("verify", "decide")
    graph.add_conditional_edges("decide", after_decision, {"search": "search", "report": "report"})
    graph.add_edge("report", END)
    return graph.compile()


def _refine(state: AgentState, config: AgentConfig, trace: list[TraceEntry]) -> AgentState:
    """Narrow each promising material's space around its best design, drop the others."""
    objective = state["requirements"].objective
    bests = {n: m.best for n, m in state["searches"].items() if m.best is not None}
    if not bests:
        return {}
    leader = min(b.objective_value(objective) for b in bests.values())
    kept = [
        n
        for n, b in bests.items()
        if b.objective_value(objective) <= leader * (1 + config.keep_within)
    ]
    dropped = sorted(set(state["shortlist"]) - set(kept))
    spaces = {
        n: state["spaces"][n].narrowed(bests[n].params.model_dump(), config.refine_fraction)
        for n in kept
    }
    reason = (
        f"{', '.join(dropped)} more than {100 * config.keep_within:.0f} % behind" if dropped else ""
    )
    trace.append(
        TraceEntry(
            step=f"decide {state['iteration']}",
            decision=f"refine {', '.join(kept)} within +/- {100 * config.refine_fraction:.0f} % "
            "of their best design, starting from it",
            reason=reason,
            source="rule",
        )
    )
    return {"shortlist": kept, "spaces": spaces}


def run_agent(
    work_dir: Path,
    tools: EngineeringTools,
    llm: AgentLLM | None = None,
    text: str | None = None,
    request: EngineeringRequest | None = None,
    config: AgentConfig | None = None,
) -> AgentState:
    agent = build_agent(llm, tools, work_dir, config)
    state: AgentState = {"text": text, "request": request, "status": "running", "trace": []}
    return agent.invoke(state, {"recursion_limit": 100})
