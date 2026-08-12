from pathlib import Path

import pytest

from mechagent.agent.graph import AgentConfig, final_status, run_agent
from mechagent.agent.llm import Decision
from mechagent.cad.bracket import BracketParameters
from mechagent.core.checks import Check, check_below
from mechagent.materials.database import load_materials
from mechagent.optimization.evaluation import DesignEvaluation
from mechagent.optimization.optimizer import (
    MaterialOptimization,
    TrialRecord,
    VerificationOutcome,
    select_materials,
)
from mechagent.simulation.validation import ValidationCase
from tests.test_agent_review import EXPLICIT_TEXT, wall_request

BASE_MASS = {"Al7075-T6": 0.21, "Al6061-T6": 0.20, "Ti-6Al-4V": 0.30}


class FakeTools:
    """Stands in for the engineering tools. Each search finds a design 10 % lighter than the
    previous one of the same material, with a smaller displacement margin."""

    def __init__(self, solver_ok=True, valid=True):
        self.materials = load_materials()
        self.solver_ok = solver_ok
        self.valid = valid
        self.calls = []
        self.searches = {}

    def validate_solver(self, work_dir):
        self.calls.append("validate_solver")
        return [
            ValidationCase(
                name="bar",
                quantity="stress",
                unit="MPa",
                reference_formula="F / A",
                reference=100.0,
                fem=100.0 if self.solver_ok else 120.0,
                tolerance=1e-3,
                analysis_verified=True,
                run_dir=work_dir,
            )
        ]

    def select_materials(self, requirements):
        self.calls.append("select_materials")
        return select_materials(requirements, self.materials)

    def optimize_material(self, requirements, material, work_dir, n_trials, space, start_from):
        self.calls.append(f"optimize {material}")
        count = self.searches[material] = self.searches.get(material, 0) + 1
        mass = BASE_MASS[material] * 0.9 ** (count - 1)
        displacement = 0.5 * (1 - 0.3 / count**2)  # 30 % margin, then 7.5 %, ...
        design = DesignEvaluation(
            params=BracketParameters(),
            material=material,
            run_dir=Path(work_dir),
            mass_kg=mass,
            verification=[Check(name="solver_success", passed=True)],
            criteria=[check_below("max_displacement", displacement, 0.5, "mm")],
        )
        trial = TrialRecord(number=0, params={}, objective=mass, valid=self.valid, constraints=[0])
        return MaterialOptimization(
            material=material,
            trials=[trial],
            valid_designs=[design] if self.valid else [],
        )

    def choose_and_verify(self, searches, requirements, work_dir):
        self.calls.append("choose_and_verify")
        designs = sorted((d for m in searches for d in m.valid_designs), key=lambda d: d.mass_kg)
        if not designs:
            return VerificationOutcome(decisions=["No valid design was found."])
        convergence_check = Check(name="mesh_convergence_stress", passed=True)
        from mechagent.simulation.convergence import ConvergenceStudy

        study = ConvergenceStudy(
            levels=[], displacement_change=0, stress_change=0, checks=[convergence_check]
        )
        return VerificationOutcome(
            chosen_material=designs[0].material,
            final=designs[0],
            convergence=study,
            valid=True,
            decisions=[f"Chose {designs[0].material}."],
        )


class FakeLLM:
    def __init__(self, request=None, action=None):
        self.request = request
        self.action = action
        self.situations = []

    def extract_request(self, text):
        return self.request

    def choose_action(self, situation, options):
        self.situations.append(situation)
        return Decision(action=self.action or options[0], reason="fake reasoning")


CONFIG = AgentConfig(n_trials=5, max_iterations=3, max_materials=3)


def test_rules_only_run_refines_then_accepts(tmp_path):
    tools = FakeTools()

    state = run_agent(tmp_path, tools, request=wall_request(), config=CONFIG)

    assert state["status"] == "verified", state["status_reason"]
    assert state["iteration"] == 2  # 30 % margin -> refine, 7.5 % -> accept
    decisions = [e.decision for e in state["trace"] if e.step.startswith("decide")]
    assert decisions[0].startswith("refine") and decisions[-1].startswith("accept")
    # Titanium is 50 % heavier than the leader, it is dropped before the second search.
    assert "optimize Ti-6Al-4V" in tools.calls[:6]
    assert tools.calls.count("optimize Ti-6Al-4V") == 1
    assert state["best"].final.material == "Al6061-T6"
    assert (tmp_path / "report.md").read_text().startswith("# MechAgent engineering report")


def test_llm_can_accept_early(tmp_path):
    llm = FakeLLM(action="accept")

    state = run_agent(tmp_path, FakeTools(), llm=llm, request=wall_request(), config=CONFIG)

    assert state["iteration"] == 1 and state["status"] == "verified"
    decision = next(e for e in state["trace"] if e.step == "decide 1")
    assert decision.source == "llm" and decision.reason == "fake reasoning"
    assert "Every requirement is met" in llm.situations[0]
    assert "against a limit of 0.5" in llm.situations[0]


def test_llm_answer_outside_the_options_is_ignored(tmp_path):
    llm = FakeLLM(action="declare_valid")

    state = run_agent(tmp_path, FakeTools(), llm=llm, request=wall_request(), config=CONFIG)

    decision = next(e for e in state["trace"] if e.step == "decide 1")
    assert decision.source == "rule" and "not allowed" in decision.reason
    assert decision.decision.startswith("refine")


def test_missing_information_stops_before_any_tool(tmp_path):
    tools = FakeTools()

    state = run_agent(tmp_path, tools, request=wall_request(load_point=None), config=CONFIG)

    assert state["status"] == "needs_input"
    assert tools.calls == []
    assert "Questions before the design can start" in state["report"]


def test_unavailable_analysis_makes_the_request_unverifiable(tmp_path):
    tools = FakeTools()

    state = run_agent(
        tmp_path, tools, request=wall_request(excitation_frequency=80.0), config=CONFIG
    )

    assert state["status"] == "unverifiable"
    assert "modal" in state["status_reason"]
    assert tools.calls == []


def test_failed_solver_validation_stops_the_design(tmp_path):
    tools = FakeTools(solver_ok=False)

    state = run_agent(tmp_path, tools, request=wall_request(), config=CONFIG)

    assert state["status"] == "solver_not_validated"
    assert tools.calls == ["validate_solver"]


def test_no_valid_design_widens_then_stops(tmp_path):
    tools = FakeTools(valid=False)

    state = run_agent(tmp_path, tools, request=wall_request(), config=CONFIG)

    assert state["status"] == "not_verified"
    assert state["iteration"] == 3
    decisions = [e.decision for e in state["trace"] if e.step.startswith("decide")]
    assert decisions[0].startswith("widen_search")
    assert decisions[-1].startswith("stop")
    assert state["n_trials"] == int(int(5 * 1.5) * 1.5)


def test_text_with_an_invented_value_needs_confirmation(tmp_path):
    invented = wall_request(max_displacement=0.3)
    tools = FakeTools()

    state = run_agent(
        tmp_path, tools, llm=FakeLLM(request=invented), text=EXPLICIT_TEXT, config=CONFIG
    )

    assert state["status"] == "needs_input"
    assert any("max_displacement = 0.3" in i.message for i in state["review"].blocking)
    assert tools.calls == []


def test_text_read_correctly_goes_through(tmp_path):
    state = run_agent(
        tmp_path,
        FakeTools(),
        llm=FakeLLM(request=wall_request(), action="accept"),
        text=EXPLICIT_TEXT,
        config=CONFIG,
    )

    assert state["status"] == "verified"
    assert state["trace"][0].source == "llm"


def test_status_guard_does_not_trust_a_design_without_validation(tmp_path):
    state = run_agent(tmp_path, FakeTools(), request=wall_request(), config=CONFIG)

    failed_validation = [c.model_copy(update={"fem": 150.0}) for c in state["validation"]]
    status, _ = final_status({**state, "validation": failed_validation})

    assert status == "solver_not_validated"
    assert final_status({**state, "validation": []})[0] == "not_verified"


def test_report_contains_every_section(tmp_path):
    state = run_agent(tmp_path, FakeTools(), request=wall_request(), config=CONFIG)
    report = state["report"]

    for title in (
        "## 1. Request",
        "## 3. Bracket placement",
        "## 4. Analysis plan",
        "## 5. Solver validation",
        "## 6. Material selection",
        "## 7. Optimisation",
        "## 8. Final design",
        "## 10. Decision trace",
        "## 11. Limitations",
    ):
        assert title in report
    assert "**Status: VERIFIED**" in report
    assert "(0, -18, 200)" in report or "(0, 18, 200)" in report


@pytest.mark.parametrize("missing", ["request", "llm"])
def test_text_without_llm_is_an_error(tmp_path, missing):
    state = run_agent(
        tmp_path, FakeTools(), text="some text" if missing == "llm" else None, config=CONFIG
    )

    assert state["status"] == "error"
