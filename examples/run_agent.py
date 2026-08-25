"""Phase 3 example: the agent designs a bracket from a text specification.

Run from the project root, with LM Studio serving the model of config/default.yaml:

    python examples/run_agent.py examples/requests/wall_bracket.txt
    python examples/run_agent.py examples/requests/vague_request.txt

The first request is complete. With 40 trials per material and per iteration it takes about
30 to 45 minutes. The second one lacks the interface geometry: the agent stops and asks.
"""

import argparse
from pathlib import Path

from mechagent.agent.graph import AgentConfig, run_agent
from mechagent.agent.llm import LMStudioLLM
from mechagent.agent.tools import EngineeringTools
from mechagent.core.config import load_settings
from mechagent.core.logging import setup_logging
from mechagent.core.system import keep_awake
from mechagent.simulation.static import make_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("request", type=Path, help="text file with the specification")
    parser.add_argument("--trials", type=int, default=40, help="trials per material")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--materials", type=int, default=3)
    args = parser.parse_args()

    keep_awake()
    settings = load_settings()
    setup_logging(settings.logging.level, settings.resolve(settings.logging.file))
    run_dir = make_run_dir(settings.resolve(settings.project.work_dir), "agent")

    state = run_agent(
        run_dir,
        EngineeringTools(solvers=settings.solvers),
        llm=LMStudioLLM(settings.llm),
        text=args.request.read_text(),
        config=AgentConfig(
            n_trials=args.trials, max_iterations=args.iterations, max_materials=args.materials
        ),
    )

    print(f"Status: {state['status']} ({state['status_reason']})")
    review = state.get("review")
    if review is not None and review.blocking:
        print("\nThe agent needs answers before designing:")
        for issue in review.blocking:
            print(f"  - {issue.message}")
    best = state.get("best")
    if best is not None and best.final is not None:
        final = best.final
        print(f"\nDesign: {final.material}, {final.mass_kg * 1000:.1f} g")
    print("\nDecision trace:")
    for entry in state["trace"]:
        print(f"  [{entry.source}] {entry.step}: {entry.decision}")
    print(f"\nReport: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
