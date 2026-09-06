"""Run the agent from the command line, used by the Streamlit app for background runs.

python -m mechagent.agent.cli --text request.txt --work-dir runs/my_run
python -m mechagent.agent.cli --request request.json --work-dir runs/my_run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mechagent.agent.graph import AgentConfig, run_agent
from mechagent.agent.llm import LMStudioLLM
from mechagent.agent.tools import EngineeringTools
from mechagent.core.config import load_settings
from mechagent.core.logging import setup_logging
from mechagent.core.request import EngineeringRequest
from mechagent.core.system import keep_awake


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MechAgent bracket design run")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", type=Path, help="specification as plain text")
    source.add_argument("--request", type=Path, help="structured request as JSON")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--materials", type=int, default=3)
    args = parser.parse_args(argv)

    keep_awake()
    settings = load_settings()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(settings.logging.level, args.work_dir / "agent.log")

    request = None
    if args.request:
        request = EngineeringRequest.model_validate_json(args.request.read_text())
    state = run_agent(
        args.work_dir,
        EngineeringTools(solvers=settings.solvers),
        llm=LMStudioLLM(settings.llm),
        text=args.text.read_text() if args.text else None,
        request=request,
        config=AgentConfig(
            n_trials=args.trials, max_iterations=args.iterations, max_materials=args.materials
        ),
    )
    print(f"{state['status']}: {state['status_reason']}")
    return 0 if state["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
