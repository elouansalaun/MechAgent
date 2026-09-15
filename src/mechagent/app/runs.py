"""Agent runs on disk: listing them, launching one in the background, reading its progress."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mechagent.agent.summary import PROGRESS_FILE, SUMMARY_FILE, RunSummary, load_summary
from mechagent.core.request import EngineeringRequest
from mechagent.simulation.static import make_run_dir

PID_FILE = "agent.pid"


@dataclass
class RunInfo:
    path: Path
    status: str
    finished: bool
    alive: bool

    @property
    def label(self) -> str:
        state = self.status if self.finished else ("running" if self.alive else "stopped")
        return f"{self.path.name} [{state}]"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_info(path: Path) -> RunInfo:
    summary = path / SUMMARY_FILE
    progress = path / PROGRESS_FILE
    pid_file = path / PID_FILE
    alive = pid_file.exists() and _alive(int(pid_file.read_text()))
    if summary.exists():
        status = json.loads(summary.read_text()).get("status", "?")
        return RunInfo(path, status, finished=True, alive=alive)
    status = (
        json.loads(progress.read_text()).get("status", "running")
        if progress.exists()
        else "starting"
    )
    return RunInfo(path, status, finished=False, alive=alive)


def list_runs(base: Path) -> list[RunInfo]:
    """Agent runs, newest first."""
    if not base.exists():
        return []
    paths = [
        p
        for p in base.iterdir()
        if p.is_dir()
        and ((p / SUMMARY_FILE).exists() or (p / PROGRESS_FILE).exists() or (p / PID_FILE).exists())
    ]
    return [run_info(p) for p in sorted(paths, reverse=True)]


def read_progress(path: Path) -> dict:
    file = path / PROGRESS_FILE
    return json.loads(file.read_text()) if file.exists() else {"status": "starting", "trace": []}


def read_summary(path: Path) -> RunSummary | None:
    return load_summary(path)


def launch_run(
    base: Path,
    request: EngineeringRequest | None = None,
    text: str | None = None,
    trials: int = 40,
    iterations: int = 3,
    materials: int = 3,
) -> Path:
    """Start the agent in its own process, so it survives a browser refresh or the app."""
    if (request is None) == (text is None):
        raise ValueError("give a structured request or a text, not both")
    work_dir = make_run_dir(base, "agent")
    if request is not None:
        (work_dir / "request.json").write_text(request.model_dump_json(indent=1))
        source = ["--request", str(work_dir / "request.json")]
    else:
        (work_dir / "request.txt").write_text(text)
        source = ["--text", str(work_dir / "request.txt")]
    command = [
        sys.executable,
        "-m",
        "mechagent.agent.cli",
        *source,
        "--work-dir",
        str(work_dir),
        "--trials",
        str(trials),
        "--iterations",
        str(iterations),
        "--materials",
        str(materials),
    ]
    with open(work_dir / "console.log", "w") as console:
        process = subprocess.Popen(
            command, stdout=console, stderr=subprocess.STDOUT, start_new_session=True
        )
    (work_dir / PID_FILE).write_text(str(process.pid))
    return work_dir
