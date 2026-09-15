import json
from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from mechagent.agent.graph import run_agent
from mechagent.app import runs
from tests.test_agent_graph import CONFIG, FakeTools
from tests.test_agent_review import wall_request

APP = Path(__file__).parents[1] / "src" / "mechagent" / "app" / "streamlit_app.py"


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Point the app at an empty runs folder through its configuration file."""
    base = tmp_path / "runs"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"project": {"work_dir": str(base)}}))
    monkeypatch.setenv("MECHAGENT_CONFIG", str(config))
    return base


def test_list_runs_reads_their_state(tmp_path):
    finished = tmp_path / "20260101_000000_1_agent"
    running = tmp_path / "20260102_000000_1_agent"
    finished.mkdir()
    running.mkdir()
    (tmp_path / "not_a_run").mkdir()
    (finished / "summary.json").write_text(json.dumps({"status": "verified"}))
    (running / "progress.json").write_text(json.dumps({"status": "running", "trace": []}))

    listed = runs.list_runs(tmp_path)

    assert [r.path.name for r in listed] == [running.name, finished.name]
    assert listed[1].finished and listed[1].status == "verified"
    assert not listed[0].finished and listed[0].label.endswith("[stopped]")


def test_launch_starts_the_cli_in_the_background(tmp_path, monkeypatch):
    started = {}

    class FakeProcess:
        pid = 4242

    def fake_popen(command, **kwargs):
        started.update(command=command, kwargs=kwargs)
        return FakeProcess()

    monkeypatch.setattr(runs.subprocess, "Popen", fake_popen)

    work_dir = runs.launch_run(tmp_path, request=wall_request(), trials=10)

    assert (work_dir / "agent.pid").read_text() == "4242"
    assert "mechagent.agent.cli" in started["command"]
    assert started["command"][started["command"].index("--trials") + 1] == "10"
    assert started["kwargs"]["start_new_session"]
    assert (work_dir / "request.json").exists()


def test_launch_needs_exactly_one_source(tmp_path):
    with pytest.raises(ValueError):
        runs.launch_run(tmp_path)
    with pytest.raises(ValueError):
        runs.launch_run(tmp_path, request=wall_request(), text="both")


def test_app_reviews_the_default_form(runs_dir):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception

    app.radio(key="mode").set_value("Form").run()
    app.button(key="review").click().run()

    assert not app.exception
    assert not app.error
    subheaders = [s.value for s in app.subheader]
    assert "Interpretation" in subheaders and "Analysis plan" in subheaders
    assert app.button(key="start")


def test_app_shows_a_finished_run(runs_dir):
    run = runs_dir / "20260101_000000_1_agent"
    run_agent(run, FakeTools(), request=wall_request(), config=CONFIG)

    app = AppTest.from_file(str(APP), default_timeout=60).run()

    assert not app.exception
    assert any("VERIFIED" in m.value for m in app.markdown)
    assert any("**Al6061-T6**, 180.0 g" in m.value for m in app.markdown)
