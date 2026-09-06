from mechagent.agent import cli
from tests.test_agent_review import wall_request


def test_cli_runs_a_structured_request(tmp_path, monkeypatch):
    calls = {}

    def fake_run(work_dir, tools, llm=None, text=None, request=None, config=None):
        calls.update(work_dir=work_dir, request=request, text=text, config=config)
        return {"status": "verified", "status_reason": "fine"}

    monkeypatch.setattr(cli, "run_agent", fake_run)
    monkeypatch.setattr(cli, "keep_awake", lambda: None)
    request_file = tmp_path / "request.json"
    request_file.write_text(wall_request().model_dump_json())

    code = cli.main(
        ["--request", str(request_file), "--work-dir", str(tmp_path / "run"), "--trials", "7"]
    )

    assert code == 0
    assert calls["request"] == wall_request()
    assert calls["text"] is None and calls["config"].n_trials == 7
