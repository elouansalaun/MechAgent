import json
import stat
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mechagent.core.environment import (
    EnvironmentReport,
    ToolStatus,
    check_executable,
    check_llm_server,
    check_python_package,
    format_report,
)


def make_script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_installed_package_is_detected():
    status = check_python_package("yaml", "pyyaml")

    assert status.available
    assert status.version


def test_missing_package_is_reported():
    status = check_python_package("surely_not_a_real_package_42")

    assert not status.available
    assert "ModuleNotFoundError" in status.detail


def test_executable_not_on_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))

    status = check_executable("calculix", "ccx", ["-v"])

    assert not status.available
    assert "not found" in status.detail


def test_executable_version_is_parsed(monkeypatch, tmp_path):
    make_script(tmp_path, "ccx", 'echo "This is Version 2.23"')
    monkeypatch.setenv("PATH", str(tmp_path))

    status = check_executable("calculix", "ccx", ["-v"])

    assert status.available
    assert status.version == "2.23"


def test_executable_without_version_is_not_trusted(monkeypatch, tmp_path):
    make_script(tmp_path, "ccx", 'echo "segmentation fault" >&2; exit 1')
    monkeypatch.setenv("PATH", str(tmp_path))

    status = check_executable("calculix", "ccx", ["-v"])

    assert not status.available
    assert "segmentation fault" in status.detail


@pytest.fixture
def fake_lmstudio():
    """A tiny HTTP server answering /v1/models like LM Studio does."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "qwen/qwen3.5-9b", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_llm_server_with_model(fake_lmstudio):
    assert check_llm_server(fake_lmstudio, "qwen/qwen3.5-9b").available


def test_llm_server_without_the_model(fake_lmstudio):
    status = check_llm_server(fake_lmstudio, "google/gemma-4-e4b")

    assert not status.available
    assert "not available" in status.detail


def test_llm_server_unreachable():
    status = check_llm_server("http://127.0.0.1:1/v1", "qwen/qwen3.5-9b", timeout=0.5)

    assert not status.available


def test_report_lists_missing_tools():
    report = EnvironmentReport(
        python_version="3.12.0",
        platform="test",
        packages=[ToolStatus(name="numpy", available=True, version="2.0")],
        executables=[ToolStatus(name="calculix", available=False, detail="not found")],
        services=[],
    )

    assert report.missing == ["calculix"]
    assert not report.all_available
    assert "Missing: calculix" in format_report(report)
