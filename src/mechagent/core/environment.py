"""Checks that the Python packages and external programs MechAgent relies on are available.

Run it with `mechagent-check-env` or `python -m mechagent.core.environment`.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

from pydantic import BaseModel

from mechagent.core.config import Settings, load_settings

# import name -> distribution name, when they differ
PYTHON_PACKAGES = {
    "pydantic": "pydantic",
    "yaml": "pyyaml",
    "numpy": "numpy",
    "scipy": "scipy",
    "cadquery": "cadquery",
    "gmsh": "gmsh",
    "optuna": "optuna",
    "pyvista": "pyvista",
    "langgraph": "langgraph",
    "langchain_ollama": "langchain-ollama",
    "streamlit": "streamlit",
    "pytest": "pytest",
}


class ToolStatus(BaseModel):
    name: str
    available: bool
    version: str | None = None
    location: str | None = None
    detail: str | None = None


class EnvironmentReport(BaseModel):
    python_version: str
    platform: str
    packages: list[ToolStatus]
    executables: list[ToolStatus]
    services: list[ToolStatus]

    @property
    def missing(self) -> list[str]:
        tools = self.packages + self.executables + self.services
        return [tool.name for tool in tools if not tool.available]

    @property
    def all_available(self) -> bool:
        return not self.missing


def check_python_package(import_name: str, dist_name: str | None = None) -> ToolStatus:
    dist_name = dist_name or import_name
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # a broken install can raise more than ImportError
        return ToolStatus(name=dist_name, available=False, detail=f"{type(exc).__name__}: {exc}")

    try:
        version = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", None)
    return ToolStatus(
        name=dist_name, available=True, version=version, location=getattr(module, "__file__", None)
    )


def _run_version_command(command: list[str], timeout: float = 15.0) -> tuple[str | None, str]:
    """Run a version command and return (first version number found, raw output)."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"

    output = (result.stdout + result.stderr).strip()
    match = re.search(r"\d+\.\d+(\.\d+)?", output)
    return (match.group(0) if match else None), output


def check_executable(name: str, executable: str, version_args: list[str]) -> ToolStatus:
    path = shutil.which(executable)
    if path is None:
        return ToolStatus(name=name, available=False, detail=f"'{executable}' not found on PATH")

    version, output = _run_version_command([path, *version_args])
    if version is None:
        # The program exists but we could not confirm it runs, so we do not count it as available.
        return ToolStatus(name=name, available=False, location=path, detail=output[:200])
    return ToolStatus(name=name, available=True, version=version, location=path)


def check_ollama_server(base_url: str, model: str, timeout: float = 3.0) -> ToolStatus:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return ToolStatus(name="ollama-server", available=False, location=base_url, detail=str(exc))

    models = [m.get("name", "") for m in payload.get("models", [])]
    if model not in models:
        return ToolStatus(
            name="ollama-server",
            available=False,
            location=base_url,
            detail=f"server is running but model '{model}' is not pulled (found: {models})",
        )
    return ToolStatus(name="ollama-server", available=True, location=base_url, detail=model)


def inspect_environment(settings: Settings | None = None) -> EnvironmentReport:
    settings = settings or load_settings()

    packages = [check_python_package(imp, dist) for imp, dist in PYTHON_PACKAGES.items()]
    executables = [
        check_executable("calculix", settings.solvers.calculix_executable, ["-v"]),
        check_executable("gmsh", settings.solvers.gmsh_executable, ["--version"]),
        check_executable("ollama", "ollama", ["--version"]),
    ]
    services = [check_ollama_server(settings.llm.base_url, settings.llm.model)]

    return EnvironmentReport(
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        packages=packages,
        executables=executables,
        services=services,
    )


def format_report(report: EnvironmentReport) -> str:
    lines = [f"Python {report.python_version} on {report.platform}", ""]
    for title, tools in [
        ("Python packages", report.packages),
        ("External programs", report.executables),
        ("Services", report.services),
    ]:
        lines.append(title)
        for tool in tools:
            mark = "OK " if tool.available else "MISSING"
            info = tool.version or ""
            if not tool.available and tool.detail:
                info = tool.detail
            lines.append(f"  [{mark:7}] {tool.name:18} {info}")
        lines.append("")

    if report.all_available:
        lines.append("Everything is available.")
    else:
        lines.append(f"Missing: {', '.join(report.missing)}")
    return "\n".join(lines)


def main() -> int:
    report = inspect_environment()
    print(format_report(report))
    return 0 if report.all_available else 1


if __name__ == "__main__":
    raise SystemExit(main())
