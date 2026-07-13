"""Writing CalculiX input decks, running ccx and checking that it really succeeded."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from mechagent.core.logging import get_logger
from mechagent.materials.material import Material
from mechagent.simulation.mesh import Mesh
from mechagent.simulation.problem import Constraint

log = get_logger(__name__)

CONSTRAINED_SET = "CONSTRAINED"


class SolverError(RuntimeError):
    pass


class SolverRun(BaseModel):
    command: list[str]
    job_dir: Path
    returncode: int | None
    duration_s: float
    success: bool
    errors: list[str] = []
    warnings: list[str] = []
    log_file: Path


def _set_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).upper()


def _write_id_list(lines: list[str], ids, per_line: int = 16) -> None:
    ids = [str(int(i)) for i in ids]
    for start in range(0, len(ids), per_line):
        lines.append(", ".join(ids[start : start + per_line]))


def write_input(
    path: Path,
    mesh: Mesh,
    material: Material,
    constraints: list[Constraint],
    forces: dict[int, np.ndarray],
    title: str = "MechAgent static analysis",
) -> Path:
    """Write a linear static C3D10 deck. Units: mm, N, MPa, tonne, s."""
    lines = ["*HEADING", f"{title} (units: mm, N, MPa, tonne, s)", "*NODE"]
    for node, (x, y, z) in zip(mesh.node_ids, mesh.coords, strict=True):
        lines.append(f"{node}, {x:.9g}, {y:.9g}, {z:.9g}")

    lines.append("*ELEMENT, TYPE=C3D10, ELSET=EALL")
    for element, nodes in zip(mesh.element_ids, mesh.elements, strict=True):
        lines.append(f"{element}, " + ", ".join(str(n) for n in nodes))

    for name, nodes in mesh.node_sets.items():
        lines.append(f"*NSET, NSET={_set_name(name)}")
        _write_id_list(lines, nodes)

    constrained = np.unique(np.concatenate([mesh.node_sets[c.surface] for c in constraints]))
    lines.append(f"*NSET, NSET={CONSTRAINED_SET}")
    _write_id_list(lines, constrained)

    lines += [
        f"*MATERIAL, NAME={_set_name(material.name)}",
        "*ELASTIC",
        f"{material.youngs_modulus:.9g}, {material.poisson_ratio:.9g}",
        "*DENSITY",
        f"{material.density:.9g}",
        f"*SOLID SECTION, ELSET=EALL, MATERIAL={_set_name(material.name)}",
        "*STEP",
        "*STATIC",
        "*BOUNDARY",
    ]
    for constraint in constraints:
        for dof in constraint.dofs:
            lines.append(f"{_set_name(constraint.surface)}, {dof}, {dof}")

    lines.append("*CLOAD")
    for node, force in sorted(forces.items()):
        for dof, value in enumerate(force, start=1):
            if value != 0.0:
                lines.append(f"{node}, {dof}, {value:.12g}")

    lines += [
        f"*NODE PRINT, NSET={CONSTRAINED_SET}",
        "RF",
        "*NODE FILE",
        "U",
        "*EL FILE",
        "S",
        "*END STEP",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def run_calculix(
    job_dir: Path,
    job_name: str,
    executable: str = "ccx",
    timeout_s: float = 600,
    threads: int = 1,
) -> SolverRun:
    """Run ccx and decide from its outputs, not only its exit code, whether it succeeded."""
    ccx = shutil.which(executable)
    if ccx is None:
        raise SolverError(f"CalculiX executable '{executable}' not found on PATH")

    command = [ccx, "-i", job_name]
    log_file = job_dir / f"{job_name}.log"
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=job_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={**os.environ, "OMP_NUM_THREADS": str(threads)},
        )
        returncode, output = result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired as exc:
        returncode, output = None, f"{exc.stdout or ''}\nTIMEOUT after {timeout_s} s"
    duration = time.perf_counter() - start
    log_file.write_text(output)

    lines = output.splitlines()
    errors = [line.strip() for line in lines if "*ERROR" in line or "TIMEOUT" in line]
    warnings = [line.strip() for line in lines if "*WARNING" in line]
    frd = job_dir / f"{job_name}.frd"
    dat = job_dir / f"{job_name}.dat"

    if returncode != 0 and not errors:
        errors.append(f"ccx exited with code {returncode}")
    if "Job finished" not in output and not errors:
        errors.append("ccx did not report 'Job finished'")
    for out_file in (frd, dat):
        if not out_file.exists() or out_file.stat().st_size == 0:
            errors.append(f"missing or empty output file {out_file.name}")

    run = SolverRun(
        command=command,
        job_dir=job_dir,
        returncode=returncode,
        duration_s=duration,
        success=not errors,
        errors=errors,
        warnings=warnings,
        log_file=log_file,
    )
    if run.success:
        log.info("ccx finished %s in %.1f s", job_name, duration)
    else:
        log.error("ccx failed on %s: %s", job_name, "; ".join(errors))
    return run


def read_node_forces(dat_file: Path, set_name: str = CONSTRAINED_SET) -> dict[int, np.ndarray]:
    """Nodal forces (N) printed by *NODE PRINT with RF.

    CalculiX prints the three components of every node in the set, also on the directions
    that are not constrained. The caller keeps only the constrained ones.
    """
    lines = dat_file.read_text().splitlines()
    header = f"forces (fx,fy,fz) for set {set_name.upper()}"
    forces: dict[int, np.ndarray] = {}
    for i, line in enumerate(lines):
        if header not in line:
            continue
        for following in lines[i + 1 :]:
            parts = following.split()
            if not parts:
                if forces:
                    break
                continue
            if len(parts) != 4:
                break
            forces[int(parts[0])] = np.array([float(v) for v in parts[1:]])
    if not forces:
        raise SolverError(f"no nodal forces for set {set_name} in {dat_file}")
    return forces
