"""Smoke tests on the real external tools.

They are skipped (with a visible reason) when a tool is missing, they never pretend it worked.
"""

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.external

# Unit cube (1 mm) in tension along x, one C3D8 element. Units: mm, N, MPa.
SINGLE_ELEMENT_INP = """\
*NODE
1, 0, 0, 0
2, 1, 0, 0
3, 1, 1, 0
4, 0, 1, 0
5, 0, 0, 1
6, 1, 0, 1
7, 1, 1, 1
8, 0, 1, 1
*ELEMENT, TYPE=C3D8, ELSET=EALL
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=LOADED
2, 3, 6, 7
*MATERIAL, NAME=STEEL
*ELASTIC
210000., 0.3
*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL
*BOUNDARY
1, 1, 3
4, 1, 1
4, 3, 3
5, 1, 2
8, 1, 1
*STEP
*STATIC
*CLOAD
LOADED, 1, 25.
*NODE PRINT, NSET=LOADED
U
*NODE FILE
U
*EL FILE
S
*END STEP
"""


def read_ux_from_dat(dat_text):
    """Return the x displacements printed in the .dat file."""
    values = []
    in_table = False
    for line in dat_text.splitlines():
        if "displacements" in line:
            in_table = True
            continue
        parts = line.split()
        if in_table and len(parts) == 4:
            values.append(float(parts[1]))
    return values


def test_calculix_solves_single_element_tension(tmp_path):
    ccx = shutil.which("ccx")
    if ccx is None:
        pytest.skip("CalculiX (ccx) is not on PATH")

    (tmp_path / "cube.inp").write_text(SINGLE_ELEMENT_INP)
    result = subprocess.run(
        [ccx, "-i", "cube"], cwd=tmp_path, capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Job finished" in result.stdout
    assert (tmp_path / "cube.frd").exists()

    # Uniform stress state, so a single linear element must match u = F L / (E A) exactly.
    ux = read_ux_from_dat((tmp_path / "cube.dat").read_text())
    expected = 100.0 * 1.0 / (210000.0 * 1.0)
    assert len(ux) == 4
    assert ux == pytest.approx([expected] * 4, rel=1e-6)


def test_gmsh_meshes_a_box():
    gmsh = pytest.importorskip("gmsh")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 10, 5, 2)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", 2.0)
        gmsh.model.mesh.generate(3)
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        element_types, _, _ = gmsh.model.mesh.getElements(dim=3)
    finally:
        gmsh.finalize()

    assert len(node_tags) > 8
    assert 4 in element_types  # 4-node tetrahedra


def test_cadquery_builds_a_box():
    cq = pytest.importorskip("cadquery")

    box = cq.Workplane("XY").box(10, 20, 5)

    assert box.val().Volume() == pytest.approx(1000.0)


def test_lmstudio_model_answers():
    from mechagent.core.config import load_settings
    from mechagent.core.environment import check_llm_server

    llm_settings = load_settings().llm
    status = check_llm_server(llm_settings.base_url, llm_settings.model)
    if not status.available:
        pytest.skip(f"LM Studio is not ready: {status.detail}")

    from langchain_openai import ChatOpenAI

    # LM Studio ignores the API key, but the OpenAI client requires one.
    llm = ChatOpenAI(
        base_url=llm_settings.base_url,
        model=llm_settings.model,
        temperature=llm_settings.temperature,
        api_key="lm-studio",
        timeout=120,
    )
    answer = llm.invoke("Reply with the single word: ready")

    assert "ready" in answer.content.lower()
