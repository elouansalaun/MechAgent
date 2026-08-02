# MechAgent

MechAgent is an autonomous mechanical design and optimization agent. It takes engineering
requirements as input: mounting points, loads, boundary conditions, material constraints and
design targets. From those it generates a parametric part, simulates it, optimizes it and
writes an engineering report.

The idea behind the project: **the LLM orchestrates, deterministic tools compute.**
The language model (local, served by LM Studio) interprets requirements, chooses which analyses
to run and decides whether to iterate. Every stress, displacement, mass or fatigue life is
computed by CadQuery, Gmsh, CalculiX, NumPy/SciPy or Optuna, never by the LLM. A design is
only declared valid once every required verification has run and passed.

The first supported component is a mechanical mounting bracket, optimized with parametric
(not topology) optimization.

## Architecture

```
                 ┌──────────────────────────────────────────┐
  requirements → │ agent/ LangGraph workflow (LM Studio LLM)│ → report
                 └──────┬──────────┬──────────┬─────────────┘
                        │ calls deterministic tools
        ┌───────────────┼──────────┼──────────┼──────────────────┐
        ▼               ▼          ▼          ▼                  ▼
      cad/        simulation/  materials/  optimization/     reporting/
   CadQuery      Gmsh mesh,    material    Optuna parametric  report, PyVista
   bracket       CalculiX FEM, database,   search             figures
                 fatigue       selection
                        ▲
                 core/  config, logging, units, environment checks, shared data models
                 app/   Streamlit interface on top of the agent
```

| Package                    | Role                                                         | Status   |
| -------------------------- | ------------------------------------------------------------ | -------- |
| `mechagent.core`           | Configuration, logging, units, geometry interface, checks    | Done     |
| `mechagent.cad`            | Parametric bracket and reference geometries with CadQuery    | Done     |
| `mechagent.simulation`     | Gmsh, CalculiX static FEM, rainflow, S-N fatigue, validation | Done     |
| `mechagent.materials`      | Material database, S-N curves, ranking, cost model           | Done     |
| `mechagent.optimization`   | Design evaluation and parametric optimization with Optuna    | Done     |
| `mechagent.agent`          | LangGraph orchestration of the engineering workflow          | Phase 3  |
| `mechagent.reporting`      | Engineering report and 3D visualizations                     | Phase 4  |
| `mechagent.app`            | Streamlit user interface                                     | Phase 4  |

Other folders:

- `config/` holds the YAML configuration (`default.yaml`).
- `data/` holds the material database (`materials.yaml`, typical handbook values).
- `docs/` holds the implementation notes: [simulation](docs/simulation.md),
  [materials and fatigue](docs/fatigue.md), [optimization](docs/optimization.md).
- `examples/` holds runnable examples.
- `tests/` holds the pytest suite, including smoke tests on the real solvers.
- `runs/` and `logs/` are created at runtime and ignored by git.

## Units

Everything runs in one consistent unit system, the one CalculiX users usually pick:

| Quantity | Unit                   |
| -------- | ---------------------- |
| Length   | mm                     |
| Force    | N                      |
| Stress   | MPa                    |
| Mass     | tonne (ρ in tonne/mm³) |
| Time     | s                      |

The configuration rejects any other unit system, so mixed units cannot slip in silently.

## Installation

CalculiX and the OpenCascade kernel behind CadQuery are installed from conda-forge. The
Python dependencies are then installed from `pyproject.toml` in editable mode:

```bash
conda env create -f environment.yml
```

```bash
conda activate mechagent
```

The agent also needs a local LLM. We use [LM Studio](https://lmstudio.ai) and its
OpenAI compatible server, which LangChain talks to through `langchain-openai`. Load the
model set in `config/default.yaml` (by default `qwen/qwen3.5-9b`) and start the server on
port 1234, either from the LM Studio Developer tab or from its CLI:

```bash
lms load qwen/qwen3.5-9b
```

```bash
lms server start
```

## Checking the environment

```bash
mechagent-check-env
```

The command lists every Python package, external program and service MechAgent needs. It
exits with code 1 if anything is missing. A program only counts as available if it actually
runs and reports a version.

## Configuration

Settings come from `config/default.yaml`. To use another file, set `MECHAGENT_CONFIG`:

```bash
MECHAGENT_CONFIG=my_config.yaml mechagent-check-env
```

The file is validated with Pydantic, so unknown keys (typos) and unsupported units are errors.

## Static analysis of the bracket

```bash
python examples/bracket_static.py
```

The example validates the solver chain on the analytical reference cases, then runs a mesh
convergence study on the default aluminium bracket under a 2000 N pin load and prints every
result, check and assumption. Each run gets its own folder under `runs/` with the BREP,
CalculiX deck and outputs, `problem.json`, `result.json` and the nodal fields (`fields.npz`).

From Python:

```python
from pathlib import Path
from mechagent.cad.bracket import BracketParameters, build_bracket
from mechagent.materials.material import Material
from mechagent.simulation.bracket import bracket_static_problem
from mechagent.simulation.static import run_static_analysis

material = Material(name="Al6061-T6", youngs_modulus=68900, poisson_ratio=0.33,
                    density=2.70e-9, yield_strength=276, ultimate_strength=310)
params = BracketParameters(base_thickness=10, rib_thickness=3)
geometry = build_bracket(params, Path("runs/my_bracket"))
problem = bracket_static_problem(geometry, params, material, force=(0, 0, -2000))
result = run_static_analysis(problem, Path("runs/my_bracket/static"))
print(result.design_von_mises, result.max_displacement, result.verified)
```

How the chain works, which checks it runs and what its limits are is described in
[docs/simulation.md](docs/simulation.md).

## Optimizing the bracket

```bash
python examples/bracket_optimization.py 40
```

This optimizes a bracket for minimum mass under a 2000 N cyclic pin load: 1e6 cycles, max
displacement 0.5 mm, static safety factor 2. The script:

1. ranks the six materials of `data/materials.yaml`
2. optimizes the geometry of the three best ones with Optuna
3. picks the lightest valid design
4. verifies it again on a finer mesh

It prints the decision trace, the final design, every check and every assumption. It takes
about 15 minutes with 40 trials per material and 30 minutes with 80. The
fatigue method (S-N, Marin factors, mean stress correction, Miner, rainflow) is described
in [docs/fatigue.md](docs/fatigue.md), and the optimization in
[docs/optimization.md](docs/optimization.md).

## Tests

```bash
pytest
```

Tests marked `external` run the real tools: CalculiX analyses of the bracket, the
analytical validation cases (static and fatigue), a convergence study, design evaluations,
a short real optimization and an LM Studio prompt. When a tool is
missing these tests are skipped with a visible reason rather than passing. To run only the
fast unit tests:

```bash
pytest -m "not external"
```

## Roadmap

- **Phase 0**: environment, project structure, configuration, logging, tests (done).
- **Phase 1**: parametric bracket CAD, Gmsh meshing, CalculiX static analysis, result
  parsing, mesh convergence and analytical validation cases (done).
- **Phase 2**: material database, S-N fatigue with mean stress correction and Miner damage,
  rainflow counting, cost model, Optuna parametric optimization (done).
- **Phase 3**: LangGraph agent: requirement interpretation, missing-data detection, analysis
  selection, iteration decisions and a decision trace.
- **Phase 4**: Streamlit interface, PyVista visualizations and the engineering report.
