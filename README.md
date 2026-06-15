# MechAgent

MechAgent is an autonomous mechanical design and optimization agent. It takes engineering
requirements as input: mounting points, loads, boundary conditions, material constraints and
design targets. From those it generates a parametric part, simulates it, optimizes it and
writes an engineering report.

The idea behind the project: **the LLM orchestrates, deterministic tools compute.**
The language model (local, through Ollama) interprets requirements, chooses which analyses
to run and decides whether to iterate. Every stress, displacement, mass or fatigue life is
computed by CadQuery, Gmsh, CalculiX, NumPy/SciPy or Optuna, never by the LLM. A design is
only declared valid once every required verification has run and passed.

The first supported component is a mechanical mounting bracket, optimized with parametric
(not topology) optimization.

## Architecture

```
                 ┌──────────────────────────────────────────┐
  requirements → │  agent/  LangGraph workflow (Ollama LLM) │ → report
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
| `mechagent.core`           | Configuration, logging, unit system, environment checks      | Phase 0  |
| `mechagent.cad`            | Parametric bracket geometry with CadQuery                    | Phase 1  |
| `mechagent.simulation`     | Gmsh meshing, CalculiX static FEM, S-N fatigue, Miner damage | Phase 1-2|
| `mechagent.materials`      | Material database and candidate selection                    | Phase 2  |
| `mechagent.optimization`   | Parametric optimization with Optuna                          | Phase 2  |
| `mechagent.agent`          | LangGraph orchestration of the engineering workflow          | Phase 3  |
| `mechagent.reporting`      | Engineering report and 3D visualizations                     | Phase 4  |
| `mechagent.app`            | Streamlit user interface                                     | Phase 4  |

Other folders:

- `config/` holds the YAML configuration (`default.yaml`).
- `data/` holds input data such as material properties and S-N curves.
- `examples/` holds example requirement files.
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

The agent also needs a local [Ollama](https://ollama.com) server with the model set in
`config/default.yaml` (by default `qwen2.5:7b`):

```bash
brew install ollama
```

```bash
ollama serve
```

```bash
ollama pull qwen2.5:7b
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

## Tests

```bash
pytest
```

Tests marked `external` run the real tools. CalculiX solves a single-element tension
problem, which is checked against the analytical result u = FL/EA. Gmsh meshes a box, and
CadQuery builds one. When a tool is missing these tests are skipped with a visible reason
rather than passing. To run only the fast unit tests:

```bash
pytest -m "not external"
```

## Roadmap

- **Phase 0**: environment, project structure, configuration, logging, tests (done).
- **Phase 1**: parametric bracket CAD, Gmsh meshing, CalculiX static analysis, result
  parsing, mesh convergence and analytical validation cases.
- **Phase 2**: material database, S-N fatigue with mean stress correction and Miner damage,
  Optuna parametric optimization.
- **Phase 3**: LangGraph agent: requirement interpretation, missing-data detection, analysis
  selection, iteration decisions and a decision trace.
- **Phase 4**: Streamlit interface, PyVista visualizations and the engineering report.
