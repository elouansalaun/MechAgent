# MechAgent

**An AI engineering agent that turns a plain-text specification into a verified mechanical
part, with its simulation, its optimization, a PDF engineering report and a STEP CAD file.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C)
![CalculiX](https://img.shields.io/badge/CalculiX-FEM-orange)
![CadQuery](https://img.shields.io/badge/CadQuery-CAD-blue)
![Optuna](https://img.shields.io/badge/Optuna-optimization-6A5ACD)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-186-success)

> **The LLM orchestrates, deterministic tools compute.** The language model reads the
> request and picks the next step among the actions engineering rules allow. Every stress,
> displacement, mass and fatigue life comes from a real solver, never from the LLM.

![MechAgent in 50 seconds](docs/media/mechagent_explainer_en.gif)


## Overview

| You give it                                                   | It gives you back                                                    |
| ------------------------------------------------------------- | -------------------------------------------------------------------- |
| A short text: mounting points, load, deflection limit, safety factor, material family | A bracket designed, meshed, simulated with finite elements, optimized for minimum mass and checked |
|                                                               | A [PDF engineering report](docs/media/example_report.pdf) with every check, assumption and decision |
|                                                               | A [STEP model](docs/media/example_design.step) placed in your global frame, ready for CAD |

Result of the run shown below (titanium shelf bracket, 2000 N static load), about 20 minutes
on a laptop:

| Mass      | Max displacement         | Yield safety factor     | Checks passed | Status       |
| --------- | ------------------------ | ----------------------- | ------------- | ------------ |
| 960.5 g   | 1.86 mm (limit 2 mm)     | 7.34 (required 2)       | 10 / 10       | **VERIFIED** |

A second example, an aluminium wall bracket under a cyclic load of one million cycles,
ends verified at 239 g of Al7075-T6, 19 % lighter than the first iteration after the LLM
chose to refine the search.

## Agentic workflow

The agent is a [LangGraph](https://github.com/langchain-ai/langgraph) state graph. The LLM
appears at only two points: reading the request and choosing between actions the rules
already allow. Everything else is engineering rules and deterministic tools, and every node
writes its decision to the trace with its source.

```mermaid
flowchart TD
    IN(["Plain-text specification"]) --> parse

    parse["parse<br/>LLM extracts loads, points and limits,<br/>every value is checked against the text"]:::llm
    review["review<br/>places the bracket on the mounting points,<br/>lists missing or inconsistent data"]:::rule
    plan["plan<br/>static, fatigue, buckling, modal:<br/>required or not, and why"]:::rule
    validate["validate_solver<br/>the FEM chain must reproduce<br/>6 analytical reference cases"]:::tool
    select["select_materials<br/>Ashby index ranking, shortlist"]:::tool
    search["search<br/>Optuna optimization,<br/>one full FEM evaluation per trial"]:::tool
    verify["verify<br/>best designs re-checked on a finer mesh,<br/>mesh convergence study"]:::tool
    decide{"decide<br/>rules list the allowed actions,<br/>the LLM picks one"}:::llm
    report["report<br/>PDF, Markdown, STEP, summary"]:::tool
    OUT(["Status + deliverables"])

    parse --> review --> plan --> validate --> select --> search --> verify --> decide
    decide -- "refine / widen_search" --> search
    decide -- "accept / stop" --> report
    parse -. "text unreadable: error" .-> report
    review -. "blocking question: needs_input" .-> report
    plan -. "analysis not implemented: unverifiable" .-> report
    validate -. "a case fails: solver_not_validated" .-> report
    report --> OUT

    classDef llm fill:#e9d5ff,stroke:#7c3aed,color:#1e1b4b
    classDef tool fill:#dbeafe,stroke:#2563eb,color:#0c1a3a
    classDef rule fill:#fef3c7,stroke:#d97706,color:#3b2600
```

🟪 LLM &nbsp; 🟦 deterministic tool &nbsp; 🟨 engineering rule &nbsp; · &nbsp; dotted arrows
stop the run early with an explicit status instead of producing an unchecked design.

### How a decision is made

The LLM never chooses freely. After each verification, the rules compute which actions are
allowed; the LLM only arbitrates when there is more than one, and an answer outside the
list falls back to the rule default.

```mermaid
flowchart LR
    S["verification results<br/>+ remaining budget"] --> R["engineering rules<br/>list the allowed actions"]:::rule
    R --> N{"how many?"}
    N -- "one" --> A["apply it<br/>traced as [rule]"]:::rule
    N -- "several" --> L["LLM chooses one<br/>and explains why"]:::llm
    L --> C{"in the<br/>allowed list?"}
    C -- "yes" --> B["apply it<br/>traced as [llm]"]:::llm
    C -- "no, or LLM error" --> D["rule default<br/>traced as [rule]"]:::rule

    classDef llm fill:#e9d5ff,stroke:#7c3aed,color:#1e1b4b
    classDef rule fill:#fef3c7,stroke:#d97706,color:#3b2600
```

| Situation after verification                                              | Allowed actions        |
| ------------------------------------------------------------------------- | ---------------------- |
| Valid design, a requirement within 10 % of its limit, or no iteration left | `accept`               |
| Valid design with margin left on every requirement                        | `refine`, `accept`     |
| Valid designs found, none passed the final verification                   | `refine`, `stop`       |
| No valid design found                                                     | `widen_search`, `stop` |
| No valid design and no iteration left                                     | `stop`                 |

In the titanium run, the LLM chose `refine` after iteration 1 (14 % margin on displacement,
1004.7 g). After iteration 2 (7 % margin, 960.5 g) the rules allowed only `accept`.

### What one optimization trial runs

```mermaid
flowchart LR
    O["Optuna TPE sampler<br/>proposes thicknesses, width,<br/>ribs, fillets"] --> C["CadQuery<br/>parametric bracket"]
    C --> M["Gmsh<br/>quadratic tetra mesh"]
    M --> F["CalculiX<br/>linear static FEM"]
    F --> P["post-processing<br/>yield, displacement,<br/>fatigue S-N + Miner, buckling"]
    P --> K["checks<br/>solver, mesh, equilibrium,<br/>linearity, requirements"]
    K -- "mass, cost, valid or not" --> O

    classDef tool fill:#dbeafe,stroke:#2563eb,color:#0c1a3a
    class O,C,M,F,P,K tool
```

## How it works

### 1. Describe the part in plain text

The LLM extracts the request, then every number and vector it read is checked against the
original text. The agent places the bracket on the mounting points and decides which
analyses are needed (static, fatigue, buckling, modal), each with its reason.

![Specification, interpretation and analysis plan](docs/media/Streamlit_dashboard_prompt_example.png)

<details>
<summary>Empty home screen, with the solver and LLM status and the search budget</summary>

![Home screen](docs/media/Streamlit_dashboard_home.png)

</details>

### 2. The agent designs, simulates and iterates

The run continues in the background. The agent validates the solver against analytical
cases, ranks materials, optimizes the geometry with Optuna and decides whether to refine,
widen, accept or stop. Every decision is logged with its source: `[rule]`, `[llm]` or
`[tool]`.

![Decision trace](docs/media/Streamlit_dashboard_result_decision_trace.png)

### 3. A verified design, with its evidence

A design is only declared **verified** once every required check has run and passed,
including a mesh convergence study on a finer mesh.

![Verified result](docs/media/Streamlit_dashboard_result_verified1.png)

![Checks, final geometry and optimization history](docs/media/Streamlit_dashboard_result_verified2.png)

| Von Mises stress (MPa)                            | Displacement (mm)                                     |
| ------------------------------------------------- | ----------------------------------------------------- |
| ![Von Mises stress](docs/media/result_von_mises.png) | ![Displacement](docs/media/result_displacement.png) |

### 4. Deliverables

<a href="docs/media/example_report.pdf"><img src="docs/media/example_report_page1.png" alt="First page of the PDF report" width="360" align="right"></a>

Each run produces:

- **[report.pdf](docs/media/example_report.pdf)**: the request, its review, the bracket
  placement, the analysis plan, the solver validation, the material ranking, every
  optimization iteration, every check with its limit, the figures, the assumptions and the
  full decision trace
- **[design.step](docs/media/example_design.step)**: the final geometry in the global
  frame of the request, ready to open in any CAD tool
- `report.md` and `summary.json`, for reading or for other tools

Click the page on the right to open the full example report.

<br clear="right">

## Area of work

| Area                              | What is in the project                                                                                       | Tools                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------- |
| **Finite element analysis**       | Linear static FEM with quadratic tetrahedra, mesh quality and equilibrium checks, mesh convergence, validation against 6 analytical cases | CalculiX, Gmsh                      |
| **Fatigue and strength**          | S-N curves, Marin factors, Goodman / Gerber / Soderberg mean stress correction, rainflow counting, Miner damage, buckling screening | NumPy, SciPy                        |
| **Parametric CAD**                | Parametric L bracket with ribs and fillets, placement on arbitrary mounting points by frame transforms, STEP export | CadQuery (OpenCascade)              |
| **Optimization and materials**    | Mass minimization under constraints, warm-started refinement, material ranking with Ashby indices, cost model | Optuna                              |
| **AI agents and LLMs**            | Agent as a state graph, local LLM, decisions restricted to rule-allowed actions, every extracted value checked against the source text, decision trace | LangGraph, LangChain, LM Studio (Qwen) |
| **Software engineering**          | Typed data models, validated configuration, single unit system enforced, 186 tests (unit tests with a fake LLM and smoke tests on the real solvers), linting, packaging | Pydantic, pytest, ruff, conda       |
| **Visualization and reporting**   | 3D result figures, PDF report generation, web interface with background runs                                 | PyVista, ReportLab, Streamlit       |

## Architecture

```
                 ┌──────────────────────────────────────────┐
  requirements → │ agent/ LangGraph workflow (local LLM)    │ → report, STEP
                 └──────┬──────────┬──────────┬─────────────┘
                        │ calls deterministic tools
        ┌───────────────┼──────────┼──────────┼──────────────────┐
        ▼               ▼          ▼          ▼                  ▼
      cad/        simulation/  materials/  optimization/     reporting/
   CadQuery      Gmsh mesh,    material    Optuna parametric  PDF report,
   bracket       CalculiX FEM, database,   search             PyVista figures
                 fatigue       selection
                        ▲
                 core/  config, logging, units, environment checks, shared data models
                 app/   Streamlit interface on top of the agent
```


## Quick start

```bash
conda env create -f environment.yml
conda activate mechagent
mechagent-check-env   # lists every package, program and service needed
mechagent-app         # opens the Streamlit interface
```

The agent needs a local LLM served by [LM Studio](https://lmstudio.ai) on port 1234
(default model `qwen/qwen3.5-9b`):

```bash
lms load qwen/qwen3.5-9b
lms server start
```

In the app, pick the `wall_bracket` example, click **Review**, then **Start the design**.


## Limitations and next steps

The first supported component is a two-bolt L bracket, optimized parametrically (not by
topology optimization). Modal analysis is not implemented yet, so a request that needs it
is marked unverifiable rather than passed. Next steps: a contact-based bolted joint model,
buckling and modal FEM, more component types, and a Pareto front of mass against cost.
