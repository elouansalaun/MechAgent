# Project: MechAgent

You are acting as a senior Python software engineer with expertise in mechanical engineering, numerical simulation, finite element analysis and AI agent development.

I want to develop **MechAgent**, an autonomous mechanical design and optimization agent.

## Objective

Build a Python application capable of designing and optimizing mechanical components from engineering requirements.

The system receives:
- 3D coordinates and orientations of mounting points.
- Mechanical boundary conditions.
- Static or cyclic loads.
- Material constraints.
- Design requirements such as maximum weight, minimum safety factor, maximum displacement or minimum cost.

It must generate a parametric CAD model, select appropriate materials, determine which simulations are required, execute numerical simulations, optimize the design and generate an engineering report.

## Fundamental architecture

The LLM must orchestrate the engineering workflow using LangGraph.

All physical calculations must be performed by deterministic engineering tools, not by the LLM.

Use:
- Python
- LangGraph
- Ollama for local LLM inference
- Pydantic for data validation
- CadQuery for parametric CAD
- Gmsh for meshing
- CalculiX for FEM
- NumPy and SciPy for numerical calculations
- Optuna for optimization
- PyVista for 3D visualization
- Streamlit for the interface
- pytest for testing

## Development methodology

Develop the project incrementally.

Do not attempt to implement the entire application at once.

For each phase:
1. Explain the implementation strategy.
2. Implement the required modules.
3. Write unit tests.
4. Execute the tests.
5. Fix errors.
6. Document the implementation.
7. Summarize what is complete and what remains.

Never create fake simulation results or assume that a solver succeeded without checking its output.

Use modular architecture and clear interfaces between CAD, FEM, materials, optimization and agent components.

## Engineering requirements

The first supported component must be a mechanical mounting bracket.

The geometry must be parametric.

The first optimization method must be parametric optimization, not topology optimization.

The initial analyses must include:
- Static linear elastic FEM.
- Fatigue analysis based on S-N curves.
- Mean stress correction.
- Miner cumulative damage.

All simulation assumptions, boundary conditions, material properties and units must be explicitly recorded.

Include mesh convergence checks and basic validation against analytical reference problems.

## Agent requirements

The agent must:
- Interpret the engineering requirements.
- Identify missing or inconsistent information.
- Select appropriate simulation methods.
- Select candidate materials.
- Generate and evaluate candidate geometries.
- Invoke simulation and optimization tools.
- Interpret simulation outputs.
- Decide whether another iteration is necessary.
- Generate a final engineering report.

The agent must not declare a design valid if any required verification is missing or failed.

## Repository organization

Use the following architecture:

src/mechagent/
    core/
    cad/
    simulation/
    materials/
    optimization/
    agent/
    reporting/
    app/

tests/
data/
examples/

## Development guidelines
Develop and code this project as a human could do, write simple understandable code, use "we" pronoun, dont over comment the code, do commits when adding new concepts

## First task

Start with Phase 0 only:

- Inspect the development environment.
- Check the availability of required external software.
- Initialize the Python project.
- Create the repository structure.
- Configure dependencies, logging and pytest.
- Create a minimal configuration file.
- Write a README explaining the project architecture.

Do not implement the CAD generator, FEM solver or AI agent yet.

At the end, provide:
- Files created.
- Dependencies installed.
- External tools successfully detected.
- Tests executed and their results.
- Any remaining environment limitations.

Wait for my validation before proceeding to Phase 1.