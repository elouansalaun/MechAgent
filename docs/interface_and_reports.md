# Interface and reports (Phase 4)

## Streamlit app

```bash
mechagent-app
```

That is the same as `streamlit run src/mechagent/app/streamlit_app.py`. The sidebar shows
whether CalculiX and the LM Studio model are available, and sets the search budget: trials
per material and iteration, iterations, materials.

**New design**

1. Write the specification as text (or start from an example), or fill in the form: bolt
   holes, load hole, force, load type, cycles, load ratio, limits, objective, materials,
   width envelope and excitation frequency.
2. **Review** runs the LLM on the text, then the deterministic review and analysis plan, in
   about 15 s. It shows:
   - the blocking questions
   - the assumptions
   - the bracket frame in global coordinates, the parameters fixed by the interface and the
     load in the bracket frame
   - the analysis plan
   - the request as the LLM understood it

   After a text review, the form is filled with what the LLM read, so mistakes can be
   corrected there.
3. **Start the design** is only offered when nothing blocks. It starts
   `python -m mechagent.agent.cli` as a separate process, so the run survives a browser
   refresh or closing the app. A text run reads the text again, so the trace and the text
   checks stay complete.

**Runs**

- **A running design:** the progress panel refreshes every 10 s with the latest trace
  entries, read from `progress.json`.
- **A finished design:**
  - status, material, mass, displacement and safety factors
  - the figures, every check, the parameters
  - the optimisation table and the decision trace
  - downloads: `report.pdf`, `report.md`, `design.step`

## Files of a run

Each agent run writes to `runs/<timestamp>_agent/`:

| File                       | Content                                                    |
| -------------------------- | ---------------------------------------------------------- |
| `request.json` / `.txt`    | the input (app runs)                                       |
| `progress.json`            | status, iteration and trace, rewritten after every node    |
| `summary.json`             | `RunSummary`: request, review, plan, validation, searches, verified design, trace |
| `report.md`, `report.pdf`  | the engineering report                                     |
| `design.step`              | the verified bracket in the global frame of the request    |
| `figures/*.png`            | mesh, von Mises stress and displacement of the final design |
| `validation/`, `iteration_*/` | every analysis with its deck, log and JSON records      |
| `agent.log`, `console.log` | logs (CLI runs)                                            |

`design.step` and the figures are only written for a verified design. A deliverable that
fails is written to the trace; it never changes the engineering status.

## Report

`reporting/markdown.py` builds the report from the state only: tool results, rule
decisions and LLM choices, each with its source. `reporting/pdf.py` converts that Markdown
to PDF with ReportLab (headings, tables, lists, quotes, bold, images), so the two versions
cannot differ.

Sections:

1. request
2. review
3. bracket placement
4. analysis plan
5. solver validation
6. material selection
7. optimisation
8. final design (parameters, global hole positions, results, checks, figures, STEP)
9. assumptions
10. decision trace
11. limitations

## Figures

`reporting/figures.py` renders off screen with PyVista. The C3D10 elements map directly
onto VTK quadratic tetrahedra, so the stresses are drawn on the real quadratic mesh:

- the mesh
- the von Mises stress on the deformed shape. Its colour scale is capped at the design
  stress, because the singular values at the rigid bolt supports would otherwise wash out
  the rest of the part.
- the displacement on the deformed shape

Deformations are magnified so that the largest one is drawn at 5 % of the part size, and
the factor is written on the figure.

## STEP export

`cad/export.py` builds the bracket and moves it with a rigid OpenCascade transform (global
= origin + R · local) taken from the interface mapping. A test checks that the wall bracket
lands on the wall, keeps its volume and survives a STEP round trip. Mirrors are refused.

## Long runs on macOS

A sleeping Mac pauses the monotonic clock that subprocess timeouts rely on: one solve
seemed to run for 66 minutes against a 600 s timeout. The CLI and the examples call
`core.system.keep_awake`, which runs `caffeinate` for the lifetime of the process.
