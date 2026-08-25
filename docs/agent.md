# The engineering agent (Phase 3)

The agent turns a specification into a verified bracket design, or explains why it cannot.
It is a LangGraph state machine. The LLM (Qwen 3.5 9B on LM Studio) does two things: it
reads free text specifications, and it chooses between the actions the rules allow after
each optimisation round. Every number comes from deterministic tools. Every decision is
written to a trace with its source: `rule`, `llm`, `tool` or `user`.

```
parse ─► review ─► plan ─► validate_solver ─► select_materials ─► search ─► verify ─► decide
                                                                     ▲                  │
                                                                     └─ refine / widen ─┤
any blocking problem ─────────────────────────────────────────────► report ◄─ accept / stop
```

| Node               | What it does                                                    | Source |
| ------------------ | --------------------------------------------------------------- | ------ |
| `parse`            | text → `EngineeringRequest` (skipped for a structured request)  | llm    |
| `review`           | map the interface, find missing / inconsistent data, defaults   | rule   |
| `plan`             | required analyses and why                                       | rule   |
| `validate_solver`  | run the analytical reference cases                              | tool   |
| `select_materials` | Ashby ranking, shortlist                                        | tool   |
| `search`           | one Optuna search per shortlisted material                      | tool   |
| `verify`           | convergence study and finer mesh evaluation, in objective order | tool   |
| `decide`           | allowed actions from the rules, choice by the LLM               | rule + llm |
| `report`           | final status from the facts, Markdown report                    | rule   |

Code: `agent/graph.py` (graph), `agent/review.py`, `agent/interface.py`,
`agent/planning.py`, `agent/llm.py`, `agent/tools.py`, `reporting/markdown.py`.

## Request

`core.request.EngineeringRequest` is stated in the user's global frame. Every field is
optional, so the review decides what is missing:

- `mounting_points`: two bolt holes. Position is where the bolt axis meets the support
  surface, axis is the surface normal pointing into the part, plus the hole diameter.
- `load_point`: the pin hole. Position is where its axis meets the face of the part on the
  support side, plus the axis and the diameter.
- `force` (N, global, maximum load), `load_type`, `cycles`, `load_ratio`
- `min_safety_factor`, `max_displacement`, `max_mass`, `objective`,
  `candidate_materials`, `manufacturing`, `excitation_frequency`, `max_width`

## Reading text with the LLM

`LMStudioLLM.extract_request` sends the text with a strict JSON schema. Every property is
required but may be null; with optional properties the grammar accepts `{}` and the model
returns exactly that. Reasoning is switched off: in reasoning mode the model used 12 000
tokens in 9 minutes without answering. Without reasoning an extraction takes about 14 s.

The output is never trusted as it is. In `agent/review.py`:

- every scalar (cycles, load ratio, safety factor, limits, diameters, ...) must appear as a
  number in the text, with thousand separators, decimal commas, `kN` and `million`
  understood
- every position, axis and force direction must appear as an explicit `(x, y, z)` triple
  (directions compared up to their length)

Anything else becomes a blocking question showing what the model understood. This is not
theoretical. On the specification of the project description, without coordinates, the model:

- invented a load point at (0, 0, 0)
- read "vertical" as the y axis
- invented a load ratio of 0

All three were stopped. Even when told that z points up, it read "vertical" as y, which is
why directions written in words are always sent back to the user.

## Interface mapping

`agent/interface.py` finds the bracket frame from the two bolt holes and the load hole:

- z: common bolt axis (support normal)
- x: from the foot of the load hole to the middle of the bolts, in the support plane
- y: z × x

It checks that the interface is an L bracket:

- parallel bolt axes on one surface
- load hole axis parallel to the support and pointing towards the bolts
- load hole above the support and on the mid plane of the bolts
- one bolt diameter for both holes

From there it fixes the hole positions, the base length and the flange height. A self check
places the local holes back in the global frame and compares them with the request.

The design space keeps the Phase 2 ranges. The width starts at `spacing + 2 × bolt diameter`
and is capped by `max_width` when given.

## Review

Blocking: anything missing or inconsistent in the interface, an unknown load type, a cyclic
load without a cycle count, unknown materials, a static load with a cycle count, a width
limit below what the bolts need, unverified values.

Defaults, written down as assumptions:

| Field               | Default                                  |
| ------------------- | ---------------------------------------- |
| load ratio          | R = 0                                    |
| safety factor       | 1.5 on yield                             |
| objective           | minimum mass                             |
| candidate materials | every material of the database           |
| fatigue settings    | machined, 99 % reliability, Goodman      |
| manufacturing       | CNC machining (only process modelled)    |

## Analysis plan

| Situation                             | Analysis                                 | Available |
| ------------------------------------- | ---------------------------------------- | --------- |
| any load                              | linear static FEM + mesh convergence     | yes       |
| cyclic load                           | S-N fatigue, Goodman, Miner              | yes       |
| flange compressed at some point       | Euler column buckling screening, factor ≥ 3 | yes    |
| vibration excitation                  | modal analysis                           | no        |

A required analysis that is not available makes the request `unverifiable`, and the agent
stops before designing anything.

## Iterations

After each round of searches and verification, `iteration_options` lists what the rules
allow, with the rule's choice first:

| Situation                                              | Allowed actions        |
| ------------------------------------------------------ | ---------------------- |
| verified design, a requirement within 10 % of its limit | accept                 |
| verified design, every requirement more than 10 % away | refine, accept         |
| valid designs, none passed the final verification      | refine, stop           |
| no valid design                                        | widen_search, stop     |
| iteration budget spent                                 | accept, or stop        |

With one option the rule decides. With two, the LLM gets the situation in plain words
(values against limits, per material results) and must answer with one allowed action and a
reason. An action outside the list, or a failed call, falls back to the rule's choice. Both
cases are traced.

- **refine**: drop materials more than 15 % behind the leader. Narrow each remaining
  material's space to ±30 % around its best design, and start its next search from that
  design.
- **widen_search**: upper bounds ×1.5 and 1.5 times more trials.

## Final status

`final_status` recomputes the status from the state, whatever path the graph took:

| Status                 | When                                                     |
| ---------------------- | -------------------------------------------------------- |
| `needs_input`          | blocking questions                                       |
| `unverifiable`         | a required analysis is not available                     |
| `solver_not_validated` | an analytical validation case failed                     |
| `verified`             | a design passed every check, the convergence study and the solver validation |
| `not_verified`         | otherwise                                                |
| `error`                | no request could be read                                 |

## Report

`reporting/markdown.py` writes `report.md` in the run folder:

- the request and the review
- the bracket placement and the analysis plan
- the solver validation, the material ranking and the iterations
- the final design (parameters, global hole positions, results, every check)
- the assumptions, the full decision trace and the limitations

The LLM does not write any of it.

## Example

```bash
python examples/run_agent.py examples/requests/vague_request.txt
```

```bash
python examples/run_agent.py examples/requests/wall_bracket.txt
```

**Vague request** (the French specification of the project description: force, cycles,
displacement and safety factor, no geometry). The agent stops after 15 s with four
questions:

- the two bolt holes are missing
- the load point was read as (0, 0, 0), a value the text never gives
- "verticale" was read as the y axis for the load hole axis and for the force; there is no
  such vector in the text

The invented load ratio of 0 is also caught (since the scalar check covers every value).

**Wall bracket** (`examples/requests/wall_bracket.txt`, 40 trials per material and per
iteration). Status **verified** after 69 minutes of computation:

| Step              | Outcome                                                                  |
| ----------------- | ------------------------------------------------------------------------ |
| parse + review    | every number and vector found in the text, one documented assumption     |
| plan              | static and fatigue required, flange never compressed, no vibration       |
| solver validation | 6 / 6 reference cases passed                                             |
| iteration 1       | Al7075-T6 294.2 g (displacement margin 20 %), Al6061-T6 301.7 g, Ti-6Al-4V 323.6 g |
| decision 1        | the LLM chose **refine** (rules allowed refine or accept)                |
| iteration 2       | Al7075-T6 **239.1 g** (margin 4 %), Al6061-T6 267.1 g, Ti-6Al-4V 319.1 g |
| decision 2        | accept, the only action left once a requirement is within 10 % of its limit |

Refining around the best designs and warm starting from them gave a design 19 % lighter.
The final Al7075-T6 bracket:

- 68.5 mm wide, 8.9 mm base, 7.5 mm flange, 4.5 mm ribs
- displacement 0.479 mm (limit 0.5), yield safety factor 3.56 (limit 2)
- fatigue damage 0.00044 over 1e6 cycles, fatigue safety factor 2.21
- indicative cost 31 EUR

The run was first stopped twice by real problems, both fixed:

1. A rib toe ending 0.1 mm from the end of the base made OpenCascade fail on the toe
   fillet. The exception was not caught and stopped the whole run.
2. The Mac went to sleep. That pauses the clock behind the solver timeout, so a solve
   seemed to run for 66 minutes. The examples now call `core.system.keep_awake`.

## Tests

- `tests/test_agent_graph.py` runs the whole graph with fake tools and a fake LLM: refine
  then accept, an early accept by the LLM, an LLM answer outside the options, missing data,
  an unavailable analysis, a failed solver validation, no valid design, invented values in
  text, and the status guard.
- `tests/test_agent_review.py` covers the interface mapping (including an arbitrary rotation),
  the review and the text checks.
- `tests/test_agent_llm.py` talks to the real LM Studio model (marked `external`).

## Limitations

- One component type: the two-bolt L bracket. The load hole must be on the mid plane of
  the bolts.
- Questions go back to the user as a list. Answering means editing the specification and
  running again (the Streamlit interface of Phase 4 will make this interactive).
- The 9B model only reads text and picks among allowed actions. Its reasons are recorded but
  never used for anything else.
