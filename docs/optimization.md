# Design evaluation and optimisation (Phase 2)

## Requirements

`core.requirements.DesignRequirements` gathers everything the verification depends on:

| Field                 | Meaning                                              |
| --------------------- | ---------------------------------------------------- |
| `force`               | maximum load on the load hole, N (reference force)   |
| `min_safety_factor`   | static, yield strength / design von Mises stress     |
| `max_displacement`    | mm, optional                                         |
| `max_mass`            | kg, optional                                         |
| `fatigue`             | constant amplitude (cycles, R) or a load history with repeats, required fatigue safety factor, allowable damage, Marin and mean stress settings |
| `objective`           | `mass` or `cost`                                     |
| `candidate_materials` | names or categories, all materials when empty        |

## Evaluating one design

`optimization.evaluation.evaluate_bracket` runs:

1. CAD of the bracket
2. static analysis under +F
3. static analysis under −F, only if the cyclic load reverses (pin bearing side changes)
4. fatigue on those solutions
5. cost estimate (stock material + machining time + setup, `materials.cost`)

It returns two lists of checks:

- **verification**: the analyses are trustworthy (solver, mesh quality, equilibrium,
  linearity, fatigue method validity, S-N data available)
- **criteria**: the design meets the requirements (static safety factor, displacement, mass,
  fatigue damage, fatigue safety factor)

A design is valid only when every check of both lists passed. A failed CAD build or analysis
is stored as an error, never replaced by estimated values.

The cost model is meant for comparisons. It buys the bounding box plus 3 mm per side and
machines the removed volume at a typical removal rate (80 EUR/h). Setup is one hour per
batch of 10.

## Optimisation

`optimization.optimizer.optimize_bracket_design`:

1. ranks the materials with the Ashby indices and keeps the best `max_materials`
2. for each one, runs a seeded multivariate Optuna TPE search over the design space and
   writes its trials to `trials.json`
3. sorts all valid designs of all materials by objective
4. verifies them in that order (at most `max_verifications`, default 3): a mesh convergence
   study from the search mesh to a 0.7 times finer mesh, then a full evaluation on the finer
   mesh. The first design that passes every check is the result
5. writes `optimization.json` with every trial, the comparison and the decision trace

Default design space. Hole positions come from the mounting interface and stay fixed.

| Variable           | Range             |
| ------------------ | ----------------- |
| `base_thickness`   | 3 to 14 mm        |
| `flange_thickness` | 3 to 14 mm        |
| `width`            | 50 to 90 mm       |
| `fillet_radius`    | 2 to 12 mm        |
| `rib_thickness`    | off, or 2 to 6 mm |
| `rib_size`         | 20 to 60 mm       |

The rib fillet radius stays at 2 mm, a typical tool corner radius.

### Constraints

Each criterion becomes an Optuna constraint equal to `search_margin - margin` (≤ 0 when
satisfied). During the search a criterion must pass by 3 %, a buffer against the coarser
search mesh. One more constraint counts the failed verification checks. Infeasible
geometries (rejected by `BracketParameters`) and failed analyses get violated constraints
and a penalty objective, so TPE learns to avoid them without anything being made up.

### Search mesh

The full resolution rule (element size = half the thinnest wall) made thin trial designs
very expensive: 200k to 290k nodes and up to 4 minutes per solve. During the search the
element size therefore never drops below 2.5 mm. On a 3.2 mm wall design, the displacement
changed by less than 1 % between 4 mm and 1.2 mm elements, so the search mesh ranks designs
correctly. The final verification then checks that the search mesh had converged.

CalculiX runs SPOOLES on 4 threads (`solvers.threads` in the config), about 1.6 times
faster than one thread on these models.

### Decision trace

Besides the ranking and the choice, the trace reports:

- for each material, the requirement closest to its limit for its best design
- for the final design, the same, with a warning when every margin is above 10 %, a sign
  that the search has not converged and another iteration would pay off
- other materials whose best design is within 5 % of the chosen one, with both indicative
  costs, since such a difference is within the noise of the search
- every candidate that failed the final verification, and why

## Example

`examples/bracket_optimization.py` uses the requirements from the project description:
2000 N vertical pin load, cyclic (assumed pulsating, R = 0), 1e6 cycles, max displacement
0.5 mm, static safety factor 2, minimum mass. Results with 40 trials per material:

Results with 80 trials per material (30 minutes on an Apple Silicon Mac):

| Material  | Best valid design | Valid trials | Closest requirement   | Indicative cost |
| --------- | ----------------- | ------------ | --------------------- | --------------- |
| Al6061-T6 | **207.7 g**       | 45 / 80      | displacement, 21 %    | 25 EUR          |
| Al7075-T6 | 213.5 g           | 45 / 80      | displacement, 9 %     | 34 EUR          |
| Ti-6Al-4V | 280.9 g           | 58 / 80      | displacement, 53 %    | not computed    |

The chosen Al6061-T6 design is 54.1 mm wide, with a 10.9 mm base, an 8.5 mm flange, a
2.7 mm corner fillet and no ribs. It passed verification on the finer mesh (convergence
0.5 % on displacement, 1.1 % on stress):

| Quantity               | Value      | Requirement |
| ---------------------- | ---------- | ----------- |
| Max displacement       | 0.399 mm   | ≤ 0.5 mm    |
| Static safety factor   | 2.59       | ≥ 2         |
| Fatigue damage, 1e6 cycles | 0.00095 | ≤ 1       |
| Fatigue safety factor  | 1.76       | ≥ 1         |

The trace correctly says that the Al6061-T6 and Ti-6Al-4V searches have not converged, and
that Al7075-T6 is a close alternative (+2.8 %). With 40 trials per material (15 minutes)
the same example chose Ti-6Al-4V at 232.0 g, while the aluminium searches had stopped
with 41 % margin. The search budget matters as much as the material here. Deciding to run
another, narrower iteration from these diagnostics is a job for the Phase 3 agent.

## Limitations

- The search is a sample, not a proof of global optimality. With 40 to 80 trials per
  material the best design found changes by up to 20 % between runs (titanium: 232 g with
  40 trials, 281 g with 80). Results depend on the seed and on the budget.
- During the search every trial uses one mesh. Only the verified candidates get the
  convergence study, so a design close to a limit can fail on the finer mesh. The next
  candidate is then verified and the trace reports it.
- Every material starts from the same seed. With few trials the searches can stop at the
  same geometry, far from the limit, and the comparison between materials then reflects the
  search rather than the materials. The per-material margins in the trace show it.
- One objective at a time: no Pareto front for mass vs cost yet.
- The cost model is indicative and machining-only.
