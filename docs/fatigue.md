# Materials and fatigue (Phase 2)

## Material database

`data/materials.yaml` holds six candidates. The values are **typical handbook data**
(ASM / MatWeb style data sheets, EN minimum strengths, Shigley estimates), not certified
design allowables. A real design must use supplier, MMPDS or test data.

| Material     | E (MPa) | ρ (kg/dm³) | Sy (MPa) | Su (MPa) | S-N knee (MPa @ cycles) | Endurance limit | Price (EUR/kg) |
| ------------ | ------- | ---------- | -------- | -------- | ----------------------- | --------------- | -------------- |
| Al6061-T6    | 68 900  | 2.70       | 276      | 310      | 96.5 @ 5e8              | no              | 4.0            |
| Al7075-T6    | 71 700  | 2.81       | 503      | 572      | 159 @ 5e8               | no              | 8.0            |
| S355J2       | 210 000 | 7.85       | 355      | 510      | 255 @ 1e6               | yes             | 1.5            |
| 42CrMo4-QT   | 210 000 | 7.85       | 650      | 900      | 450 @ 1e6               | yes             | 3.0            |
| X5CrNi18-10  | 193 000 | 8.00       | 215      | 505      | 240 @ 1e7               | yes             | 4.5            |
| Ti-6Al-4V    | 113 800 | 4.43       | 880      | 950      | 510 @ 1e7               | no              | 30.0           |

The S-N data are for smooth polished specimens under fully reversed loading (R = -1). Where
no measured curve is given, the strength at 1e3 cycles is f·Su (f from Shigley fig. 6-18),
and the steel endurance limit is 0.5·Su at 1e6 cycles.

## Material ranking

`materials.selection.rank_materials` shortlists candidates before optimisation using Ashby
performance indices for a plate in bending, the closest simple idealisation of a bracket
wall:

| Criterion | Index (min mass) | Used when                |
| --------- | ---------------- | ------------------------ |
| Strength  | Sy^½ / ρ         | always                   |
| Stiffness | E^⅓ / ρ          | a displacement limit is set |
| Fatigue   | Sf(N)^½ / ρ      | the load is cyclic       |

For a minimum cost objective, ρ is replaced by ρ·price. Each index is divided by the best
value among the candidates. A material's score is its lowest normalised index, because
whichever criterion is active will size the part.

Materials without S-N data are rejected for cyclic loads, and materials without price data
are rejected for a cost objective. The ranking only chooses which materials get optimised;
the final choice compares the optimised designs.

## Component S-N curve

`materials.fatigue.SNCurve` is a Basquin line in log-log space through (1e3, S_1e3) and
(N_knee, S_knee). Above the knee it is either flat (endurance limit, infinite life below
it) or the same line continued (aluminium, titanium).

The knee strength is reduced from specimen to component by Shigley's Marin factors:

- surface finish ka = a·Su^b (polished, ground, machined, hot rolled, as forged)
- size kb, 0.9 by default, typical for 10 to 50 mm sections
- reliability ke (0.5 to 0.9999, default 0.99 → 0.814)

The 1e3-cycle point is not corrected, following Shigley. Below 1e3 cycles stress-life is not
valid, and the `fatigue_high_cycle_range` check fails.

## Fatigue evaluation

Loads are load factors λ of the reference force F_ref (the maximum load, |λ| ≤ 1). The
analysis is linear, so σ(λ) = λ·σ(F_ref).

A pin cannot pull on its hole, so a reversed load bears on the opposite half of the hole.
When the load factor goes negative, the evaluation runs a second static analysis with
−F_ref on the same mesh and uses σ(λ) = |λ|·σ(−F_ref) for λ < 0.

Load histories are counted with the ASTM E1049 rainflow method (`simulation.rainflow`).
Each counted cycle keeps its two extreme load factors. Constant amplitude loading is one
block from R to 1.

For each cycle and each node outside the support exclusion zone:

1. amplitude σa = von Mises of (σ_high − σ_low)/2
2. mean σm = signed von Mises of (σ_high + σ_low)/2, signed by the principal stress with the
   largest absolute value
3. equivalent fully reversed amplitude:
   - Goodman σa / (1 − σm/Su)
   - Gerber σa / (1 − (σm/Su)²)
   - Soderberg σa / (1 − σm/Sy)
   - SWT √(σmax·σa)

   Compressive means get no benefit. A mean at or above the strength is a static failure
   (infinite amplitude).
4. N from the component S-N curve, damage D = Σ n/N (Palmgren-Miner)

The **fatigue safety factor** is the load factor s that brings the maximum damage to the
allowable value: D(s·loads) = D_allow, solved with Brent's method. With an endurance limit
the damage jumps from 0 at the knee, and the safety factor finds that jump.

### Checks

| Check                      | Kind         | Rule                                         |
| -------------------------- | ------------ | -------------------------------------------- |
| `fatigue_cyclic_yield`     | verification | max von Mises at a cycle end ≤ Sy            |
| `fatigue_high_cycle_range` | verification | every block lasts at least 1e3 cycles        |
| `fatigue_damage`           | criterion    | D ≤ allowable damage (default 1)             |
| `fatigue_safety_factor`    | criterion    | s ≥ required factor                          |
| `fatigue_data`             | verification | fails when a cyclic load meets a material without S-N data |

### Validation

- The rainflow counter reproduces the ASTM E1049 example (ranges 3, 4, 6, 8, 9 counted 0.5,
  1.5, 0.5, 1.0, 0.5).
- Goodman, Gerber, Soderberg, SWT, Marin factors, Miner sums and the S-N curve match hand
  calculations in `tests/test_fatigue.py` and `tests/test_materials.py`.
- The safety factor matches the closed-form Goodman solution
  s = S_N / (σa + S_N·σm/Su) to 1e-5.
- The axial bar validation case runs the full FE chain with 200 MPa amplitude and 100 MPa
  mean for 1e5 cycles. Its damage, 0.13463, matches the hand calculation (0.13465) to
  0.015 %.

### Limitations

- Stress-life only: no strain-life for low cycle fatigue, no crack growth.
- Proportional loading per branch (one reference solution per load direction). Several
  independent load directions would need a critical plane method.
- No notch sensitivity: FE peak stresses are used directly (Kf = Kt), which is conservative
  at sharp notches.
- Stresses at the bolt holes sit in the support exclusion zone. Fatigue at the bolted joint
  itself is not assessed (see the bolt model limitation in [simulation.md](simulation.md)).
- Material data are typical values.
