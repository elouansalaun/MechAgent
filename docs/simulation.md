# Static simulation chain (Phase 1)

This note describes how MechAgent turns bracket parameters into a verified static result,
which conventions it relies on and where its limits are.

## Overview

```
BracketParameters ──► cad.bracket ──► BREP + face fingerprints (GeometryModel)
                                            │
StaticProblem (material, constraints,       ▼
loads, mesh settings, assumptions) ──► simulation.mesh (Gmsh, C3D10)
                                            │
                                            ▼
                         simulation.loads ──► consistent nodal forces
                                            │
                                            ▼
                     simulation.calculix ──► job.inp ──► ccx ──► job.frd / job.dat
                                            │
                                            ▼
                       simulation.static ──► StaticResult (values + checks + assumptions)
```

| Module                     | Responsibility                                                   |
| -------------------------- | ---------------------------------------------------------------- |
| `core.geometry`            | `GeometryModel`, the interface between CAD and simulation        |
| `core.checks`              | `Check` records, a result is trusted only if all checks pass     |
| `materials.material`       | `Material` with elastic constants and strengths                  |
| `cad.bracket`              | Parametric L bracket, feasibility rules, hole faces              |
| `cad.reference`            | Box and quarter plate with hole for validation                   |
| `simulation.problem`       | `Constraint`, `SurfaceForce`, `StaticProblem`                    |
| `simulation.mesh`          | Gmsh import, face matching, refinement, C3D10 mesh               |
| `simulation.loads`         | Surface force to nodal forces, pin bearing distribution          |
| `simulation.calculix`      | Deck writer, ccx runner, reaction reader                         |
| `simulation.frd`           | `.frd` reader and von Mises stress                               |
| `simulation.static`        | The full chain and its verification checks                       |
| `simulation.bracket`       | Bracket load case: bolted base, pin load in the flange           |
| `simulation.convergence`   | Mesh convergence study                                           |
| `simulation.validation`    | Analytical reference cases                                       |

## Geometry

The bracket is an L profile extruded along y, with an optional inner fillet, two optional
triangular ribs along the side edges, two bolt holes in the base (axis z) and one load hole
in the flange (axis x). The local frame is described at the top of `cad/bracket.py`.

`BracketParameters` refuses infeasible designs before any CAD work. Every hole must keep
a ligament of at least half its diameter to edges, the fillet, the ribs and the other holes.
The fillet and ribs must fit inside the legs. These rules define the design domain the
optimizer will search in Phase 2.

The CAD side exports the solid to BREP and describes each named face group (`bolt_holes`,
`load_hole`) by the centroid and area of its faces. After importing the BREP, the mesher
finds exactly one Gmsh surface per fingerprint, otherwise it stops with a `MeshError`.

## Mesh

- Quadratic tetrahedra only (C3D10). Linear tetrahedra are too stiff in bending.
- Gmsh numbers the last two edge nodes of a tet10 the other way around compared to CalculiX.
  We reorder them, and a test checks every edge node sits between the right corners.
- Default bracket mesh: element size = half the thinnest wall (two quadratic elements
  through the thickness), half that size around the holes, plus curvature refinement.
- The minimum Gmsh SICN quality is recorded and must be at least 0.1.

## Boundary conditions and loads

- Constraints fix chosen degrees of freedom on a face group. A constraint is either a
  `support` (stands for the surroundings of the part) or a `symmetry` plane.
- A `SurfaceForce` is a total force spread as a uniform traction. On quadratic triangles the
  consistent nodal forces are zero on corners and a third of the facet force on each edge
  node. The nodal forces always add up exactly to the requested force.
- With a bearing axis and centre, only the half of the hole the pin pushes against carries
  the force. This approximates a pin without running a contact analysis.

For the bracket:

| Item       | Model                                                            |
| ---------- | ---------------------------------------------------------------- |
| Bolt holes | All degrees of freedom fixed on the hole surfaces (rigid bolts)  |
| Load       | Pin bearing on the half of the load hole facing the force        |
| Material   | Linear elastic isotropic                                         |

## Results and checks

`run_static_analysis` raises a `SolverError` if ccx fails. A run only counts as successful
when ccx exits with 0, prints `Job finished`, writes no `*ERROR`, and produces non-empty
`.frd` and `.dat` files. It also raises if the results do not cover every mesh node. No
`result.json` is written for a failed run.

Every successful run records these checks:

| Check                | Rule                                                              |
| -------------------- | ----------------------------------------------------------------- |
| `solver_success`     | ccx finished and wrote its result files                           |
| `mesh_quality`       | minimum SICN ≥ 0.1                                                |
| `load_on_free_dofs`  | less than 0.1 % of the load sits on constrained dofs              |
| `equilibrium`        | reactions on constrained dofs balance the applied load to 0.1 %   |
| `small_displacement` | max displacement ≤ 1 % of the bounding box diagonal               |
| `linear_elastic`     | design von Mises stress ≤ yield strength                          |

Reactions are summed only on the constrained directions of each node. CalculiX prints all
three components for every node in the set, and an applied load on a node that is only
constrained in another direction would otherwise be counted as a reaction.

### Design stress and the support singularity

Fully fixed hole surfaces are infinitely stiff. The stress at their edges grows without bound
as the mesh is refined. On the default bracket, the raw peak went from 279 to 309 to 331 MPa
when the element size went from 4 to 2.8 to 2 mm, while stresses a few millimetres away
stayed within about 2 %.

The design value (`design_von_mises`) therefore ignores support nodes and every node closer
than `stress_exclusion_distance` to them. For the bracket that distance is a quarter of the
bolt hole diameter. The raw peak is still reported as `max_von_mises`. Symmetry planes are
exact, so they get no exclusion zone.

## Mesh convergence

`run_convergence_study` solves the problem on successively finer meshes (default factor
0.7, three levels). It checks the relative change between the two finest levels:

- max displacement: at most 2 %
- design von Mises stress: at most 5 %

Default bracket, Al 6061-T6, 2000 N downward pin load:

| Element size | Nodes   | Max displacement | Design von Mises |
| ------------ | ------- | ---------------- | ---------------- |
| 4.00 mm      | 29 326  | 0.8882 mm        | 178.5 MPa        |
| 2.80 mm      | 60 129  | 0.8890 mm        | 180.6 MPa        |
| 1.96 mm      | 135 572 | 0.8901 mm        | 175.0 MPa        |

Changes between the two finest meshes: 0.13 % on displacement and 3.2 % on stress, so the
study passes.

## Analytical validation

The reference cases go through exactly the same code path as a real analysis. A case only
passes if the error is within tolerance and every check of its analysis passed.

| Case                              | Quantity                      | Reference                          | FEM       | Error   | Limit |
| --------------------------------- | ----------------------------- | ---------------------------------- | --------- | ------- | ----- |
| Bar in tension, symmetry planes   | end displacement              | FL/EA = 0.047619 mm                | 0.047619  | 0.00 %  | 0.1 % |
| Bar in tension, symmetry planes   | axial stress                  | F/A = 100 MPa                      | 99.999    | 0.00 %  | 0.1 % |
| Clamped cantilever, tip load      | tip deflection                | Timoshenko: PL³/3EI + PL/kGA       | 0.95524   | 0.48 %  | 2 %   |
| Clamped cantilever, tip load      | bending stress, mid span      | Mc/I = 75 MPa                      | 74.999    | 0.00 %  | 2 %   |
| Finite plate with hole, tension   | peak stress at hole           | Kt σ_net (Pilkey) = 313.5 MPa      | 317.43    | 1.25 %  | 3 %   |

The cantilever is a little stiffer than beam theory because the clamp also stops the cross
section from contracting. The plate result includes the small 3D through-thickness effect
that the plane stress Kt formula ignores.

## Known limitations

- **Bolted joint model.** The bolt holes are rigidly fixed, and there is no preload,
  washer, contact with the support surface or friction. On the default bracket the critical
  zone is next to the bolt holes, so the design stress depends on this model and on the
  exclusion distance. The convergence study shows the design value is mesh independent, not
  that the joint model is right. A spring-supported or contact-based joint would be the next
  improvement.
- **Pin load.** A uniform traction on the bearing half of the hole is an approximation of
  pin contact. Hole edge stresses near the load are indicative only.
- **Linear only.** No plasticity, contact or large displacements. The `linear_elastic` and
  `small_displacement` checks flag cases where this is not acceptable. For example, a
  2000 N horizontal pull on the default bracket deflects 2.8 mm (2.2 % of the part size),
  so it is not verified.
- **Local frame.** The bracket is built in its own frame. Placing it on the user's 3D
  mounting points (a rigid transform), and checking the points are compatible with the
  parametric shape, are part of requirement interpretation in Phase 3.
- **Materials.** Only a `Material` model exists. The example uses typical handbook values
  for 6061-T6. The material database, S-N data and selection come in Phase 2.
- **Run time.** About 4 s for the default mesh (29 k nodes) and 35 s for the finest
  convergence level (135 k nodes) on an Apple Silicon Mac.
