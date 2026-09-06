"""Engineering report of an agent run, as Markdown.

Everything in it comes from the agent state: tool results, rule decisions and the LLM choices,
each with its source. Nothing is rewritten by the LLM.
"""

from __future__ import annotations

STATUS_TEXT = {
    "verified": "VERIFIED",
    "not_verified": "NOT VERIFIED",
    "needs_input": "NEEDS INPUT",
    "unverifiable": "UNVERIFIABLE",
    "solver_not_validated": "SOLVER NOT VALIDATED",
    "error": "ERROR",
    "running": "INCOMPLETE",
}

LIMITATIONS = [
    "Bolt holes are modelled as rigidly fixed: no preload, contact or friction. Stresses "
    "within a quarter of the bolt hole diameter of the holes are excluded from design values, "
    "and fatigue of the bolted joint itself is not assessed.",
    "The pin load is a uniform traction on the bearing half of the hole, without contact.",
    "Linear elastic, small displacements. Stress-life fatigue only, no notch sensitivity "
    "reduction (conservative), no crack growth.",
    "Buckling is screened with a conservative Euler column, there is no buckling FEM.",
    "Material properties, S-N data and prices are typical handbook values, not certified "
    "design allowables.",
    "The optimisation samples the design space: the result is a good verified design, not a "
    "proven global optimum.",
]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines + [""]


def _vec(values, digits: int = 4) -> str:
    return "(" + ", ".join(f"{v:.{digits}g}" for v in values) + ")"


def _request_section(state) -> list[str]:
    request = state.get("request")
    lines = ["## 1. Request", ""]
    if state.get("text"):
        lines += ["Original text:", ""] + [f"> {line}" for line in state["text"].splitlines()]
        lines.append("")
    if request is None:
        return lines + ["No request could be read.", ""]
    rows = []
    for i, point in enumerate(request.mounting_points):
        rows.append(
            [
                f"bolt hole {i + 1}",
                _vec(point.position.array()),
                _vec(point.axis.array()),
                point.hole_diameter,
            ]
        )
    if request.load_point:
        point = request.load_point
        rows.append(
            [
                "load hole",
                _vec(point.position.array()),
                _vec(point.axis.array()),
                point.hole_diameter,
            ]
        )
    if rows:
        lines += _table(["Interface", "Position (mm)", "Axis", "Diameter (mm)"], rows)
    values = request.model_dump(
        exclude={"mounting_points", "load_point", "notes"}, exclude_none=True
    )
    if request.force:
        values["force"] = _vec(request.force.array()) + " N"
    lines += _table(["Requirement", "Value"], [[k, v] for k, v in values.items()])
    return lines


def _review_section(state) -> list[str]:
    review = state.get("review")
    if review is None:
        return []
    lines = ["## 2. Review of the request", ""]
    if not review.issues:
        return lines + ["The request is complete and consistent.", ""]
    rows = [
        [issue.kind, issue.field, "yes" if issue.blocking else "no", issue.message]
        for issue in review.issues
    ]
    return lines + _table(["Kind", "Field", "Blocking", "Message"], rows)


def _interface_section(state) -> list[str]:
    review = state.get("review")
    if review is None or review.interface is None:
        return []
    interface = review.interface
    lines = [
        "## 3. Bracket placement",
        "",
        "The bracket is built in its own frame and placed on the requested interface:",
        "",
    ]
    lines += _table(
        ["Local axis", "Global direction"],
        [
            ["x (load hole to bolts)", _vec(interface.x_axis)],
            ["y", _vec(interface.y_axis)],
            ["z (out of the support)", _vec(interface.z_axis)],
            ["origin", _vec(interface.origin)],
        ],
    )
    lines.append(f"Load in the bracket frame: {_vec(interface.force_local)} N.")
    lines.append("")
    lines += _table(
        ["Parameter fixed by the interface", "Value (mm)"],
        [[k, f"{v:.2f}"] for k, v in interface.fixed.items()],
    )
    return lines


def _plan_section(state) -> list[str]:
    plan = state.get("plan")
    if plan is None:
        return []
    rows = [
        [
            a.analysis,
            "yes" if a.required else "no",
            "yes" if a.available else "no",
            a.method,
            a.reason,
        ]
        for a in plan.analyses
    ]
    return ["## 4. Analysis plan", ""] + _table(
        ["Analysis", "Required", "Available", "Method", "Reason"], rows
    )


def _validation_section(state) -> list[str]:
    cases = state.get("validation")
    if not cases:
        return []
    rows = [
        [
            c.name,
            c.quantity,
            f"{c.fem:.5g}",
            f"{c.reference:.5g}",
            c.unit,
            f"{100 * c.relative_error:.2f} %",
            f"{100 * c.tolerance:.1f} %",
            "PASS" if c.passed else "FAIL",
        ]
        for c in cases
    ]
    return ["## 5. Solver validation", ""] + _table(
        ["Case", "Quantity", "FEM", "Reference", "Unit", "Error", "Limit", "Result"], rows
    )


def _materials_section(state) -> list[str]:
    selection = state.get("selection")
    if selection is None:
        return []
    rows = [
        [r.name, f"{r.score:.2f}", r.governing]
        + [f"{r.normalised[c]:.2f}" for c in selection.criteria]
        for r in selection.ranking
    ]
    lines = ["## 6. Material selection", ""]
    lines.append(
        f"Ashby indices for a plate in bending, minimum {selection.objective}, normalised by "
        "the best candidate. A material scores its worst index."
    )
    lines.append("")
    lines += _table(["Material", "Score", "Limiting"] + selection.criteria, rows)
    for rejected in selection.rejected:
        lines.append(f"- Rejected {rejected.name}: {rejected.reason}")
    return lines + [""]


def _iterations_section(state) -> list[str]:
    history = state.get("history")
    if not history:
        return []
    objective = state["requirements"].objective
    rows = []
    for i, searches in enumerate(history, start=1):
        for name, optimization in searches.items():
            best = optimization.best
            value = best.objective_value(objective) if best else None
            margin = min(c.margin for c in best.criteria) if best else None
            rows.append(
                [
                    i,
                    name,
                    len(optimization.trials),
                    optimization.n_valid,
                    f"{value * 1000:.1f} g"
                    if value is not None and objective == "mass"
                    else (f"{value:.2f} EUR" if value is not None else "none"),
                    f"{100 * margin:.0f} %" if margin is not None else "-",
                ]
            )
    return ["## 7. Optimisation", ""] + _table(
        ["Iteration", "Material", "Trials", "Valid", "Best", "Smallest margin"], rows
    )


def _final_section(state) -> list[str]:
    best = state.get("best")
    if best is None or best.final is None:
        return ["## 8. Final design", "", "No design passed every required verification.", ""]
    final = best.final
    lines = ["## 8. Final design", ""]
    lines.append(
        f"Material: **{final.material}**, mass **{final.mass_kg * 1000:.1f} g**"
        + (f", indicative cost {final.cost.total:.2f} EUR" if final.cost else "")
        + "."
    )
    lines.append("")
    params = final.params.model_dump()
    if final.params.rib_thickness == 0:
        params = {k: v for k, v in params.items() if not k.startswith("rib_")}
    lines += _table(["Parameter", "Value (mm)"], [[k, f"{v:.2f}"] for k, v in params.items()])

    interface = state["review"].interface
    holes = [(h.name, interface.to_global(h.center)) for h in _holes(final.params)]
    lines += _table(["Hole", "Global position (mm)"], [[n, _vec(p)] for n, p in holes])

    static = final.static
    results = []
    if static is not None:
        results += [
            ["max displacement", f"{static.max_displacement:.4g} mm"],
            ["design von Mises stress", f"{static.design_von_mises:.4g} MPa"],
            [
                "raw peak von Mises (support singularity included)",
                f"{static.max_von_mises:.4g} MPa",
            ],
            ["yield safety factor", f"{static.yield_safety_factor:.3g}"],
            ["mesh", f"{static.n_nodes} nodes, {static.n_elements} C3D10 elements"],
        ]
    if final.fatigue:
        results += [
            ["fatigue damage over the required life", f"{final.fatigue.damage:.3g}"],
            ["fatigue safety factor on the load", f"{final.fatigue.safety_factor:.3g}"],
        ]
    if results:
        lines += _table(["Result", "Value"], results)

    checks = (
        final.verification + final.criteria + (best.convergence.checks if best.convergence else [])
    )
    rows = [
        [
            "PASS" if c.passed else "FAIL",
            c.name,
            "" if c.value is None else f"{c.value:.4g}",
            "" if c.limit is None else f"{c.limit:.4g}",
            c.message,
        ]
        for c in checks
    ]
    return (
        lines + ["### Checks", ""] + _table(["Result", "Check", "Value", "Limit", "Meaning"], rows)
    )


FIGURE_CAPTIONS = {
    "mesh": "Final mesh (bracket frame)",
    "von_mises": "von Mises stress on the deformed shape (bracket frame)",
    "displacement": "Displacement on the deformed shape (bracket frame)",
}


def _files_section(state) -> list[str]:
    files = state.get("deliverables") or {}
    figures = [name for name in FIGURE_CAPTIONS if name in files]
    if not figures and "step" not in files:
        return []
    lines = ["### Figures and files", ""]
    if "step" in files:
        lines += [f"- CAD model in the global frame of the request: `{files['step']}`", ""]
    for name in figures:
        lines += [f"![{FIGURE_CAPTIONS[name]}]({files[name]})", ""]
    return lines


def _holes(params):
    from mechagent.cad.bracket import bracket_holes

    return bracket_holes(params)


def _assumptions_section(state) -> list[str]:
    assumptions = []
    review = state.get("review")
    if review is not None:
        assumptions += [f"(request) {a}" for a in review.assumptions]
    best = state.get("best")
    if best is not None and best.final is not None and best.final.static is not None:
        assumptions += best.final.static.assumptions
        if best.final.fatigue:
            assumptions += best.final.fatigue.assumptions
    if not assumptions:
        return []
    return ["## 9. Assumptions", ""] + [f"- {a}" for a in assumptions] + [""]


def _trace_section(state) -> list[str]:
    lines = ["## 10. Decision trace", ""]
    for i, entry in enumerate(state.get("trace", []), start=1):
        text = f"{i}. [{entry.source}] **{entry.step}**: {entry.decision}"
        if entry.reason:
            text += f" ({entry.reason})"
        lines.append(text)
    return lines + [""]


def render_report(state) -> str:
    status = state.get("status", "running")
    lines = [
        "# MechAgent engineering report",
        "",
        f"**Status: {STATUS_TEXT.get(status, status)}**: {state.get('status_reason', '')}",
        "",
        "Units: mm, N, MPa, kg. Every number comes from deterministic tools, the LLM only "
        "read the request and chose between allowed actions (marked [llm] in the trace).",
        "",
    ]
    review = state.get("review")
    if review is not None and review.blocking:
        lines += ["## Questions before the design can start", ""]
        lines += [f"- {issue.message}" for issue in review.blocking] + [""]
    for section in (
        _request_section,
        _review_section,
        _interface_section,
        _plan_section,
        _validation_section,
        _materials_section,
        _iterations_section,
        _final_section,
        _files_section,
        _assumptions_section,
        _trace_section,
    ):
        lines += section(state)
    lines += ["## 11. Limitations", ""] + [f"- {item}" for item in LIMITATIONS] + [""]
    return "\n".join(lines)
