"""MechAgent Streamlit interface.

    streamlit run src/mechagent/app/streamlit_app.py      (or: mechagent-app)

New design: write the specification as text (read by the LLM) or fill in the form, review it
instantly (questions, interpretation, analysis plan), then start the design in the
background. Runs: follow a running design and browse finished ones (design, checks, figures,
decision trace and downloads).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from mechagent.agent.planning import plan_analyses
from mechagent.agent.review import review_request
from mechagent.app.runs import launch_run, list_runs, read_progress, read_summary
from mechagent.core.config import PROJECT_ROOT, load_settings
from mechagent.core.environment import check_llm_server
from mechagent.core.request import EngineeringRequest, InterfacePoint, Vector
from mechagent.materials.database import load_materials

EXAMPLES = PROJECT_ROOT / "examples" / "requests"

DEFAULT_REQUEST = EngineeringRequest(
    mounting_points=[
        InterfacePoint(
            position=Vector(x=0, y=-18, z=200), axis=Vector(x=1, y=0, z=0), hole_diameter=9.0
        ),
        InterfacePoint(
            position=Vector(x=0, y=18, z=200), axis=Vector(x=1, y=0, z=0), hole_diameter=9.0
        ),
    ],
    load_point=InterfacePoint(
        position=Vector(x=60, y=0, z=150), axis=Vector(x=0, y=0, z=1), hole_diameter=12.0
    ),
    force=Vector(x=0, y=0, z=-2000),
    load_type="cyclic",
    cycles=1e6,
    load_ratio=0.0,
    min_safety_factor=2.0,
    max_displacement=0.5,
    objective="mass",
    candidate_materials=["aluminium", "Ti-6Al-4V"],
)


def settings():
    return load_settings()  # cheap, and the config file may change between sessions


@st.cache_resource
def materials():
    return load_materials()


def runs_dir() -> Path:
    return settings().resolve(settings().project.work_dir)


def sidebar() -> dict:
    st.sidebar.title("MechAgent")
    st.sidebar.caption("Bracket design agent: the LLM orchestrates, the solvers compute.")
    llm = settings().llm
    llm_status = check_llm_server(llm.base_url, llm.model, timeout=1.0)
    ccx = shutil.which(settings().solvers.calculix_executable)
    st.sidebar.markdown(
        f"- CalculiX: {'available' if ccx else 'NOT FOUND'}\n"
        f"- LLM `{llm.model}`: {'ready' if llm_status.available else 'not available'}"
    )
    st.sidebar.subheader("Search budget")
    return {
        "trials": st.sidebar.slider("Trials per material and iteration", 5, 80, 40, step=5),
        "iterations": st.sidebar.slider("Maximum iterations", 1, 4, 3),
        "materials": st.sidebar.slider("Materials optimised", 1, 4, 3),
    }


def _points_frame(request: EngineeringRequest) -> pd.DataFrame:
    rows = [
        {
            "x": p.position.x,
            "y": p.position.y,
            "z": p.position.z,
            "axis x": p.axis.x,
            "axis y": p.axis.y,
            "axis z": p.axis.z,
            "diameter": p.hole_diameter,
        }
        for p in request.mounting_points
    ]
    return pd.DataFrame(rows)


def _vector_inputs(label: str, vector: Vector, key: str) -> Vector:
    columns = st.columns(3)
    values = [
        column.number_input(
            f"{label} {axis}", value=float(getattr(vector, axis)), key=f"{key}_{axis}"
        )
        for column, axis in zip(columns, "xyz", strict=True)
    ]
    return Vector(x=values[0], y=values[1], z=values[2])


def request_form() -> EngineeringRequest:
    base = st.session_state.get("form_request") or DEFAULT_REQUEST
    st.markdown(
        "**Bolt holes**: position where the bolt meets the support, axis = support "
        "normal pointing into the part (mm)."
    )
    points = st.data_editor(_points_frame(base), num_rows="fixed", key="mounting_points")
    st.markdown(
        "**Load hole**: position where its axis meets the part face on the support side (mm)."
    )
    load = base.load_point
    load_position = _vector_inputs("position", load.position, "load_position")
    load_axis = _vector_inputs("axis", load.axis, "load_axis")
    load_diameter = st.number_input(
        "load hole diameter (mm)", value=float(load.hole_diameter or 12)
    )
    st.markdown("**Load** (N, global frame, maximum value)")
    force = _vector_inputs("force", base.force, "force")

    left, middle, right = st.columns(3)
    load_type = left.selectbox(
        "load type", ["cyclic", "static"], index=0 if base.load_type == "cyclic" else 1
    )
    cycles = middle.number_input("cycles", value=float(base.cycles or 1e6), format="%.0f")
    ratio = right.number_input("load ratio R", value=float(base.load_ratio or 0.0))
    left, middle, right = st.columns(3)
    safety = left.number_input("static safety factor", value=float(base.min_safety_factor or 1.5))
    displacement = middle.number_input(
        "max displacement (mm, 0 = none)", value=float(base.max_displacement or 0.0)
    )
    mass = right.number_input("max mass (kg, 0 = none)", value=float(base.max_mass or 0.0))
    left, middle, right = st.columns(3)
    objective = left.selectbox(
        "objective", ["mass", "cost"], index=0 if base.objective != "cost" else 1
    )
    width = middle.number_input("max width (mm, 0 = none)", value=float(base.max_width or 0.0))
    frequency = right.number_input(
        "excitation frequency (Hz, 0 = none)", value=float(base.excitation_frequency or 0.0)
    )
    choices = sorted(materials()) + sorted({m.category for m in materials().values()})
    candidates = st.multiselect(
        "candidate materials (empty = all)",
        choices,
        default=[c for c in (base.candidate_materials or []) if c in choices],
    )

    mounting = [
        InterfacePoint(
            position=Vector(x=r["x"], y=r["y"], z=r["z"]),
            axis=Vector(x=r["axis x"], y=r["axis y"], z=r["axis z"]),
            hole_diameter=r["diameter"],
        )
        for r in points.to_dict("records")
    ]
    return EngineeringRequest(
        component="mounting bracket",
        mounting_points=mounting,
        load_point=InterfacePoint(
            position=load_position, axis=load_axis, hole_diameter=load_diameter
        ),
        force=force,
        load_type=load_type,
        cycles=cycles if load_type == "cyclic" else None,
        load_ratio=ratio if load_type == "cyclic" else None,
        min_safety_factor=safety,
        max_displacement=displacement or None,
        max_mass=mass or None,
        objective=objective,
        candidate_materials=candidates or None,
        max_width=width or None,
        excitation_frequency=frequency or None,
        manufacturing="CNC machining",
    )


def show_review(request: EngineeringRequest, text: str | None) -> bool:
    """Review and plan the request, return True when the design can start."""
    review = review_request(request, materials(), text)
    if review.blocking:
        st.error("The agent needs answers before designing:")
        for issue in review.blocking:
            st.markdown(f"- **{issue.field}**: {issue.message}")
    for message in review.assumptions:
        st.info(f"Assumption: {message}")
    if review.interface is not None:
        interface = review.interface
        st.subheader("Interpretation")
        st.markdown(
            f"Bracket frame in global coordinates: x {interface.x_axis}, y {interface.y_axis}, "
            f"z {interface.z_axis}, origin {interface.origin}. Load in the bracket frame: "
            f"{tuple(round(v, 1) for v in interface.force_local)} N."
        )
        st.dataframe(pd.DataFrame([interface.fixed]), hide_index=True)
    if review.blocking:
        return False
    plan, _ = plan_analyses(request, review.interface, review.requirements)
    st.subheader("Analysis plan")
    st.dataframe(
        pd.DataFrame([a.model_dump() for a in plan.analyses]),
        hide_index=True,
    )
    if plan.unsupported:
        st.error("A required analysis is not available, the design cannot be verified.")
        return False
    return True


def new_design(budget: dict) -> None:
    mode = st.radio("Specification", ["Text", "Form"], horizontal=True, key="mode")
    text = None
    if mode == "Text":
        examples = {p.stem: p for p in sorted(EXAMPLES.glob("*.txt"))}
        chosen = st.selectbox("Start from an example", ["(none)"] + list(examples))
        default = examples[chosen].read_text() if chosen in examples else ""
        text = st.text_area("Specification", value=default, height=220, key=f"text_{chosen}")
        st.caption(
            "Write positions, axes and the force as (x, y, z) triples: directions "
            "given in words are sent back for confirmation."
        )
    else:
        request = request_form()

    if st.button("Review", key="review", type="primary"):
        if mode == "Text":
            from mechagent.agent.llm import LMStudioLLM

            with st.spinner("The LLM reads the specification..."):
                try:
                    request = LMStudioLLM(settings().llm).extract_request(text)
                except Exception as exc:
                    st.error(f"The LLM could not read the text: {exc}")
                    return
            st.session_state["form_request"] = request
        st.session_state["reviewed"] = (request, text)

    if "reviewed" not in st.session_state:
        return
    request, text = st.session_state["reviewed"]
    with st.expander("Request as understood", expanded=False):
        st.json(request.model_dump(exclude_none=True))
    ready = show_review(request, text)
    if mode == "Text" and not ready:
        st.caption("Fix the text, or switch to Form: it is now filled with what the LLM read.")
    if ready and st.button("Start the design", key="start"):
        # A text run reads the text again, so the trace and the text checks stay complete.
        source = {"text": text} if text is not None else {"request": request}
        run = launch_run(runs_dir(), **source, **budget)
        st.success(f"Design started in the background: {run.name}. Follow it in Runs.")


@st.fragment(run_every=10)
def progress_panel(path: Path) -> None:
    progress = read_progress(path)
    st.markdown(
        f"**Running**, iteration {progress.get('iteration', 0)}, "
        f"status {progress['status']}. This panel refreshes every 10 s."
    )
    for entry in progress["trace"][-12:]:
        st.markdown(f"- [{entry['source']}] **{entry['step']}**: {entry['decision']}")


def show_run(path: Path) -> None:
    summary = read_summary(path)
    if summary is None:
        progress_panel(path)
        return
    colour = {"verified": "green", "needs_input": "orange"}.get(summary.status, "red")
    st.markdown(f"### :{colour}[{summary.status.upper()}]  \n{summary.status_reason}")

    best = summary.best
    if best is not None and best.final is not None:
        final = best.final
        static, fatigue = final.static, final.fatigue
        line = f"**{final.material}**, {final.mass_kg * 1000:.1f} g"
        if final.cost:
            line += f", indicative cost {final.cost.total:.0f} EUR"
        st.markdown(line)
        columns = st.columns(3)
        if static:
            columns[0].metric("max displacement", f"{static.max_displacement:.3f} mm")
            columns[1].metric("yield SF", f"{static.yield_safety_factor:.2f}")
        if fatigue:
            columns[2].metric("fatigue SF", f"{fatigue.safety_factor:.2f}")
        for name in ("von_mises", "displacement", "mesh"):
            if name in summary.deliverables:
                st.image(str(path / summary.deliverables[name]))
        checks = (
            final.verification
            + final.criteria
            + (best.convergence.checks if best.convergence else [])
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"passed": c.passed, "check": c.name, "value": c.value, "limit": c.limit}
                    for c in checks
                ]
            ),
            hide_index=True,
        )
        st.dataframe(pd.DataFrame([final.params.model_dump()]), hide_index=True)

    if summary.review is not None and summary.review.blocking:
        st.error("Questions:")
        for issue in summary.review.blocking:
            st.markdown(f"- {issue.message}")
    if summary.searches:
        st.subheader("Optimisation")
        st.dataframe(
            pd.DataFrame([s.model_dump(exclude={"best_params"}) for s in summary.searches]),
            hide_index=True,
        )
    with st.expander("Decision trace"):
        for i, entry in enumerate(summary.trace, start=1):
            st.markdown(
                f"{i}. [{entry.source}] **{entry.step}**: {entry.decision}"
                + (f" ({entry.reason})" if entry.reason else "")
            )

    downloads = {
        "pdf": ("report.pdf", "application/pdf"),
        "report": ("report.md", "text/markdown"),
        "step": ("design.step", "application/step"),
    }
    columns = st.columns(len(downloads))
    for column, (key, (name, mime)) in zip(columns, downloads.items(), strict=True):
        if key in summary.deliverables:
            column.download_button(
                name,
                (path / summary.deliverables[key]).read_bytes(),
                file_name=name,
                mime=mime,
                key=f"download_{key}",
            )


def runs_view() -> None:
    runs = list_runs(runs_dir())
    if not runs:
        st.info("No agent run yet.")
        return
    labels = {run.label: run for run in runs}
    chosen = st.selectbox("Run", list(labels), key="run")
    show_run(labels[chosen].path)


def main() -> None:
    st.set_page_config(page_title="MechAgent", layout="wide")
    budget = sidebar()
    design_tab, runs_tab = st.tabs(["New design", "Runs"])
    with design_tab:
        new_design(budget)
    with runs_tab:
        runs_view()


main()
