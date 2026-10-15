"""Render the animated explainer of MechAgent, in English and French, from a verified agent run.

    python scripts/make_explainer_gif.py runs/<timestamp>_agent --figure <run>/figures/von_mises.png

Every number shown comes from the run: the Optuna trials of the chosen material and the checks
table of its report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, S = 1200, 675, 2
FPS = 12.5
FADE = 0.35

BG = (245, 243, 239)
CARD = (252, 251, 249)
INK = (31, 30, 29)
MUTED = (118, 113, 106)
LINE = (220, 215, 206)
EDGE = (184, 177, 166)
ACCENT = (201, 113, 84)
ACCENT_SOFT = (244, 222, 210)
GREEN = (70, 124, 88)
GREEN_SOFT = (222, 236, 225)
BLUE = (72, 104, 150)
BLUE_SOFT = (222, 230, 242)
RULE = (150, 124, 74)
RULE_SOFT = (240, 231, 210)

AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"
WEIGHTS = {"bold": 0, "demi": 2, "medium": 5, "regular": 7}

KIND_COLORS = {"llm": (ACCENT, ACCENT_SOFT), "rule": (RULE, RULE_SOFT), "tool": (BLUE, BLUE_SOFT)}

TEXT = {
    "en": {
        "subtitle": "From a written specification to a verified, optimised bracket",
        "tagline": "A local LLM orchestrates, deterministic tools do the physics",
        "read_title": "1   The LLM reads the request",
        "spec": "Shelf bracket on a vertical wall, z points up. Two 9 mm bolt holes at "
        "(0, -18, 200) and (0, 18, 200) mm, bolt axis (1, 0, 0). A pin in a 12 mm hole at "
        "(60, 0, 150) mm carries (0, 0, -2000) N, cyclic from zero (R = 0), for 1 million "
        "cycles. Maximum deflection 0.5 mm, safety factor 2 on yield. Minimise the mass.",
        "fields": [
            ("bolt holes", "Ø 9 at (0, ±18, 200) mm", ["9 mm", "(0, -18, 200)", "(0, 18, 200)"]),
            ("load", "(0, 0, -2000) N at (60, 0, 150)", ["(0, 0, -2000)", "(60, 0, 150)"]),
            ("fatigue", "R = 0, 1 million cycles", ["R = 0", "1 million cycles"]),
            ("deflection", "≤ 0.5 mm", ["0.5 mm"]),
            ("yield safety factor", "≥ 2", ["safety factor 2"]),
            ("objective", "minimum mass", ["Minimise the mass"]),
        ],
        "found": "found in the text",
        "read_note": "Every number and vector must appear in the text. "
        "If something is missing, the agent stops and asks.",
        "graph_title": "2   A LangGraph agent runs the workflow",
        "nodes": [
            "read text",
            "review",
            "plan analyses",
            "validate solver",
            "select materials",
            "optimise",
            "verify",
            "decide",
            "report",
        ],
        "kinds": {"llm": "LLM", "rule": "rule", "tool": "tool"},
        "ask": "missing data: stop and ask",
        "loop": "refine or widen the search",
        "iteration": "iteration {n}: best {mass:.0f} g, the LLM chooses {action}",
        "actions": {"refine": "to refine", "accept": "to accept"},
        "graph_note": "Each step is traced with its source: LLM, rule or tool.",
        "split_title": "3   The LLM decides, tools compute",
        "llm_card": "Local LLM",
        "llm_model": "Qwen 3.5 9B in LM Studio",
        "llm_items": [
            "reads the request, checked against the text",
            "chooses the next step among allowed actions",
            "computes nothing",
        ],
        "llm_timing": "about 14 s to read the text, 2 s per decision",
        "tools_card": "Deterministic tools",
        "tools": [
            ("CadQuery", "parametric geometry"),
            ("Gmsh", "quadratic tetrahedral mesh"),
            ("CalculiX", "static finite elements"),
            ("S-N, Goodman, Miner", "fatigue"),
            ("Optuna", "parametric optimisation"),
            ("PyVista, ReportLab", "figures and report"),
        ],
        "search_title": "4   Optuna explores the geometry, CalculiX checks each trial",
        "trial": "trial",
        "mass": "mass (g)",
        "refine": "iteration 2: refine around the best",
        "valid": "meets every requirement",
        "invalid": "fails a requirement",
        "infeasible": "infeasible geometry",
        "params": [
            ("width", "width"),
            ("base_thickness", "base thickness"),
            ("flange_thickness", "flange thickness"),
            ("rib_thickness", "rib thickness"),
        ],
        "fails": "fails: {check}",
        "checks": {
            "max_displacement": "deflection",
            "static_safety_factor": "yield",
            "small_displacement": "small displacement",
            "mesh_quality": "mesh quality",
            "fatigue_safety_factor": "fatigue",
        },
        "rejected": "rejected",
        "materials_note": "{material} shown. {others} were searched the same way.",
        "and": "and",
        "verify_title": "5   The best design is verified before it is accepted",
        "verify_rows": [
            ("Solver validated on analytical cases", "{passed} / {cases} within tolerance"),
            ("Equilibrium: reactions balance the load", "error {equilibrium}"),
            ("Mesh convergence on a finer mesh", "Δu {conv_u} %,  Δσ {conv_s} %"),
            ("Deflection", "{disp} mm ≤ {disp_limit} mm"),
            ("Yield safety factor", "{sf} ≥ {sf_limit}"),
            ("Fatigue, 1 million cycles, Goodman, Miner", "safety factor {fatigue_sf}"),
        ],
        "verified": "VERIFIED",
        "verify_note": "One missing or failed check, and the design is never declared valid.",
        "files_title": "6   What you get",
        "files": [
            ("report.pdf", "assumptions, units, checks and the source of every decision"),
            ("design.step", "the bracket placed in your global coordinates"),
            ("figures", "mesh, von Mises stress, displacement"),
        ],
        "files_note": "From the Streamlit app or the command line",
        "figure": "von Mises stress on the deformed shape",
        "result": "verified in {iterations} iterations, {trials} finite element trials",
    },
    "fr": {
        "subtitle": "D'un cahier des charges écrit à une équerre optimisée et vérifiée",
        "tagline": "Un LLM local orchestre, des outils déterministes font la physique",
        "read_title": "1   Le LLM lit la demande",
        "spec": "Équerre d'étagère sur un mur vertical, z vers le haut. Deux trous de 9 mm "
        "en (0, -18, 200) et (0, 18, 200) mm, axe des vis (1, 0, 0). Un axe dans un trou de "
        "12 mm en (60, 0, 150) mm porte (0, 0, -2000) N, cyclique depuis zéro (R = 0), "
        "pendant 1 million de cycles. Flèche maximale 0,5 mm, coefficient de sécurité 2 sur "
        "la limite élastique. Minimiser la masse.",
        "fields": [
            (
                "trous de fixation",
                "Ø 9 en (0, ±18, 200) mm",
                ["9 mm", "(0, -18, 200)", "(0, 18, 200)"],
            ),
            ("charge", "(0, 0, -2000) N en (60, 0, 150)", ["(0, 0, -2000)", "(60, 0, 150)"]),
            ("fatigue", "R = 0, 1 million de cycles", ["R = 0", "1 million de cycles"]),
            ("flèche", "≤ 0,5 mm", ["0,5 mm"]),
            ("sécurité élastique", "≥ 2", ["coefficient de sécurité 2"]),
            ("objectif", "masse minimale", ["Minimiser la masse"]),
        ],
        "found": "trouvé dans le texte",
        "read_note": "Chaque nombre et chaque vecteur doit figurer dans le texte. "
        "S'il manque une donnée, l'agent s'arrête et demande.",
        "graph_title": "2   Un agent LangGraph pilote le processus",
        "nodes": [
            "lecture du texte",
            "revue",
            "plan d'analyses",
            "validation solveur",
            "choix matériaux",
            "optimisation",
            "vérification",
            "décision",
            "rapport",
        ],
        "kinds": {"llm": "LLM", "rule": "règle", "tool": "outil"},
        "ask": "donnée manquante : arrêt et questions",
        "loop": "affiner ou élargir la recherche",
        "iteration": "itération {n} : meilleur {mass:.0f} g, le LLM choisit {action}",
        "actions": {"refine": "d'affiner", "accept": "d'accepter"},
        "graph_note": "Chaque étape est tracée avec sa source : LLM, règle ou outil.",
        "split_title": "3   Le LLM décide, les outils calculent",
        "llm_card": "LLM local",
        "llm_model": "Qwen 3.5 9B dans LM Studio",
        "llm_items": [
            "lit la demande, vérifiée contre le texte",
            "choisit l'étape suivante parmi les actions permises",
            "ne calcule rien",
        ],
        "llm_timing": "environ 14 s pour lire le texte, 2 s par décision",
        "tools_card": "Outils déterministes",
        "tools": [
            ("CadQuery", "géométrie paramétrique"),
            ("Gmsh", "maillage tétraédrique quadratique"),
            ("CalculiX", "éléments finis statiques"),
            ("S-N, Goodman, Miner", "fatigue"),
            ("Optuna", "optimisation paramétrique"),
            ("PyVista, ReportLab", "figures et rapport"),
        ],
        "search_title": "4   Optuna explore la géométrie, CalculiX vérifie chaque essai",
        "trial": "essai",
        "mass": "masse (g)",
        "refine": "itération 2 : affiner autour du meilleur",
        "valid": "respecte toutes les exigences",
        "invalid": "échoue sur une exigence",
        "infeasible": "géométrie impossible",
        "params": [
            ("width", "largeur"),
            ("base_thickness", "épaisseur de base"),
            ("flange_thickness", "épaisseur d'aile"),
            ("rib_thickness", "épaisseur de nervure"),
        ],
        "fails": "échoue : {check}",
        "checks": {
            "max_displacement": "flèche",
            "static_safety_factor": "limite élastique",
            "small_displacement": "petits déplacements",
            "mesh_quality": "qualité du maillage",
            "fatigue_safety_factor": "fatigue",
        },
        "rejected": "rejeté",
        "materials_note": "{material} ici. {others} sont optimisés de la même façon.",
        "and": "et",
        "verify_title": "5   Le meilleur design est vérifié avant d'être accepté",
        "verify_rows": [
            ("Solveur validé sur des cas analytiques", "{passed} / {cases} dans la tolérance"),
            ("Équilibre : réactions et charge s'équilibrent", "écart {equilibrium}"),
            ("Convergence sur un maillage plus fin", "Δu {conv_u} %,  Δσ {conv_s} %"),
            ("Flèche", "{disp} mm ≤ {disp_limit} mm"),
            ("Coefficient de sécurité élastique", "{sf} ≥ {sf_limit}"),
            ("Fatigue, 1 million de cycles, Goodman, Miner", "coefficient {fatigue_sf}"),
        ],
        "verified": "VÉRIFIÉ",
        "verify_note": "Une seule vérification manquante ou en échec, "
        "et le design n'est jamais déclaré valide.",
        "files_title": "6   Ce que vous obtenez",
        "files": [
            ("report.pdf", "hypothèses, unités, vérifications et source de chaque décision"),
            ("design.step", "l'équerre placée dans vos coordonnées globales"),
            ("figures", "maillage, contrainte de von Mises, déplacement"),
        ],
        "files_note": "Depuis l'application Streamlit ou la ligne de commande",
        "figure": "contrainte de von Mises sur la déformée",
        "result": "vérifiée en {iterations} itérations, {trials} essais éléments finis",
    },
}


# --- run data ------------------------------------------------------------------------------


def load_run(run_dir: Path) -> dict:
    report = (run_dir / "report.md").read_text()
    material, mass = re.search(r"Material: \*\*(.+?)\*\*, mass \*\*([\d.]+) g\*\*", report).groups()
    checks = {
        name: (float(value), float(limit))
        for name, value, limit in re.findall(
            r"^\| PASS \| (\w+) \| ([-\d.e+]+) \| ([-\d.e+]+) \|", report, re.MULTILINE
        )
    }
    passed, cases = re.search(r"(\d+) of (\d+) analytical reference cases passed", report).groups()

    iterations = sorted(run_dir.glob("iteration_*"))
    materials = sorted(p.name for p in iterations[0].iterdir() if (p / "trials.json").exists())
    trials = []
    for iteration in iterations:
        for trial in json.loads((iteration / material / "trials.json").read_text()):
            trials.append({**trial, "iteration": int(iteration.name.split("_")[1])})
    total = sum(
        len(json.loads((iteration / name / "trials.json").read_text()))
        for iteration in iterations
        for name in materials
    )
    return {
        "material": material,
        "mass": float(mass),
        "checks": checks,
        "passed": int(passed),
        "cases": int(cases),
        "trials": trials,
        "others": [name for name in materials if name != material],
        "iterations": len(iterations),
        "total_trials": total,
    }


def best_by_iteration(trials: list[dict]) -> dict[int, float]:
    best = {}
    for trial in trials:
        if trial["valid"]:
            mass = trial["objective"] * 1000
            best[trial["iteration"]] = min(best.get(trial["iteration"], math.inf), mass)
    return best


def number(value: float, digits: int, lang: str) -> str:
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if lang == "fr" else text


def number_e(value: float, lang: str) -> str:
    text = f"{value:.1e}".replace("e-0", "e-")
    return text.replace(".", ",") if lang == "fr" else text


def verify_values(run: dict, lang: str) -> dict:
    checks = run["checks"]
    return {
        "passed": run["passed"],
        "cases": run["cases"],
        "equilibrium": number_e(checks["equilibrium"][0], lang),
        "conv_u": number(checks["mesh_convergence_displacement"][0] * 100, 2, lang),
        "conv_s": number(checks["mesh_convergence_stress"][0] * 100, 2, lang),
        "disp": number(checks["max_displacement"][0], 3, lang),
        "disp_limit": number(checks["max_displacement"][1], 1, lang),
        "sf": number(checks["static_safety_factor"][0], 2, lang),
        "sf_limit": number(checks["static_safety_factor"][1], 0, lang),
        "fatigue_sf": number(checks["fatigue_safety_factor"][0], 2, lang),
    }


# --- drawing -------------------------------------------------------------------------------


@lru_cache
def font(size: float, weight: str = "regular", mono: bool = False):
    if mono:
        return ImageFont.truetype(MENLO, round(size * S), index=1 if weight == "bold" else 0)
    return ImageFont.truetype(AVENIR, round(size * S), index=WEIGHTS[weight])


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def ease(x: float) -> float:
    x = clamp(x)
    return x * x * (3 - 2 * x)


def appear(t: float, start: float, duration: float = 0.4) -> float:
    return ease((t - start) / duration)


def mix(a, b, p: float):
    return tuple(round(a[i] + (b[i] - a[i]) * p) for i in range(3))


class Frame:
    def __init__(self):
        self.image = Image.new("RGB", (W * S, H * S), BG)
        self.draw = ImageDraw.Draw(self.image)

    def text(
        self,
        xy,
        text,
        size,
        color=INK,
        weight="regular",
        anchor="la",
        p=1.0,
        bg=BG,
        mono=False,
        rise=10,
    ):
        if p <= 0:
            return
        x, y = xy
        self.draw.text(
            (x * S, (y + (1 - p) * rise) * S),
            text,
            font=font(size, weight, mono),
            fill=mix(bg, color, p),
            anchor=anchor,
        )

    def width(self, text, size, weight="regular", mono=False) -> float:
        return font(size, weight, mono).getlength(text) / S

    def box(self, xy, fill=CARD, outline=LINE, radius=12, width=1.0, p=1.0, bg=BG):
        if p <= 0:
            return
        self.draw.rounded_rectangle(
            [v * S for v in xy],
            radius=radius * S,
            fill=mix(bg, fill, p) if fill else None,
            outline=mix(bg, outline, p) if outline else None,
            width=max(1, round(width * S)),
        )

    def line(self, points, color, width=2.0, p=1.0, bg=BG):
        if p <= 0:
            return
        self.draw.line(
            [(x * S, y * S) for x, y in points],
            fill=mix(bg, color, p),
            width=round(width * S),
            joint="curve",
        )

    def dashed(self, start, end, color, width=1.5, dash=6.0, p=1.0, bg=BG):
        (x0, y0), (x1, y1) = start, end
        length = math.hypot(x1 - x0, y1 - y0)
        for i in range(int(length // (2 * dash)) + 1):
            a, b = 2 * i * dash / length, min(1.0, (2 * i + 1) * dash / length)
            if a < 1:
                self.line(
                    [
                        (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a),
                        (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b),
                    ],
                    color,
                    width,
                    p,
                    bg,
                )

    def arrow(self, points, color, width=2.0, p=1.0, bg=BG, head=8.0):
        self.line(points, color, width, p, bg)
        (x0, y0), (x1, y1) = points[-2], points[-1]
        angle = math.atan2(y1 - y0, x1 - x0)
        tip = [(x1, y1)] + [
            (x1 - head * math.cos(angle + s * 0.45), y1 - head * math.sin(angle + s * 0.45))
            for s in (-1, 1)
        ]
        if p > 0:
            self.draw.polygon([(x * S, y * S) for x, y in tip], fill=mix(bg, color, p))

    def circle(self, center, radius, fill=None, outline=None, width=1.5, p=1.0, bg=BG):
        if p <= 0:
            return
        x, y = center
        self.draw.ellipse(
            [(x - radius) * S, (y - radius) * S, (x + radius) * S, (y + radius) * S],
            fill=mix(bg, fill, p) if fill else None,
            outline=mix(bg, outline, p) if outline else None,
            width=max(1, round(width * S)),
        )

    def tick(self, center, radius, p=1.0, bg=BG):
        x, y = center
        self.circle(center, radius * (0.6 + 0.4 * p), fill=GREEN, p=p, bg=bg)
        r = radius * 0.5
        self.line(
            [(x - r, y), (x - r * 0.25, y + r * 0.7), (x + r, y - r * 0.6)],
            CARD,
            width=radius * 0.28,
            p=p,
            bg=GREEN,
        )

    def cross(self, center, size, color, width=1.5, p=1.0, bg=BG):
        x, y = center
        self.line([(x - size, y - size), (x + size, y + size)], color, width, p, bg)
        self.line([(x - size, y + size), (x + size, y - size)], color, width, p, bg)

    def paste(self, image: Image.Image, xy, p=1.0):
        if p <= 0:
            return
        scaled = image.resize((round(image.width * S), round(image.height * S)), Image.LANCZOS)
        region = (round(xy[0] * S), round(xy[1] * S))
        under = self.image.crop((*region, region[0] + scaled.width, region[1] + scaled.height))
        self.image.paste(Image.blend(under, scaled.convert("RGB"), p), region)

    def result(self) -> Image.Image:
        return self.image.resize((W, H), Image.LANCZOS)


def header(f: Frame, t: float, title: str):
    f.text((60, 52), title, 30, weight="demi", p=appear(t, 0.1))
    f.line([(60, 100), (60 + 80 * appear(t, 0.2, 0.6), 100)], ACCENT, width=3)


def wrap(text: str, size: float, max_width: float, f: Frame, mono=True) -> list[tuple[int, str]]:
    """Greedy wrap that keeps the start index of every line in the original text."""
    lines, start = [], 0
    while start < len(text):
        end = len(text)
        while f.width(text[start:end], size, mono=mono) > max_width:
            end = text.rfind(" ", start, end - 1)
        lines.append((start, text[start:end]))
        start = end + 1
    return lines


# --- scenes --------------------------------------------------------------------------------


def bracket_icon(f: Frame, center, scale, p):
    cx, cy = center
    outline = [(-1, 1), (-1, -1), (-0.7, -1), (-0.7, 0.7), (1, 0.7), (1, 1), (-1, 1)]
    points = [(cx + x * scale, cy + y * scale) for x, y in outline]
    lengths = [math.dist(a, b) for a, b in pairwise(points)]
    drawn, remaining = [points[0]], p * sum(lengths)
    for (a, b), length in zip(pairwise(points), lengths, strict=True):
        if remaining <= 0:
            break
        k = min(1.0, remaining / length)
        drawn.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))
        remaining -= length
    f.line(drawn, INK, width=4)
    rib = [(cx - 0.7 * scale, cy - 0.15 * scale), (cx + 0.15 * scale, cy + 0.7 * scale)]
    f.line(rib, ACCENT, width=4, p=appear(p, 0.85, 0.15))
    f.circle(
        (cx - 0.85 * scale, cy - 0.55 * scale),
        0.07 * scale,
        outline=INK,
        width=3,
        p=appear(p, 0.9, 0.1),
    )
    f.circle(
        (cx + 0.6 * scale, cy + 0.85 * scale),
        0.07 * scale,
        outline=INK,
        width=3,
        p=appear(p, 0.9, 0.1),
    )


def scene_title(f: Frame, t: float, L: dict, run: dict):
    bracket_icon(f, (600, 175), 62, ease(t / 1.4))
    f.text((600, 330), "MechAgent", 76, weight="bold", anchor="mm", p=appear(t, 0.5))
    f.text((600, 405), L["subtitle"], 27, weight="medium", anchor="mm", p=appear(t, 0.9))
    f.text((600, 450), L["tagline"], 19, color=MUTED, anchor="mm", p=appear(t, 1.3))


def scene_read(f: Frame, t: float, L: dict, run: dict):
    header(f, t, L["read_title"])
    spec, size = L["spec"], 16
    f.box((60, 130, 590, 520), p=appear(t, 0.2))
    lines = wrap(spec, size, 480, f)
    typed = int(len(spec) * clamp((t - 0.4) / 2.4))
    char = f.width("m", size, mono=True)

    field_times = [3.1 + 0.5 * i for i in range(len(L["fields"]))]
    for (_, _, parts), start in zip(L["fields"], field_times, strict=True):
        p = appear(t, start, 0.3)
        for part in parts:
            index = spec.find(part)
            for line_start, line in lines:
                a, b = max(index, line_start), min(index + len(part), line_start + len(line))
                if a < b and p > 0:
                    x0 = 85 + (a - line_start) * char
                    y0 = 155 + lines.index((line_start, line)) * 30
                    f.box(
                        (x0 - 2, y0 - 3, x0 + (b - a) * char + 2, y0 + 23),
                        fill=ACCENT_SOFT,
                        outline=None,
                        radius=4,
                        p=p,
                        bg=CARD,
                    )
    for row, (line_start, line) in enumerate(lines):
        visible = line[: max(0, typed - line_start)]
        f.text((85, 155 + row * 30), visible, size, mono=True, bg=CARD, rise=0)
    if 0 < typed < len(spec) or (t % 0.8 < 0.4 and typed == len(spec) and t < 3):
        row = max(i for i, (s, _) in enumerate(lines) if s <= max(typed - 1, 0))
        x = 85 + (typed - lines[row][0]) * char
        f.line([(x + 1, 155 + row * 30), (x + 1, 177 + row * 30)], ACCENT, width=2)

    for i, ((label, value, _), start) in enumerate(zip(L["fields"], field_times, strict=True)):
        p = appear(t, start)
        y = 135 + i * 64
        f.box((630, y, 1140, y + 54), p=p)
        f.text((650, y + 8), label, 14, color=MUTED, weight="medium", p=p, bg=CARD, rise=0)
        f.text((650, y + 26), value, 18, weight="demi", p=p, bg=CARD, rise=0)
        f.tick((1115, y + 27), 10, p=appear(t, start + 0.2, 0.3), bg=CARD)
        f.text(
            (1098, y + 27),
            L["found"],
            13,
            color=GREEN,
            anchor="rm",
            p=appear(t, start + 0.2),
            bg=CARD,
            rise=0,
        )
    f.text((60, 560), L["read_note"], 18, color=MUTED, p=appear(t, 6.2))


NODE_KINDS = ["llm", "rule", "rule", "tool", "tool", "tool", "tool", "llm", "tool"]
COLUMNS = [140, 370, 600, 830, 1060]
ROW1, ROW2 = 250, 420
NODE_POS = [(COLUMNS[i], ROW1) for i in range(5)] + [
    (COLUMNS[4], ROW2),
    (COLUMNS[3], ROW2),
    (COLUMNS[2], ROW2),
    (COLUMNS[1], ROW2),
]
LOOP_Y = 500
HOPS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 5), (5, 6), (6, 7), (7, 8)]


def hop_path(a: int, b: int) -> list[tuple[float, float]]:
    if (a, b) == (7, 5):
        return [NODE_POS[7], (NODE_POS[7][0], LOOP_Y), (NODE_POS[5][0], LOOP_Y), NODE_POS[5]]
    return [NODE_POS[a], NODE_POS[b]]


def along(points, k: float):
    lengths = [math.dist(a, b) for a, b in pairwise(points)]
    remaining = k * sum(lengths)
    for (a, b), length in zip(pairwise(points), lengths, strict=True):
        if remaining <= length:
            u = remaining / length
            return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
        remaining -= length
    return points[-1]


def scene_graph(f: Frame, t: float, L: dict, run: dict):
    header(f, t, L["graph_title"])
    hw, hh = 95, 32
    hop_start, hop_time = 1.0, 0.55
    position = (t - hop_start) / hop_time
    hop = min(int(max(position, 0)), len(HOPS))
    visited = {0} | {b for a, b in HOPS[:hop]}
    current = HOPS[hop - 1][1] if hop > 0 else 0

    edges = [((NODE_POS[i][0] + hw, ROW1), (NODE_POS[i + 1][0] - hw, ROW1)) for i in range(4)]
    edges.append(((COLUMNS[4], ROW1 + hh), (COLUMNS[4], ROW2 - hh)))
    edges += [((NODE_POS[i][0] - hw, ROW2), (NODE_POS[i + 1][0] + hw, ROW2)) for i in (5, 6, 7)]
    for i, edge in enumerate(edges):
        f.arrow(list(edge), EDGE, width=2, p=appear(t, 0.2 + 0.06 * i))
    loop = [
        (COLUMNS[2], ROW2 + hh),
        (COLUMNS[2], LOOP_Y),
        (COLUMNS[4], LOOP_Y),
        (COLUMNS[4], ROW2 + hh),
    ]
    f.arrow(loop, EDGE, width=2, p=appear(t, 0.7))
    f.text((715, LOOP_Y + 8), L["loop"], 15, color=MUTED, anchor="ma", p=appear(t, 0.7))

    f.dashed((COLUMNS[1], ROW1 - hh), (COLUMNS[1], 168), MUTED, p=appear(t, 0.6))
    ask_w = f.width(L["ask"], 15) + 30
    f.box(
        (COLUMNS[1] - ask_w / 2, 132, COLUMNS[1] + ask_w / 2, 168),
        outline=MUTED,
        radius=18,
        p=appear(t, 0.6),
    )
    f.text(
        (COLUMNS[1], 150), L["ask"], 15, color=MUTED, anchor="mm", p=appear(t, 0.6), bg=CARD, rise=0
    )

    if 0 <= position < len(HOPS):
        a, b = HOPS[hop]
        f.circle(along(hop_path(a, b), ease(position - hop)), 7, fill=ACCENT)

    for i, ((x, y), label, kind) in enumerate(zip(NODE_POS, L["nodes"], NODE_KINDS, strict=True)):
        p = appear(t, 0.15 + 0.05 * i)
        color, soft = KIND_COLORS[kind]
        lit = i in visited and t > hop_start
        fill = soft if lit else CARD
        f.box(
            (x - hw, y - hh, x + hw, y + hh),
            fill=fill,
            outline=color if lit else LINE,
            width=2.5 if i == current and lit else 1.2,
            p=p,
        )
        f.text((x, y - 6), label, 17, weight="demi", anchor="mm", p=p, bg=fill, rise=0)
        f.text(
            (x, y + 15),
            L["kinds"][kind],
            13,
            color=color,
            weight="demi",
            anchor="mm",
            p=p,
            bg=fill,
            rise=0,
        )

    best = best_by_iteration(run["trials"])
    decisions = [i for i, (_, b) in enumerate(HOPS) if b == 7]
    for n, index in enumerate(decisions, start=1):
        start = hop_start + (index + 1) * hop_time
        end = hop_start + (decisions[n] + 1) * hop_time if n < len(decisions) else math.inf
        p = appear(t, start, 0.3) * (1 - appear(t, end - 0.25, 0.25))
        action = "refine" if n < len(decisions) else "accept"
        text = L["iteration"].format(n=n, mass=best[n], action=L["actions"][action])
        f.text((600, 560), text, 19, weight="medium", color=ACCENT, anchor="ma", p=p)
    f.text((600, 615), L["graph_note"], 17, color=MUTED, anchor="ma", p=appear(t, 1.0))


def scene_split(f: Frame, t: float, L: dict, run: dict):
    header(f, t, L["split_title"])
    p = appear(t, 0.2)
    f.box((60, 135, 520, 560), fill=ACCENT_SOFT, outline=ACCENT, p=p)
    f.text((90, 160), L["llm_card"], 26, weight="demi", color=ACCENT, p=p, bg=ACCENT_SOFT)
    f.text((90, 198), L["llm_model"], 16, color=MUTED, p=p, bg=ACCENT_SOFT)
    y = 255
    for i, item in enumerate(L["llm_items"]):
        q = appear(t, 0.6 + 0.35 * i)
        f.circle((98, y + 14), 4, fill=ACCENT, p=q, bg=ACCENT_SOFT)
        for line in wrap(item, 19, 370, f, mono=False):
            f.text((115, y), line[1], 19, weight="medium", p=q, bg=ACCENT_SOFT)
            y += 27
        y += 26
    for k, line in enumerate(wrap(L["llm_timing"], 16, 400, f, mono=False)):
        f.text((90, 490 + k * 22), line[1], 16, color=MUTED, p=appear(t, 1.6), bg=ACCENT_SOFT)

    p = appear(t, 1.4)
    f.box((560, 135, 1140, 560), fill=BLUE_SOFT, outline=BLUE, p=p)
    f.text((590, 160), L["tools_card"], 26, weight="demi", color=BLUE, p=p, bg=BLUE_SOFT)
    for i, (tool, role) in enumerate(L["tools"]):
        q = appear(t, 1.8 + 0.3 * i)
        y = 225 + i * 52
        f.text((590, y), tool, 19, weight="demi", p=q, bg=BLUE_SOFT)
        f.text((820, y), role, 19, color=INK, p=q, bg=BLUE_SOFT)


def scene_search(f: Frame, t: float, L: dict, run: dict, lang: str):
    header(f, t, L["search_title"])
    trials = run["trials"]
    x0, x1, y0, y1 = 110, 790, 150, 520
    low, high = 170, 460
    n = len(trials)

    def px(i):
        return x0 + 10 + (x1 - x0 - 20) * i / (n - 1)

    def py(mass):
        return y1 - (y1 - y0) * (min(max(mass, low), high) - low) / (high - low)

    p = appear(t, 0.2)
    f.line([(x0, y0), (x0, y1), (x1, y1)], MUTED, width=1.5, p=p)
    for mass in range(200, high + 1, 50):
        f.line([(x0 - 5, py(mass)), (x0, py(mass))], MUTED, width=1.5, p=p)
        f.text((x0 - 10, py(mass)), str(mass), 14, color=MUTED, anchor="rm", p=p, rise=0)
    for i in range(0, n + 1, 10):
        x = px(max(i - 1, 0))
        f.text((x, y1 + 10), str(max(i, 1)), 14, color=MUTED, anchor="ma", p=p, rise=0)
    f.text((x1, y1 + 32), L["trial"], 15, color=MUTED, anchor="ra", p=p)
    f.text((x0, y0 - 18), L["mass"], 15, color=MUTED, anchor="mm", p=p)

    split = next(i for i, trial in enumerate(trials) if trial["iteration"] > 1)
    xs = (px(split - 1) + px(split)) / 2
    shown = int(clamp((t - 0.6) / 5.2) * n + 0.999) if t > 0.6 else 0
    q = 1.0 if shown > split else 0.0
    f.dashed((xs, y0), (xs, y1), ACCENT, p=q)
    f.text((xs + 8, y0 - 4), L["refine"], 14, color=ACCENT, weight="medium", p=q, rise=0)

    best, steps, labels = math.inf, [], {}
    for i, trial in enumerate(trials[:shown]):
        x = px(i)
        if trial["objective"] is None:
            f.cross((x, y1 - 9), 4, MUTED)
            continue
        mass = trial["objective"] * 1000
        if trial["valid"]:
            f.circle((x, py(mass)), 5.5, fill=ACCENT)
            if mass < best:
                best = mass
                labels[trial["iteration"]] = (x, mass)
        else:
            f.circle((x, py(mass)), 5, outline=MUTED, width=1.5)
        if best < math.inf:
            steps.append((x, py(best)))
    if len(steps) > 1:
        path = [steps[0]]
        for x, y in steps[1:]:
            path += [(x, path[-1][1]), (x, y)]
        f.line(path, ACCENT, width=2)
    for x, mass in labels.values():
        f.text(
            (x, py(mass) + 12),
            f"{mass:.0f} g",
            15,
            weight="demi",
            color=ACCENT,
            anchor="ma",
            rise=0,
        )

    legend = [(L["valid"], "valid"), (L["invalid"], "invalid"), (L["infeasible"], "cross")]
    lx = x0
    for text, kind in legend:
        if kind == "valid":
            f.circle((lx + 6, 590), 5.5, fill=ACCENT, p=p)
        elif kind == "invalid":
            f.circle((lx + 6, 590), 5, outline=MUTED, p=p)
        else:
            f.cross((lx + 6, 590), 4, MUTED, p=p)
        f.text((lx + 18, 590), text, 15, color=MUTED, anchor="lm", p=p, rise=0)
        lx += f.width(text, 15) + 50
    others = f" {L['and']} ".join(run["others"])
    note = L["materials_note"].format(material=run["material"], others=others)
    f.text((x0, 622), note, 15, color=MUTED, p=p, rise=0)

    if shown:
        trial = trials[shown - 1]
        bg = CARD
        f.box((830, 150, 1140, 470), p=p)
        f.text((855, 172), f"{L['trial']} {shown}", 26, weight="demi", bg=bg, rise=0)
        f.text((1115, 182), run["material"], 15, color=MUTED, anchor="ra", bg=bg, rise=0)
        for k, (key, label) in enumerate(L["params"]):
            y = 230 + k * 42
            f.text((855, y), label, 16, color=MUTED, bg=bg, rise=0)
            value = number(trial["params"][key], 1, lang) + " mm"
            f.text((1115, y), value, 17, weight="demi", anchor="ra", bg=bg, rise=0)
        if trial["objective"] is None:
            status, color = L["infeasible"], MUTED
        elif trial["valid"]:
            status, color = L["valid"], GREEN
        elif trial["failed_checks"]:
            check = (
                trial["failed_checks"][-1]
                if "max_displacement" not in trial["failed_checks"]
                else "max_displacement"
            )
            status = L["fails"].format(check=L["checks"].get(check, check))
            color = ACCENT
        else:
            status, color = L["rejected"], ACCENT
        if trial["objective"] is not None:
            mass = number(trial["objective"] * 1000, 0, lang)
            f.text((855, 410), f"{mass} g", 22, weight="demi", bg=bg, rise=0)
        f.text((855, 442), status, 15, weight="medium", color=color, bg=bg, rise=0)


def scene_verify(f: Frame, t: float, L: dict, run: dict, lang: str):
    header(f, t, L["verify_title"])
    values = verify_values(run, lang)
    for i, (label, value) in enumerate(L["verify_rows"]):
        start = 0.5 + 0.5 * i
        p = appear(t, start)
        y = 135 + i * 60
        f.box((60, y, 780, y + 50), p=p)
        f.tick((88, y + 25), 12, p=appear(t, start + 0.25, 0.3), bg=CARD)
        f.text((112, y + 25), label, 18, weight="medium", anchor="lm", p=p, bg=CARD, rise=0)
        f.text(
            (760, y + 25),
            value.format(**values),
            18,
            weight="demi",
            anchor="rm",
            p=p,
            bg=CARD,
            rise=0,
        )

    p = appear(t, 3.9, 0.35)
    scale = 1.25 - 0.25 * p
    cx, cy = 965, 300
    w, h = 150 * scale, 85 * scale
    f.box((cx - w, cy - h, cx + w, cy + h), fill=GREEN_SOFT, outline=GREEN, width=3, radius=16, p=p)
    f.text(
        (cx, cy - 22 * scale),
        L["verified"],
        38 * scale,
        weight="bold",
        color=GREEN,
        anchor="mm",
        p=p,
        bg=GREEN_SOFT,
        rise=0,
    )
    mass = number(run["mass"], 1, lang)
    f.text(
        (cx, cy + 30 * scale),
        f"{run['material']}  ·  {mass} g",
        21 * scale,
        weight="demi",
        anchor="mm",
        p=p,
        bg=GREEN_SOFT,
        rise=0,
    )
    f.text((60, 530), L["verify_note"], 18, color=MUTED, p=appear(t, 4.3))


def scene_files(f: Frame, t: float, L: dict, run: dict, figure: Image.Image):
    header(f, t, L["files_title"])
    for i, (name, description) in enumerate(L["files"]):
        p = appear(t, 0.4 + 0.4 * i)
        y = 135 + i * 110
        f.box((60, y, 565, y + 96), p=p)
        f.box((80, y + 24, 112, y + 68), fill=ACCENT_SOFT, outline=ACCENT, radius=5, p=p, bg=CARD)
        for k in range(3):
            f.line(
                [(88, y + 37 + 9 * k), (104 - 6 * (k == 2), y + 37 + 9 * k)],
                ACCENT,
                1.5,
                p=p,
                bg=ACCENT_SOFT,
            )
        f.text((132, y + 16), name, 21, weight="demi", mono=True, p=p, bg=CARD, rise=0)
        for k, line in enumerate(wrap(description, 16, 410, f, mono=False)):
            f.text((132, y + 48 + k * 21), line[1], 16, color=MUTED, p=p, bg=CARD, rise=0)
    f.text((60, 490), L["files_note"], 18, color=MUTED, p=appear(t, 1.8))

    p = appear(t, 0.9, 0.6)
    f.box((600, 130, 1140, 600), fill=(255, 255, 255), p=p)
    f.paste(figure, (610, 140), p)
    f.text((870, 620), L["figure"], 15, color=MUTED, anchor="ma", p=p)


def scene_end(f: Frame, t: float, L: dict, run: dict, lang: str):
    bracket_icon(f, (600, 160), 42, 1.0 if t > 0 else 0)
    mass = number(run["mass"], 1, lang)
    f.text(
        (600, 295),
        f"{run['material']}  ·  {mass} g",
        60,
        weight="bold",
        anchor="mm",
        p=appear(t, 0.2),
    )
    text = L["result"].format(iterations=run["iterations"], trials=run["total_trials"])
    f.text((600, 365), text, 24, weight="medium", color=GREEN, anchor="mm", p=appear(t, 0.6))
    f.line([(560, 430), (640, 430)], ACCENT, width=3, p=appear(t, 1.0))
    f.text((600, 480), "MechAgent", 34, weight="bold", anchor="mm", p=appear(t, 1.2))
    f.text((600, 520), L["tagline"], 18, color=MUTED, anchor="mm", p=appear(t, 1.4))


# --- rendering -----------------------------------------------------------------------------


def scenes(L: dict, run: dict, figure: Image.Image, lang: str):
    return [
        (3.6, lambda f, t: scene_title(f, t, L, run)),
        (8.2, lambda f, t: scene_read(f, t, L, run)),
        (8.4, lambda f, t: scene_graph(f, t, L, run)),
        (5.4, lambda f, t: scene_split(f, t, L, run)),
        (8.0, lambda f, t: scene_search(f, t, L, run, lang)),
        (6.4, lambda f, t: scene_verify(f, t, L, run, lang)),
        (5.4, lambda f, t: scene_files(f, t, L, run, figure)),
        (4.4, lambda f, t: scene_end(f, t, L, run, lang)),
    ]


def render(lang: str, run: dict, figure: Image.Image) -> list[Image.Image]:
    L = TEXT[lang]
    background = Image.new("RGB", (W, H), BG)
    frames = []
    for duration, draw in scenes(L, run, figure, lang):
        for k in range(round(duration * FPS)):
            t = k / FPS
            frame = Frame()
            draw(frame, t)
            image = frame.result()
            fade = min(ease(t / FADE), ease((duration - t) / FADE))
            frames.append(Image.blend(background, image, fade) if fade < 1 else image)
    return frames


def to_palette(frame: Image.Image, palette: list[int]) -> Image.Image:
    """Exact nearest colour mapping: Pillow's own lookup rounds colours and shifts flat ones."""
    colours = np.array(palette, dtype=np.int32).reshape(-1, 3)
    pixels = np.asarray(frame, dtype=np.int32).reshape(-1, 3)
    packed, inverse = np.unique(pixels @ [65536, 256, 1], return_inverse=True)
    unique = np.stack([packed >> 16, (packed >> 8) & 255, packed & 255], axis=1)
    distance = ((unique[:, None, :] - colours[None, :, :]) ** 2).sum(axis=2)
    indices = distance.argmin(axis=1).astype(np.uint8)[inverse.ravel()]
    image = Image.fromarray(indices.reshape(frame.height, frame.width), mode="P")
    image.putpalette(palette)
    return image


def save_gif(frames: list[Image.Image], path: Path, figure: Image.Image):
    # one palette for the whole GIF, so flat colours never flicker: the interface colours and
    # the colour map of the figure are quantised apart, so neither eats the other's colours
    sample = Image.new("RGB", (W, H * 8))
    for i, frame in enumerate(frames[:: max(1, len(frames) // 8)][:8]):
        sample.paste(frame, (0, i * H))
    exact = [
        BG,
        CARD,
        INK,
        MUTED,
        LINE,
        EDGE,
        ACCENT,
        ACCENT_SOFT,
        GREEN,
        GREEN_SOFT,
        BLUE,
        BLUE_SOFT,
        RULE,
        RULE_SOFT,
        (255, 255, 255),
    ]
    interface = sample.quantize(colors=145, method=Image.Quantize.MEDIANCUT).getpalette()
    colour_map = figure.quantize(colors=95, method=Image.Quantize.MEDIANCUT).getpalette()
    palette = [v for c in exact for v in c] + interface[: 145 * 3] + colour_map[: 95 * 3]
    indexed = [to_palette(frame, palette) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        path,
        save_all=True,
        append_images=indexed[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=1,
    )


def load_figure(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    left, top, right, bottom = 320, 70, 1150, 870
    crop = image.crop((left, top, min(right, image.width), min(bottom, image.height)))
    crop.thumbnail((520, 450), Image.LANCZOS)
    return crop


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="agent run folder with report.md and trials")
    parser.add_argument("--figure", type=Path, required=True, help="von Mises figure of a run")
    parser.add_argument("--out", type=Path, default=Path("docs/media"))
    parser.add_argument("--lang", nargs="+", default=["en", "fr"])
    args = parser.parse_args()

    run = load_run(args.run)
    figure = load_figure(args.figure)
    for lang in args.lang:
        frames = render(lang, run, figure)
        path = args.out / f"mechagent_explainer_{lang}.gif"
        save_gif(frames, path, figure)
        print(f"{path}: {len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
