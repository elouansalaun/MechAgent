"""PDF version of the Markdown report, with ReportLab.

We only convert the Markdown our own report uses: headings, tables, lists, quotes, bold text
and images. Keeping one source (the Markdown) means the PDF can never say something else.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_WIDTH = A4[0] - 30 * mm
_IMAGE = re.compile(r"^!\[(?P<caption>[^\]]*)\]\((?P<path>[^)]+)\)$")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": base["Title"],
        "h1": base["Heading1"],
        "h2": base["Heading2"],
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9, leading=12),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=7.5, leading=9),
        "quote": ParagraphStyle(
            "quote", parent=base["BodyText"], fontSize=9, leftIndent=10, textColor=colors.grey
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["BodyText"], fontSize=8, textColor=colors.grey
        ),
    }


def inline(text: str) -> str:
    """Markdown bold to ReportLab markup, everything else escaped."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text))


def _table(rows: list[list[str]], styles) -> Table:
    widths = [max(len(row[i]) if i < len(row) else 0 for row in rows) for i in range(len(rows[0]))]
    widths = [min(max(w, 4), 60) for w in widths]
    scale = PAGE_WIDTH / sum(widths)
    data = [[Paragraph(inline(cell), styles["cell"]) for cell in row] for row in rows]
    table = Table(data, colWidths=[w * scale for w in widths], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde3ea")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9aa5b1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_to_flowables(markdown: str, base_dir: Path) -> list:
    styles = _styles()
    flowables: list = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.fullmatch(r"\|(\s*-+\s*\|)+", lines[i].strip()):
                    rows.append(_split_row(lines[i]))
                i += 1
            flowables += [_table(rows, styles), Spacer(1, 4)]
            continue
        image = _IMAGE.match(line.strip())
        if image:
            path = base_dir / image["path"]
            picture = Image(str(path))
            ratio = picture.imageHeight / picture.imageWidth
            picture.drawWidth, picture.drawHeight = PAGE_WIDTH * 0.8, PAGE_WIDTH * 0.8 * ratio
            caption = Paragraph(inline(image["caption"]), styles["caption"])
            flowables.append(KeepTogether([picture, caption, Spacer(1, 6)]))
        elif line.startswith("# "):
            flowables.append(Paragraph(inline(line[2:]), styles["title"]))
        elif line.startswith("## "):
            flowables.append(Paragraph(inline(line[3:]), styles["h1"]))
        elif line.startswith("### "):
            flowables.append(Paragraph(inline(line[4:]), styles["h2"]))
        elif line.startswith("> "):
            flowables.append(Paragraph(inline(line[2:]), styles["quote"]))
        elif line.startswith("- "):
            flowables.append(Paragraph(inline(line[2:]), styles["body"], bulletText="•"))
        elif line.strip():
            flowables.append(Paragraph(inline(line), styles["body"]))
        i += 1
    return flowables


def markdown_to_pdf(markdown: str, path: Path, base_dir: Path | None = None) -> Path:
    """Write the report as PDF. Image paths are relative to `base_dir` (the report folder)."""
    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title="MechAgent engineering report",
    )
    document.build(markdown_to_flowables(markdown, base_dir or path.parent))
    return path
