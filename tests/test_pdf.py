import numpy as np
from PIL import Image

from mechagent.reporting.pdf import inline, markdown_to_flowables, markdown_to_pdf

MARKDOWN = """# Report

**Status: VERIFIED**: fine

## 1. Section

> quoted <text> & more

| Check | Value |
|---|---|
| fatigue_damage | 0.0004 |
| max_displacement < 0.5 | **PASS** |

- first point
- second point

![A figure](figure.png)
"""


def test_inline_markup_is_escaped_and_bold():
    assert inline("a < b & **c**") == "a &lt; b &amp; <b>c</b>"


def test_markdown_becomes_a_pdf(tmp_path):
    Image.fromarray(np.full((40, 60, 3), 200, dtype=np.uint8)).save(tmp_path / "figure.png")

    path = markdown_to_pdf(MARKDOWN, tmp_path / "report.pdf")

    assert path.read_bytes()[:5] == b"%PDF-"
    flowables = markdown_to_flowables(MARKDOWN, tmp_path)
    kinds = [type(f).__name__ for f in flowables]
    assert kinds.count("Table") == 1 and "KeepTogether" in kinds
