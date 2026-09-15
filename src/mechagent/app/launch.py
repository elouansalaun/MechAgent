"""Console entry point: `mechagent-app` starts the Streamlit interface."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def main() -> None:
    app = Path(__file__).with_name("streamlit_app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())
