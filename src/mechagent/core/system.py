"""Operating system helpers."""

from __future__ import annotations

import os
import shutil
import subprocess


def keep_awake() -> subprocess.Popen | None:
    """Stop macOS from sleeping while this process runs.

    A sleeping Mac pauses the monotonic clock behind subprocess timeouts: during one agent
    run a solve "lasted" 66 minutes with a 600 s timeout. caffeinate exits with our process.
    """
    caffeinate = shutil.which("caffeinate")
    if caffeinate is None:
        return None
    return subprocess.Popen([caffeinate, "-i", "-w", str(os.getpid())])
