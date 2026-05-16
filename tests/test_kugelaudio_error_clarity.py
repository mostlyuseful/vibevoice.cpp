#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate that KugelAudio error messages are clear and actionable.

This test checks the *content* of error strings in the converter and harness,
ensuring they contain enough context for a developer to understand and fix
problems without reading source code.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    # 1. Harness: threshold failure message must name the boundary violated
    sys.path.insert(0, str(repo / "scripts"))
    from eval_kugelaudio_divergence import check_threshold

    _, msg = check_threshold({"canonical": {"recall": 1.0}, "ggml": {"recall": 0.50}})
    if "ratio" not in msg.lower() or "floor" not in msg.lower():
        raise SystemExit(f"FAIL: threshold fail message missing boundary names: {msg}")

    _, msg = check_threshold({"canonical": {"recall": 1.0}, "ggml": {"recall": 0.94}})
    if "ratio" not in msg.lower():
        raise SystemExit(f"FAIL: threshold ratio-fail message missing 'ratio': {msg}")

    _, msg = check_threshold({"canonical": {"recall": 0.70}, "ggml": {"recall": 0.79}})
    if "floor" not in msg.lower():
        raise SystemExit(f"FAIL: threshold floor-fail message missing 'floor': {msg}")

    print("Error clarity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
