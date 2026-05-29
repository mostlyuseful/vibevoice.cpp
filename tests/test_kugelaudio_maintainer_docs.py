#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate maintainer docs reflect the KugelAudio-first repo reality."""

from __future__ import annotations

from pathlib import Path


REQUIRED_SUBSTRINGS = {
    "AGENTS.md": [
        "Start here for the KugelAudio v1 acceptance path",
        "scripts/eval_kugelaudio_divergence.py",
        "tests/fixtures/kugelaudio_eval_config.json",
        "docs/kugelaudio-parity.md",
        "treat it as legacy unless it",
        "part of the KugelAudio v1 acceptance path",
        "results.json + per-step logs",
    ],
}


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    for rel, needles in REQUIRED_SUBSTRINGS.items():
        text = (repo / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                raise SystemExit(f"FAIL: {rel} missing maintainer-doc text: {needle}")
    print("Maintainer docs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
