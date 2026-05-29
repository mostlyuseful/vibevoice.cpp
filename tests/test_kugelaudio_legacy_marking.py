#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate that removed legacy operator workflows stay off the published path."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent

    readme = (repo / "README.md").read_text(encoding="utf-8")
    if "## Removed legacy operator workflows" not in readme:
        raise SystemExit("FAIL: README missing removed-legacy-workflows note")
    forbidden = [
        "## Quickstart - raw-reference voice cloning (legacy VibeVoice path, not KugelAudio v1 acceptance)",
        "## Quickstart - ASR",
        "legacy VibeVoice examples, not KugelAudio v1 acceptance",
    ]
    for needle in forbidden:
        if needle in readme:
            raise SystemExit(f"FAIL: README still exposes removed legacy operator workflow: {needle}")

    conv = (repo / "docs" / "conversion.md").read_text(encoding="utf-8")
    required = [
        "Only `kugelaudio/kugelaudio-0-open` is part of the published acceptance path",
        "legacy/internal compatibility notes",
    ]
    for needle in required:
        if needle not in conv:
            raise SystemExit(f"FAIL: docs/conversion.md missing legacy-quarantine note: {needle}")

    cli = (repo / "examples" / "cli" / "main.cpp").read_text(encoding="utf-8")
    if "legacy realtime-0.5B" in cli or "not part of KugelAudio v1" in cli:
        raise SystemExit("FAIL: published CLI still carries legacy operator guidance")

    print("Legacy operator workflow removal OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
