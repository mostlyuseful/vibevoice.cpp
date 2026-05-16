#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate that legacy VibeVoice paths are clearly marked off the KugelAudio v1 path."""

from __future__ import annotations

from pathlib import Path


REQUIRED_SUBSTRINGS = {
    "README.md": [
        "KugelAudio-first",
        "Legacy VibeVoice surfaces remain in the repo",
        "legacy VibeVoice examples, not KugelAudio v1 acceptance",
        "legacy VibeVoice 1.5B path",
    ],
    "docs/conversion.md": [
        "legacy VibeVoice-only path",
        "not part of", 
        "not used by KugelAudio v1",
    ],
    "examples/cli/main.cpp": [
        "legacy realtime-0.5B",
        "not part of KugelAudio v1",
    ],
}


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    for rel, needles in REQUIRED_SUBSTRINGS.items():
        text = (repo / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                raise SystemExit(f"FAIL: {rel} missing legacy-marking text: {needle}")
    print("Legacy path marking OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
