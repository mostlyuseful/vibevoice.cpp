#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate that deferred V2 features are not advertised as active critical-path support."""

from __future__ import annotations

from pathlib import Path


REQUIRED = {
    "src/vibevoice_tts.hpp": [
        "legacy raw-reference compatibility path supports one reference WAV",
        "KugelAudio v1 acceptance is narrower by design",
        "detail::KugelAudioRequestPolicy",
        "not KugelAudio v1 acceptance",
        "wider shapes remain deferred behind the",
    ],
    "README.md": [
        "single-speaker only",
        "conditioning: exactly one raw reference WAV",
        "## Removed legacy operator workflows",
    ],
}


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    for rel, needles in REQUIRED.items():
        text = (repo / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                raise SystemExit(f"FAIL: {rel} missing critical-path non-feature marker: {needle}")
    print("Critical-path non-features OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
