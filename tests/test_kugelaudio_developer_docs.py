#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate developer-facing docs reflect the current KugelAudio v1 scope."""

from __future__ import annotations

from pathlib import Path


REQUIRED = {
    "README.md": [
        "KugelAudio v1 acceptance path",
        "Supported v1 scope",
        "checkpoint: `kugelaudio/kugelaudio-0-open`",
        "conditioning: exactly one raw reference WAV",
        "quality target: `f16` transcript recall >= 95% of canonical recall",
        "speaker similarity >= 95% of canonical with",
        "scripts/eval_kugelaudio_divergence.py",
    ],
    "docs/conversion.md": [
        "Converting KugelAudio models to GGUF",
        "KugelAudio v1 acceptance path",
        "Current supported KugelAudio v1 scope",
        "checkpoint: `kugelaudio/kugelaudio-0-open`",
        "conditioning: exactly one raw reference WAV",
        "quality target: `f16` transcript recall >= 95% of canonical with 0.80 floor",
        "speaker similarity >= 95% of canonical with 0.60 floor",
        "execution target: `q8_0` runs end-to-end on the same path",
    ],
}


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    for rel, needles in REQUIRED.items():
        text = (repo / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                raise SystemExit(f"FAIL: {rel} missing developer-doc text: {needle}")
    print("Developer docs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
