#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate that canonical reference points into ../kugelaudio-open are documented."""

from __future__ import annotations

from pathlib import Path


REQUIRED = {
    "README.md": [
        "Canonical reference points into `../kugelaudio-open`",
        "../kugelaudio-open/src/kugelaudio_open/processors/kugelaudio_processor.py",
        "../kugelaudio-open/src/kugelaudio_open/models/kugelaudio_inference.py",
        "../kugelaudio-open/src/kugelaudio_open/utils/generation.py",
        "../kugelaudio-open/src/kugelaudio_open/processors/audio_processor.py",
        "../kugelaudio-open/src/kugelaudio_open/cli.py",
    ],
    "docs/conversion.md": [
        "Canonical reference points for this converter/runtime contract",
        "../kugelaudio-open/src/kugelaudio_open/configs/kugelaudio_1.5b.json",
        "../kugelaudio-open/src/kugelaudio_open/configs/model_config.py",
        "../kugelaudio-open/src/kugelaudio_open/processors/kugelaudio_processor.py",
        "../kugelaudio-open/src/kugelaudio_open/models/kugelaudio_inference.py",
        "../kugelaudio-open/src/kugelaudio_open/utils/generation.py",
        "../kugelaudio-open/src/kugelaudio_open/processors/audio_processor.py",
    ],
}


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    for rel, needles in REQUIRED.items():
        text = (repo / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                raise SystemExit(f"FAIL: {rel} missing canonical-reference text: {needle}")
    print("Canonical references OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
