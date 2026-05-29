#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate that KugelAudio error messages are clear and actionable."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "scripts"))
    from eval_kugelaudio_divergence import check_threshold

    _, msg = check_threshold({
        "canonical": {"transcript_recall": 1.0, "speaker_similarity": 0.90},
        "ggml": {"transcript_recall": 0.50, "speaker_similarity": 0.50},
    })
    if "transcript_recall" not in msg or "speaker_similarity" not in msg or "ratio" not in msg.lower() or "floor" not in msg.lower():
        raise SystemExit(f"FAIL: threshold fail message missing metric/boundary names: {msg}")

    _, msg = check_threshold({
        "canonical": {"transcript_recall": 1.0, "speaker_similarity": 0.90},
        "ggml": {"transcript_recall": 0.94, "speaker_similarity": 0.90},
    })
    if "transcript_recall" not in msg or "ratio" not in msg.lower():
        raise SystemExit(f"FAIL: transcript ratio-fail message missing context: {msg}")

    _, msg = check_threshold({
        "canonical": {"transcript_recall": 1.0, "speaker_similarity": 0.70},
        "ggml": {"transcript_recall": 1.0, "speaker_similarity": 0.59},
    })
    if "speaker_similarity" not in msg or "floor" not in msg.lower():
        raise SystemExit(f"FAIL: speaker floor-fail message missing context: {msg}")

    print("Error clarity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
