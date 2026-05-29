#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Unit test for acceptance threshold logic in the divergence harness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_kugelaudio_divergence import check_threshold


def main() -> int:
    r, ok = check_threshold({
        "canonical": {"transcript_recall": 1.0, "speaker_similarity": 0.90},
        "ggml": {"transcript_recall": 0.96, "speaker_similarity": 0.86},
    })
    assert r is True, f"expected pass, got {r}: {ok}"
    assert "PASS" in ok, ok

    r, ok = check_threshold({
        "canonical": {"transcript_recall": 1.0, "speaker_similarity": 0.80},
        "ggml": {"transcript_recall": 0.95, "speaker_similarity": 0.76},
    })
    assert r is True, f"expected pass at ratio boundary, got {r}: {ok}"

    r, ok = check_threshold({
        "canonical": {"transcript_recall": 0.80, "speaker_similarity": 0.60},
        "ggml": {"transcript_recall": 0.80, "speaker_similarity": 0.60},
    })
    assert r is True, f"expected pass at floor boundary, got {r}: {ok}"

    r, ok = check_threshold({
        "canonical": {"transcript_recall": 1.0, "speaker_similarity": 0.90},
        "ggml": {"transcript_recall": 0.94, "speaker_similarity": 0.86},
    })
    assert r is False, f"expected transcript ratio fail, got {r}"
    assert "transcript_recall" in ok and "ratio" in ok.lower(), ok

    r, ok = check_threshold({
        "canonical": {"transcript_recall": 1.0, "speaker_similarity": 0.70},
        "ggml": {"transcript_recall": 0.98, "speaker_similarity": 0.59},
    })
    assert r is False, f"expected speaker floor fail, got {r}"
    assert "speaker_similarity" in ok and "floor" in ok.lower(), ok

    r, ok = check_threshold({
        "canonical": {"transcript_recall": None, "speaker_similarity": 0.70},
        "ggml": {"transcript_recall": 0.9, "speaker_similarity": 0.70},
    })
    assert r is False and "missing" in ok.lower(), ok

    r, ok = check_threshold({
        "canonical": {"transcript_recall": 0.0, "speaker_similarity": 0.70},
        "ggml": {"transcript_recall": 0.0, "speaker_similarity": 0.70},
    })
    assert r is False and "zero" in ok.lower(), ok

    print("Threshold logic OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
