#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate that default eval logs do not dump raw prompt/audio contents."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_kugelaudio_divergence import summarize_reference_for_log, summarize_text_for_log


def main() -> int:
    text = "Highly sensitive private prompt contents"
    text_summary = summarize_text_for_log(text)
    if text in text_summary:
        raise SystemExit("FAIL: text summary leaked raw prompt contents")
    if "chars=" not in text_summary or "sha256=" not in text_summary:
        raise SystemExit(f"FAIL: text summary missing expected fields: {text_summary}")

    path_value = "/very/private/audio/customer_name_reference.wav"
    ref_summary = summarize_reference_for_log(path_value, "1234567890abcdef1234567890abcdef")
    if path_value in ref_summary or "customer_name_reference.wav" in ref_summary:
        raise SystemExit("FAIL: reference summary leaked raw audio path/name")
    if "sha256=1234567890abcdef" not in ref_summary:
        raise SystemExit(f"FAIL: reference summary missing truncated sha256: {ref_summary}")
    if "present=yes" not in ref_summary:
        raise SystemExit(f"FAIL: reference summary missing presence marker: {ref_summary}")

    print("Log redaction OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
