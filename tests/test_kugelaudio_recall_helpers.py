#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Unit test for transcript normalization and recall helpers in the divergence harness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_kugelaudio_divergence import extract_content, normalize_transcript_text, word_set, compute_recall


def main() -> int:
    raw1 = '{"Content":"Hello world"}'
    assert extract_content(raw1) == "Hello world", f"FAIL: {extract_content(raw1)}"
    assert normalize_transcript_text(raw1) == "Hello world", f"FAIL: {normalize_transcript_text(raw1)}"

    raw2 = '{"Start":"0.0","Content":"first"}{"Content":"second"}'
    assert extract_content(raw2) == "first second", f"FAIL: {extract_content(raw2)}"
    assert normalize_transcript_text(raw2) == "first second", f"FAIL: {normalize_transcript_text(raw2)}"

    raw3 = ' Hello world this is a test '
    assert normalize_transcript_text(raw3) == "Hello world this is a test", f"FAIL: plain transcript {normalize_transcript_text(raw3)}"

    assert word_set("Hello, world!") == {"hello", "world"}
    assert word_set("") == set()

    recall = compute_recall("Hello world.", "Hello world")
    assert recall == 1.0, f"FAIL: perfect recall = {recall}"

    recall = compute_recall("Hello world.", "Hello")
    assert recall == 0.5, f"FAIL: half recall = {recall}"

    recall = compute_recall("Hello world.", '{"Other":"Goodbye"}')
    assert recall == 0.0, f"FAIL: zero recall = {recall}"

    print("Recall helpers OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
