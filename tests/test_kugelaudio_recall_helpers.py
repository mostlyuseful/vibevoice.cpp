#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Unit test for recall computation helpers in the divergence harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import the harness helpers via a small import trick
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_kugelaudio_divergence import extract_content, word_set, compute_recall


def main() -> int:
    # Test extract_content against realistic ASR JSON output shapes
    # The C++ ASR path outputs raw text with embedded JSON fields.
    raw1 = '{"Content":"Hello world"}'
    assert extract_content(raw1) == "Hello world", f"FAIL: {extract_content(raw1)}"

    raw2 = '{"Start":"0.0","Content":"first"}{"Content":"second"}'
    assert extract_content(raw2) == "first second", f"FAIL: {extract_content(raw2)}"

    # Multi-segment ASR output (the C++ ASR may emit multiple JSON objects for long audio)
    raw3 = '{"Start":"0.00","End":"1.50","Speaker ID":"0","Content":"Hello world this is"}{"Start":"1.50","End":"3.00","Speaker ID":"0","Content":"a test of voice cloning"}'
    assert extract_content(raw3) == "Hello world this is a test of voice cloning", f"FAIL: multi-segment {extract_content(raw3)}"

    # Empty content field
    raw4 = '{"Content":""}'
    assert extract_content(raw4) == "", f"FAIL: empty content {extract_content(raw4)}"

    # Mixed JSON with extra keys between Content fields
    raw5 = '{"Start":"0.0","Content":"alpha","Speaker ID":"0"}{"Start":"1.0","Content":"beta"}'
    assert extract_content(raw5) == "alpha beta", f"FAIL: mixed keys {extract_content(raw5)}"

    # Test word_set
    assert word_set("Hello, world!") == {"hello", "world"}
    assert word_set("") == set()

    # Test compute_recall
    recall = compute_recall("Hello world.", '{"Content":"Hello world"}')
    assert recall == 1.0, f"FAIL: perfect recall = {recall}"

    recall = compute_recall("Hello world.", '{"Content":"Hello"}')
    assert recall == 0.5, f"FAIL: half recall = {recall}"

    recall = compute_recall("Hello world.", '{"Other":"Goodbye"}')
    assert recall == 0.0, f"FAIL: zero recall = {recall}"

    print("Recall helpers OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
