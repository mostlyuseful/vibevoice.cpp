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
    # Pass: ggml >= 95% of canonical and >= 0.80 floor
    r, ok = check_threshold({"canonical": {"recall": 1.0}, "ggml": {"recall": 0.96}})
    assert r is True, f"expected pass, got {r}: {ok}"
    assert "PASS" in ok, ok

    # Pass: exact boundary 95%
    r, ok = check_threshold({"canonical": {"recall": 1.0}, "ggml": {"recall": 0.95}})
    assert r is True, f"expected pass at 95%, got {r}: {ok}"

    # Pass: exact boundary floor 0.80
    r, ok = check_threshold({"canonical": {"recall": 0.80}, "ggml": {"recall": 0.80}})
    assert r is True, f"expected pass at floor, got {r}: {ok}"

    # Fail: below ratio
    r, ok = check_threshold({"canonical": {"recall": 1.0}, "ggml": {"recall": 0.94}})
    assert r is False, f"expected fail below ratio, got {r}"
    assert "ratio" in ok.lower(), ok

    # Fail: below floor
    r, ok = check_threshold({"canonical": {"recall": 0.70}, "ggml": {"recall": 0.79}})
    assert r is False, f"expected fail below floor, got {r}"
    assert "floor" in ok.lower(), ok

    # Fail: both reasons
    r, ok = check_threshold({"canonical": {"recall": 1.0}, "ggml": {"recall": 0.50}})
    assert r is False
    assert "ratio" in ok.lower() and "floor" in ok.lower(), ok

    # Missing recall
    r, ok = check_threshold({"canonical": {"recall": None}, "ggml": {"recall": 0.9}})
    assert r is False and "missing" in ok.lower(), ok

    # Zero canonical recall
    r, ok = check_threshold({"canonical": {"recall": 0.0}, "ggml": {"recall": 0.0}})
    assert r is False and "zero" in ok.lower(), ok

    print("Threshold logic OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
