#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate that deferred-feature seams remain identified in code/docs."""

from __future__ import annotations

from pathlib import Path


REQUIRED = {
    "docs/kugelaudio-parity.md": [
        "Deferred-feature seams for post-v1 work",
        "Request-shape seam",
        "Prompt-builder seam",
        "Eval / quantization seam",
        "Deferred-scope boundary",
        "KugelAudioRequestPolicy",
        "build_prompt_15b_legacy",
        "build_kugelaudio_prompt_single_speaker",
        "ggml_model_q8_0",
    ],
    "src/vibevoice_tts.hpp": [
        "Policy seam for deferred KugelAudio features",
        "KugelAudioRequestPolicy",
        "kugelaudio_v1_request_policy",
        "validate_kugelaudio_request",
    ],
    "scripts/eval_kugelaudio_divergence.py": [
        "Eval-surface seam",
        "ggml_model_q8_0",
    ],
}


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    for rel, needles in REQUIRED.items():
        text = (repo / rel).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                raise SystemExit(f"FAIL: {rel} missing seam marker: {needle}")
    print("Deferred-feature seams OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
