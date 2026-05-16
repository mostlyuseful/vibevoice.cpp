#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate that KugelAudio acceptance scripts/tests do not depend on dropped features."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


DROPPED_CONFIG_KEYS = {
    "voice",
    "voice_path",
    "voice_cache",
    "voice_cache_path",
    "language",
    "language_hint",
    "speaker_map",
    "speaker_refs",
    "multi_speaker",
}


def count_flag(argv: list[str], flag: str) -> int:
    return sum(1 for item in argv if item == flag)


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    cfg_path = repo / "tests" / "fixtures" / "kugelaudio_eval_config.json"
    script = repo / "scripts" / "eval_kugelaudio_divergence.py"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    bad_keys = sorted(k for k in cfg.keys() if k in DROPPED_CONFIG_KEYS)
    if bad_keys:
        raise SystemExit(f"FAIL: acceptance config must not include dropped-feature keys: {bad_keys}")

    text = cfg["text"]
    if "Speaker 0:" in text or "Speaker 1:" in text:
        raise SystemExit("FAIL: acceptance fixture text must remain plain untagged text")

    proc = subprocess.run(
        [
            "uv",
            "run",
            str(script),
            "--config",
            str(cfg_path),
            "--allow-missing-artifacts",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"FAIL: eval harness plan rc={proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")

    plan = json.loads(proc.stdout)
    ggml_cmd = plan["ggml"]["command"]

    if "--voice" in ggml_cmd:
        raise SystemExit("FAIL: acceptance ggml command must not use legacy --voice conditioning")
    if count_flag(ggml_cmd, "--ref-audio") != 1:
        raise SystemExit("FAIL: acceptance ggml command must use exactly one --ref-audio")
    if count_flag(ggml_cmd, "--text") != 1:
        raise SystemExit("FAIL: acceptance ggml command must use inline plain text exactly once")
    if any(flag in ggml_cmd for flag in ["--language", "--language-hint"]):
        raise SystemExit("FAIL: acceptance ggml command must not use dropped language-hint features")

    canonical_inline = plan["canonical"]["command"][4]
    if "Speaker 0:" in canonical_inline or "Speaker 1:" in canonical_inline:
        raise SystemExit("FAIL: canonical acceptance command must not inject speaker-tagged dialog")
    if "voice_cache" in canonical_inline:
        raise SystemExit("FAIL: canonical acceptance command must not depend on voice_cache")

    q8_path = cfg.get("ggml_model_q8_0", "")
    if not q8_path or "q8_0" not in q8_path:
        raise SystemExit("FAIL: acceptance config must keep an explicit q8_0 artifact slot")

    print("Acceptance surface OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
