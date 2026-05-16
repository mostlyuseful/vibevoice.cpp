#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Validate the reproducible reference fixture and its eval config wiring."""

from __future__ import annotations

import json
import subprocess
import struct
import wave
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    fixture_dir = repo / "tests" / "fixtures"
    fixture = fixture_dir / "reference_sine.wav"
    config = fixture_dir / "kugelaudio_eval_config.json"

    if not fixture.exists():
        raise SystemExit(f"FAIL: fixture not found: {fixture}")
    if not config.exists():
        raise SystemExit(f"FAIL: eval config not found: {config}")

    # Validate fixture is well-formed 24 kHz mono 16-bit PCM
    with wave.open(str(fixture), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 24000:
            raise SystemExit(
                f"FAIL: fixture properties mismatch: "
                f"channels={wf.getnchannels()} width={wf.getsampwidth()} sr={wf.getframerate()}"
            )
        n = wf.getnframes()
        if n != 3 * 24000:
            raise SystemExit(f"FAIL: fixture duration mismatch, expected 72000 frames, got {n}")
        data = wf.readframes(n)
    # Parse one sample to verify PCM structure
    if len(data) < 2:
        raise SystemExit("FAIL: fixture data is empty")
    first_sample = struct.unpack("<h", data[:2])[0]
    if not (-32768 <= first_sample <= 32767):
        raise SystemExit(f"FAIL: first sample out of int16 range: {first_sample}")

    # Validate eval config references this fixture
    cfg = json.loads(config.read_text(encoding="utf-8"))
    ref_path_value = cfg.get("reference_audio")
    if not ref_path_value:
        raise SystemExit("FAIL: eval config missing reference_audio key")
    resolved = (fixture_dir / ref_path_value).resolve()
    if resolved != fixture.resolve():
        raise SystemExit(f"FAIL: eval config resolves to {resolved}, expected {fixture.resolve()}")

    # Validate the harness can resolve this config in plan mode
    script = repo / "scripts" / "eval_kugelaudio_divergence.py"
    proc = subprocess.run(
        [
            "uv", "run", str(script),
            "--config", str(config),
            "--allow-missing-artifacts",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"FAIL: eval plan script rc={proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )
    try:
        plan = json.loads(proc.stdout)
    except Exception as e:
        raise SystemExit(f"FAIL: stdout was not valid JSON plan: {e}\n{proc.stdout}")

    shared_ref = plan.get("shared_run", {}).get("reference_audio", {})
    if shared_ref.get("path") != str(fixture.resolve()):
        raise SystemExit("FAIL: plan did not resolve to fixture path")

    voice = plan.get("voice_cloned_sample", {})
    if voice.get("name") != "kugelaudio-0-open-hello-sine":
        raise SystemExit("FAIL: voice_cloned_sample name mismatch")
    if not voice.get("ground_truth"):
        raise SystemExit("FAIL: voice_cloned_sample missing ground_truth path")
    if str(Path(voice["ground_truth"]).name) != "canonical.wav":
        raise SystemExit("FAIL: voice_cloned_sample ground_truth filename unexpected")

    print("KugelAudio eval fixture OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
