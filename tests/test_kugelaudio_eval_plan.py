#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
import json
import subprocess
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    cfg = repo / "tests" / "fixtures" / "kugelaudio_eval_config.json"
    script = repo / "scripts" / "eval_kugelaudio_divergence.py"
    proc = subprocess.run(
        [
            "uv",
            "run",
            str(script),
            "--config",
            str(cfg),
            "--allow-missing-artifacts",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"FAIL: eval plan script rc={proc.returncode}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")
    try:
        plan = json.loads(proc.stdout)
    except Exception as e:
        raise SystemExit(f"FAIL: stdout was not valid JSON plan: {e}\n{proc.stdout}")

    if plan["generation"]["seed"] != 12345:
        raise SystemExit("FAIL: seed was not propagated into plan")
    if plan["generation"]["steps"] != 8 or plan["generation"]["max_frames"] != 32:
        raise SystemExit("FAIL: generation settings were not normalized correctly")
    if plan["canonical"]["command"][0:3] != ["uv", "run", "python"]:
        raise SystemExit("FAIL: canonical command shape mismatch")
    if plan["ggml"]["command"][0] != str((repo / "build" / "bin" / "vibevoice-cli").resolve()):
        raise SystemExit("FAIL: ggml CLI path was not normalized as expected")
    if "--seed" not in plan["ggml"]["command"]:
        raise SystemExit("FAIL: ggml command missing --seed")
    if "torch.manual_seed(12345)" not in plan["canonical"]["command"][4]:
        raise SystemExit("FAIL: canonical command missing explicit torch.manual_seed")

    print("KugelAudio eval plan OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
