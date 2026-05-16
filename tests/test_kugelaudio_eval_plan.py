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
    result_template = repo / "tests" / "fixtures" / "tmp_eval_results_template.json"
    proc = subprocess.run(
        [
            "uv",
            "run",
            str(script),
            "--config",
            str(cfg),
            "--allow-missing-artifacts",
            "--write-result-template",
            str(result_template),
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

    shared = plan["shared_run"]
    if shared["generation"]["seed"] != 12345:
        raise SystemExit("FAIL: seed was not propagated into shared_run")
    if shared["generation"]["steps"] != 8 or shared["generation"]["max_frames"] != 32:
        raise SystemExit("FAIL: generation settings were not normalized correctly")
    if shared["text"] != "Hello world.":
        raise SystemExit("FAIL: text was not pinned into shared_run")
    if shared["reference_audio"]["path"] != str((repo / "artifacts" / "fixtures" / "reference.wav").resolve()):
        raise SystemExit("FAIL: reference audio path was not normalized into shared_run")
    if shared["reference_audio"]["sha256"] is not None:
        raise SystemExit("FAIL: sha256 should be absent in allow-missing-artifacts plan fixture")
    if plan["canonical"]["command"][0:3] != ["uv", "run", "python"]:
        raise SystemExit("FAIL: canonical command shape mismatch")
    if plan["ggml"]["command"][0] != str((repo / "build" / "bin" / "vibevoice-cli").resolve()):
        raise SystemExit("FAIL: ggml CLI path was not normalized as expected")
    if "--seed" not in plan["ggml"]["command"]:
        raise SystemExit("FAIL: ggml command missing --seed")
    if plan["ggml"]["command"][plan["ggml"]["command"].index("--text") + 1] != shared["text"]:
        raise SystemExit("FAIL: ggml command text diverged from shared_run")
    if plan["ggml"]["command"][plan["ggml"]["command"].index("--ref-audio") + 1] != shared["reference_audio"]["path"]:
        raise SystemExit("FAIL: ggml command reference audio diverged from shared_run")
    if plan["ggml"]["command"][plan["ggml"]["command"].index("--seed") + 1] != str(shared["generation"]["seed"]):
        raise SystemExit("FAIL: ggml command seed diverged from shared_run")
    if plan["ggml"]["command"][plan["ggml"]["command"].index("--steps") + 1] != str(shared["generation"]["steps"]):
        raise SystemExit("FAIL: ggml command steps diverged from shared_run")
    canonical_inline = plan["canonical"]["command"][4]
    if "torch.manual_seed(12345)" not in canonical_inline:
        raise SystemExit("FAIL: canonical command missing explicit torch.manual_seed")
    if shared["text"] not in canonical_inline or shared["reference_audio"]["path"] not in canonical_inline:
        raise SystemExit("FAIL: canonical command did not embed shared text/reference path")

    if not result_template.exists():
        raise SystemExit("FAIL: result template was not written")
    results = json.loads(result_template.read_text(encoding="utf-8"))
    result_template.unlink()
    if results["schema_version"] != 1:
        raise SystemExit("FAIL: results schema_version mismatch")
    if results["shared_run"] != shared:
        raise SystemExit("FAIL: results shared_run did not match plan shared_run")
    if results["canonical"]["status"] != "planned" or results["ggml"]["status"] != "planned":
        raise SystemExit("FAIL: planned results template should mark both runs as planned")
    if results["canonical"]["return_code"] is not None or results["ggml"]["return_code"] is not None:
        raise SystemExit("FAIL: planned results template should not have return codes yet")
    if results["canonical"]["output_sha256"] is not None or results["ggml"]["output_sha256"] is not None:
        raise SystemExit("FAIL: planned results template should not have output hashes yet")

    print("KugelAudio eval plan OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
