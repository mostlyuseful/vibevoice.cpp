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
    if shared["reference_audio"]["path"] != str((repo / "tests" / "fixtures" / "reference_sine.wav").resolve()):
        raise SystemExit("FAIL: reference audio path was not normalized into shared_run")
    if shared["reference_audio"]["sha256"] is None:
        raise SystemExit("FAIL: sha256 should be present for actual fixture file")
    if shared["evaluators"]["transcription"]["backend"] != "faster-whisper":
        raise SystemExit("FAIL: faster-whisper evaluator not pinned into plan")
    if shared["evaluators"]["speaker_similarity"]["backend"] != "speechbrain-ecapa":
        raise SystemExit("FAIL: speaker-similarity evaluator not pinned into plan")
    if shared["metric_thresholds"]["speaker_similarity"]["floor"] != 0.6:
        raise SystemExit("FAIL: metric thresholds not normalized into shared_run")

    if plan["canonical"]["command"][0:3] != ["uv", "run", "python"]:
        raise SystemExit("FAIL: canonical command shape mismatch")
    if plan["ggml"]["command"][0] != str((repo / "build" / "bin" / "kugelaudio-cli").resolve()):
        raise SystemExit("FAIL: ggml CLI path was not normalized as expected")
    if plan["ggml"]["backend"] != "cpu":
        raise SystemExit("FAIL: ggml backend should default to cpu from fixture config")
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
    if not plan["canonical"]["transcription_log_path"].endswith("canonical_transcription.json"):
        raise SystemExit("FAIL: canonical transcription log path mismatch")
    if not plan["ggml"]["speaker_similarity_log_path"].endswith("ggml_speaker_similarity.json"):
        raise SystemExit("FAIL: ggml speaker-similarity log path mismatch")

    if not result_template.exists():
        raise SystemExit("FAIL: result template was not written")
    results = json.loads(result_template.read_text(encoding="utf-8"))
    result_template.unlink()
    if results["schema_version"] != 2:
        raise SystemExit("FAIL: results schema_version mismatch")
    if results["shared_run"] != shared:
        raise SystemExit("FAIL: results shared_run did not match plan shared_run")
    if results["canonical"]["status"] != "planned" or results["ggml"]["status"] != "planned":
        raise SystemExit("FAIL: planned results template should mark both runs as planned")
    if results["ggml"].get("backend") != "cpu":
        raise SystemExit("FAIL: results template should record ggml backend")
    if results["canonical"]["return_code"] is not None or results["ggml"]["return_code"] is not None:
        raise SystemExit("FAIL: planned results template should not have return codes yet")
    if results["canonical"]["output_sha256"] is not None or results["ggml"]["output_sha256"] is not None:
        raise SystemExit("FAIL: planned results template should not have output hashes yet")
    if results["canonical"]["transcript_recall"] is not None or results["ggml"]["speaker_similarity"] is not None:
        raise SystemExit("FAIL: planned results template should not have evaluator metrics yet")
    if "threshold_check" not in results:
        raise SystemExit("FAIL: results template missing threshold_check field")
    if results["metric_thresholds"]["transcript_recall"]["ratio"] != 0.95:
        raise SystemExit("FAIL: results template missing metric thresholds")

    proc2 = subprocess.run(
        [
            "uv",
            "run",
            str(script),
            "--config",
            str(cfg),
            "--allow-missing-artifacts",
            "--ggml-backend",
            "vulkan",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc2.returncode != 0:
        raise SystemExit(f"FAIL: eval plan override rc={proc2.returncode}\nSTDERR:\n{proc2.stderr}\nSTDOUT:\n{proc2.stdout}")
    plan2 = json.loads(proc2.stdout)
    if plan2["ggml"]["backend"] != "vulkan":
        raise SystemExit("FAIL: --ggml-backend vulkan did not override plan backend")

    print("KugelAudio eval plan OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
