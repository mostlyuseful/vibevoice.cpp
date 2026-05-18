#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""Compare canonical and ggml tensor dump directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ALIASES_CANONICAL_TO_GGML = {
    "00_ref_audio_post_norm": "00_ref_audio_post_norm",
    "01_acoustic_encoder_out_raw": "01_acoustic_encoder_out",
    "02_acoustic_features_after_sampling": "01b_acoustic_features_after_sampling",
    "03_semantic_encoder_mean": "02_semantic_encoder_out",
    "04_semantic_aligned": "03_semantic_aligned",
    "05_acoustic_after_scale_bias": "04_acoustic_after_scale_bias",
    "06_acoustic_connector_out": "05_acoustic_connector_out",
    "07_semantic_connector_out": "06_semantic_connector_out",
    "08_fused_speech_embeds": "07_fused_speech_features",
    "09_prompt_input_ids": "08_prompt_input_ids",
    "10_pad_positions": "09_pad_positions",
    "11_prompt_embeds_pre_splice": "10_prompt_embeds_pre_splice",
    "12_prompt_embeds_post_splice": "11_prompt_embeds_post_splice",
    "13_prefill_hidden_last_pos": "12_prefill_hidden_last_pos",
    "14_prefill_hidden_last_neg": "13_prefill_hidden_last_neg",
    "15_first_logits": "14_first_logits",
    "16_first_selected_token": "15_first_selected_token",
    "17_first_diffusion_cond_pos": "16_first_diffusion_cond_pos",
    "18_first_diffusion_cond_neg": "17_first_diffusion_cond_neg",
    "19_first_diffusion_latent": "18_first_diffusion_latent",
    "20_first_step_embed": "19_first_step_embed",
}

DTYPES = {
    "float32": np.float32,
    "int32": np.int32,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lhs", required=True, help="canonical dump dir")
    p.add_argument("--rhs", required=True, help="ggml dump dir")
    p.add_argument("--output", help="optional json report path")
    return p.parse_args()


def load_index(dir_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(dir_path.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stage = meta["stage"]
        bin_path = dir_path / f"{stage}.bin"
        out[stage] = {
            "meta": meta,
            "bin": bin_path,
        }
    return out


def load_array(entry: dict[str, Any]) -> np.ndarray:
    meta = entry["meta"]
    dtype = DTYPES[meta["dtype"]]
    shape = tuple(meta["shape"])
    data = np.fromfile(entry["bin"], dtype=dtype)
    return data.reshape(shape)


def compare_arrays(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    if lhs.shape != rhs.shape:
        return {
            "shape_match": False,
            "lhs_shape": list(lhs.shape),
            "rhs_shape": list(rhs.shape),
        }
    lhs_f = lhs.astype(np.float64, copy=False).ravel()
    rhs_f = rhs.astype(np.float64, copy=False).ravel()
    diff = np.abs(lhs_f - rhs_f)
    lhs_norm = np.linalg.norm(lhs_f)
    rhs_norm = np.linalg.norm(rhs_f)
    cosine = None
    if lhs_norm > 0 and rhs_norm > 0:
        cosine = float(np.dot(lhs_f, rhs_f) / (lhs_norm * rhs_norm))
    elif lhs_norm == 0 and rhs_norm == 0:
        cosine = 1.0
    return {
        "shape_match": True,
        "lhs_shape": list(lhs.shape),
        "rhs_shape": list(rhs.shape),
        "max_abs_diff": float(diff.max(initial=0.0)),
        "mean_abs_diff": float(diff.mean() if diff.size else 0.0),
        "rmse": float(np.sqrt(np.mean((lhs_f - rhs_f) ** 2)) if diff.size else 0.0),
        "cosine": cosine,
    }


def main() -> int:
    args = parse_args()
    lhs_dir = Path(args.lhs).resolve()
    rhs_dir = Path(args.rhs).resolve()
    lhs = load_index(lhs_dir)
    rhs = load_index(rhs_dir)

    results: list[dict[str, Any]] = []
    seen_rhs: set[str] = set()

    for lhs_stage in sorted(lhs.keys()):
        rhs_stage = ALIASES_CANONICAL_TO_GGML.get(lhs_stage, lhs_stage)
        entry: dict[str, Any] = {
            "lhs_stage": lhs_stage,
            "rhs_stage": rhs_stage,
            "lhs_present": True,
            "rhs_present": rhs_stage in rhs,
        }
        if rhs_stage in rhs:
            seen_rhs.add(rhs_stage)
            entry.update(compare_arrays(load_array(lhs[lhs_stage]), load_array(rhs[rhs_stage])))
            entry["status"] = "ok" if entry.get("shape_match") else "shape_mismatch"
        else:
            entry["status"] = "missing_rhs"
        results.append(entry)

    for rhs_stage in sorted(rhs.keys()):
        if rhs_stage not in seen_rhs:
            results.append({
                "lhs_stage": None,
                "rhs_stage": rhs_stage,
                "lhs_present": False,
                "rhs_present": True,
                "status": "missing_lhs",
            })

    report = {
        "lhs": str(lhs_dir),
        "rhs": str(rhs_dir),
        "results": results,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for r in results:
        lhs_stage = r.get("lhs_stage") or "<none>"
        rhs_stage = r.get("rhs_stage") or "<none>"
        status = r["status"]
        if status == "ok":
            print(f"OK   {lhs_stage:32s} -> {rhs_stage:32s} shape={r['lhs_shape']} max={r['max_abs_diff']:.6g} mean={r['mean_abs_diff']:.6g} cos={r['cosine']}")
        elif status == "shape_mismatch":
            print(f"SHAPE {lhs_stage:32s} -> {rhs_stage:32s} lhs={r['lhs_shape']} rhs={r['rhs_shape']}")
        elif status == "missing_rhs":
            print(f"MISSR {lhs_stage:32s} -> {rhs_stage:32s}")
        else:
            print(f"MISSL {lhs_stage:32s} -> {rhs_stage:32s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
