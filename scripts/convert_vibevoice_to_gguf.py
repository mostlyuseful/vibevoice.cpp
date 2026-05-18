#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gguf",
#     "safetensors",
#     "torch",
# ]
#
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
# ///

"""Convert a VibeVoice or KugelAudio safetensors checkpoint to a single .gguf.

KugelAudio v1 support is intentionally narrow: the converter accepts only the
published `kugelaudio/kugelaudio-0-open` checkpoint shape and emits an explicit
`kugelaudio.*` metadata contract alongside legacy `vibevoice.*` keys needed by
existing runtime code during migration.

Output gguf carries:
  metadata: kugelaudio.schema_version, .checkpoint, .decoder.*, .acoustic.*,
            .semantic.*, .diffusion.*, .sample_rate, plus legacy
            vibevoice.* compatibility keys
  tensors:  remapped per the table at the bottom of this file
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


# ---------- key rewrite table ----------
# Each entry: (regex, replacement template). Order matters; the first match wins.
# Use (?P<i>\d+) etc for named groups referenced in the replacement.

REWRITES: list[tuple[re.Pattern, str]] = [
    # ---- ASR variant: top-level lm_head ----
    (re.compile(r"^lm_head\.weight$"),
     "lm_head.weight"),

    # ---- base LM (4 lower layers in realtime, 28 in ASR/1.5B) ----
    (re.compile(r"^model\.language_model\.embed_tokens\.weight$"),
     "lm.tok_embd.weight"),

    (re.compile(r"^model\.language_model\.layers\.(?P<i>\d+)\.self_attn\.(?P<p>[qkv])_proj\.(?P<x>weight|bias)$"),
     r"lm.blk.\g<i>.attn_\g<p>.\g<x>"),
    (re.compile(r"^model\.language_model\.layers\.(?P<i>\d+)\.self_attn\.o_proj\.weight$"),
     r"lm.blk.\g<i>.attn_o.weight"),
    (re.compile(r"^model\.language_model\.layers\.(?P<i>\d+)\.input_layernorm\.weight$"),
     r"lm.blk.\g<i>.attn_norm.weight"),
    (re.compile(r"^model\.language_model\.layers\.(?P<i>\d+)\.post_attention_layernorm\.weight$"),
     r"lm.blk.\g<i>.ffn_norm.weight"),
    (re.compile(r"^model\.language_model\.layers\.(?P<i>\d+)\.mlp\.(?P<p>gate|up|down)_proj\.weight$"),
     r"lm.blk.\g<i>.ffn_\g<p>.weight"),
    (re.compile(r"^model\.language_model\.norm\.weight$"),
     "lm.output_norm.weight"),

    # ---- TTS LM (20 upper layers) ----
    (re.compile(r"^model\.tts_language_model\.embed_tokens\.weight$"),
     "tlm.tok_embd.weight"),

    (re.compile(r"^model\.tts_language_model\.layers\.(?P<i>\d+)\.self_attn\.(?P<p>[qkv])_proj\.(?P<x>weight|bias)$"),
     r"tlm.blk.\g<i>.attn_\g<p>.\g<x>"),
    (re.compile(r"^model\.tts_language_model\.layers\.(?P<i>\d+)\.self_attn\.o_proj\.weight$"),
     r"tlm.blk.\g<i>.attn_o.weight"),
    (re.compile(r"^model\.tts_language_model\.layers\.(?P<i>\d+)\.input_layernorm\.weight$"),
     r"tlm.blk.\g<i>.attn_norm.weight"),
    (re.compile(r"^model\.tts_language_model\.layers\.(?P<i>\d+)\.post_attention_layernorm\.weight$"),
     r"tlm.blk.\g<i>.ffn_norm.weight"),
    (re.compile(r"^model\.tts_language_model\.layers\.(?P<i>\d+)\.mlp\.(?P<p>gate|up|down)_proj\.weight$"),
     r"tlm.blk.\g<i>.ffn_\g<p>.weight"),
    (re.compile(r"^model\.tts_language_model\.norm\.weight$"),
     "tlm.output_norm.weight"),

    # ---- TTS input-type embedding ----
    (re.compile(r"^model\.tts_input_types\.weight$"),
     "tts.input_types.weight"),

    # ---- speech scaling buffers ----
    (re.compile(r"^model\.speech_scaling_factor$"),
     "speech.scaling"),
    (re.compile(r"^model\.speech_bias_factor$"),
     "speech.bias"),

    # ---- acoustic connector ----
    (re.compile(r"^model\.acoustic_connector\.fc1\.(?P<x>weight|bias)$"),
     r"ac.fc1.\g<x>"),
    (re.compile(r"^model\.acoustic_connector\.norm\.weight$"),
     r"ac.norm.weight"),
    (re.compile(r"^model\.acoustic_connector\.fc2\.(?P<x>weight|bias)$"),
     r"ac.fc2.\g<x>"),

    # ---- prediction head (diffusion) ----
    (re.compile(r"^model\.prediction_head\.noisy_images_proj\.weight$"),
     "dh.noisy_proj"),
    (re.compile(r"^model\.prediction_head\.cond_proj\.weight$"),
     "dh.cond_proj"),
    (re.compile(r"^model\.prediction_head\.t_embedder\.mlp\.0\.weight$"),
     "dh.t_embed_lin1"),
    (re.compile(r"^model\.prediction_head\.t_embedder\.mlp\.2\.weight$"),
     "dh.t_embed_lin2"),
    (re.compile(r"^model\.prediction_head\.layers\.(?P<i>\d+)\.norm\.weight$"),
     r"dh.layer_\g<i>.norm"),
    (re.compile(r"^model\.prediction_head\.layers\.(?P<i>\d+)\.adaLN_modulation\.1\.weight$"),
     r"dh.layer_\g<i>.adaln"),
    (re.compile(r"^model\.prediction_head\.layers\.(?P<i>\d+)\.ffn\.(?P<p>gate|up|down)_proj\.weight$"),
     r"dh.layer_\g<i>.ffn_\g<p>"),
    (re.compile(r"^model\.prediction_head\.final_layer\.linear\.weight$"),
     "dh.final.proj"),
    (re.compile(r"^model\.prediction_head\.final_layer\.adaLN_modulation\.1\.weight$"),
     "dh.final.adaln"),

    # ---- acoustic encoder (ASR / 1.5B variants) ----
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.enc.stem.\g<x>"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.downsample_layers\.(?P<i>\d+)\.0\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.enc.down_\g<i>.\g<x>"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.head\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.enc.head.\g<x>"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.norm\.weight$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.norm"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_norm\.weight$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.ffn_norm"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.gamma$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.gamma"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_gamma$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.ffn_gamma"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.mixer\.conv\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.mixer_\g<x>"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear1\.(?P<x>weight|bias)$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.ffn_linear1__SUFFIX__"),
    (re.compile(r"^model\.acoustic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear2\.(?P<x>weight|bias)$"),
     r"at.enc.stage_\g<i>_block_\g<j>.weight.ffn_linear2__SUFFIX__"),

    # ---- semantic encoder (ASR / 1.5B variants) ----
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.downsample_layers\.0\.0\.conv\.conv\.(?P<x>weight|bias)$"),
     r"st.enc.stem.\g<x>"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.downsample_layers\.(?P<i>\d+)\.0\.conv\.conv\.(?P<x>weight|bias)$"),
     r"st.enc.down_\g<i>.\g<x>"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.head\.conv\.conv\.(?P<x>weight|bias)$"),
     r"st.enc.head.\g<x>"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.norm\.weight$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.norm"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_norm\.weight$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.ffn_norm"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.gamma$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.gamma"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_gamma$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.ffn_gamma"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.mixer\.conv\.conv\.conv\.(?P<x>weight|bias)$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.mixer_\g<x>"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear1\.(?P<x>weight|bias)$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.ffn_linear1__SUFFIX__"),
    (re.compile(r"^model\.semantic_tokenizer\.encoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear2\.(?P<x>weight|bias)$"),
     r"st.enc.stage_\g<i>_block_\g<j>.weight.ffn_linear2__SUFFIX__"),

    # ---- semantic connector (ASR / 1.5B variants) ----
    (re.compile(r"^model\.semantic_connector\.fc1\.(?P<x>weight|bias)$"),
     r"sc.fc1.\g<x>"),
    (re.compile(r"^model\.semantic_connector\.norm\.weight$"),
     r"sc.norm.weight"),
    (re.compile(r"^model\.semantic_connector\.fc2\.(?P<x>weight|bias)$"),
     r"sc.fc2.\g<x>"),

    # ---- acoustic decoder: stem (upsample 0) ----
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.upsample_layers\.0\.0\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.dec.stem.\g<x>"),
    # ---- acoustic decoder: transposed upsamples 1..6 ----
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.upsample_layers\.(?P<i>\d+)\.0\.convtr\.convtr\.(?P<x>weight|bias)$"),
     r"at.dec.up_\g<i>.\g<x>"),
    # ---- acoustic decoder: head ----
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.head\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.dec.head.\g<x>"),
    # ---- acoustic decoder: stages ----
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.norm\.weight$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.norm"),
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_norm\.weight$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.ffn_norm"),
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.gamma$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.gamma"),
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn_gamma$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.ffn_gamma"),
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.mixer\.conv\.conv\.conv\.(?P<x>weight|bias)$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.mixer_\g<x>"),
    # Block1D.ffn.linear[12] -> ffn_linearN (weight) or ffn_linearN_bias.
    # Use sentinel `__SUFFIX__` rewritten in remap() below.
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear1\.(?P<x>weight|bias)$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.ffn_linear1__SUFFIX__"),
    (re.compile(r"^model\.acoustic_tokenizer\.decoder\.stages\.(?P<i>\d+)\.(?P<j>\d+)\.ffn\.linear2\.(?P<x>weight|bias)$"),
     r"at.dec.stage_\g<i>_block_\g<j>.weight.ffn_linear2__SUFFIX__"),

    # ---- EOS classifier ----
    (re.compile(r"^tts_eos_classifier\.fc1\.(?P<x>weight|bias)$"),
     r"eos.fc1.\g<x>"),
    (re.compile(r"^tts_eos_classifier\.fc2\.(?P<x>weight|bias)$"),
     r"eos.fc2.\g<x>"),
]


def remap(name: str) -> str | None:
    """Return the new gguf tensor name, or None if no rule matches."""
    for pat, repl in REWRITES:
        m = pat.match(name)
        if not m:
            continue
        out = pat.sub(repl, name)
        if "__SUFFIX__" in out:
            # `weight` → "" (suffix dropped), `bias` → "_bias"
            x = m.group("x")
            out = out.replace("__SUFFIX__", "" if x == "weight" else "_bias")
        return out
    return None


SUPPORTED_KUGELAUDIO_SIGNATURE = {
    "decoder_hidden_size": 3584,
    "decoder_num_hidden_layers": 28,
    "decoder_num_attention_heads": 28,
    "decoder_num_key_value_heads": 4,
    "decoder_vocab_size": 152064,
    "tts_backbone_num_hidden_layers": None,
    "acoustic_vae_dim": 64,
    "diffusion_latent_size": 64,
}


def _import_runtime_deps() -> tuple[Any, Callable[..., Any]]:
    try:
        gguf = importlib.import_module("gguf")
        safe_mod = importlib.import_module("safetensors")
        return gguf, safe_mod.safe_open
    except ImportError as e:
        sys.stderr.write(f"error: use uv to provide gguf and safetensors\n  {e}\n")
        sys.exit(1)


def _split_depths(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(x) for x in value.split("-")]
    return list(value)


def _to_dtype(arr: np.ndarray, dtype: str) -> np.ndarray:
    has_np_bfloat16 = hasattr(np, "bfloat16")
    if has_np_bfloat16 and arr.dtype == np.dtype("bfloat16"):
        arr = arr.astype(np.float32)
    elif str(arr.dtype) in ("torch.bfloat16", "bfloat16"):
        arr = arr.astype(np.float32)
    if arr.dtype != np.float32 and arr.dtype != np.float16:
        arr = arr.astype(np.float32)
    if dtype == "fp16" and arr.dtype == np.float32 and arr.size > 1:
        arr = arr.astype(np.float16)
    return np.ascontiguousarray(arr)


def _tensor_to_numpy(t: Any, dtype: str) -> np.ndarray:
    # PyTorch tensors with bfloat16 can't call .numpy() directly.
    # Convert to float32 (or numpy bfloat16 if available) first.
    if getattr(t, "dtype", None) == getattr(torch, "bfloat16", None):
        t = t.float()
    elif hasattr(np, "bfloat16") and getattr(t, "dtype", None) == np.dtype("bfloat16"):
        t = t.float()
    arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
    return _to_dtype(np.asarray(arr), dtype)


def detect_variant(cfg: dict[str, Any], remapped_tensor_names: list[str]) -> str:
    if cfg.get("model_type") == "kugelaudio":
        dec = cfg.get("decoder_config") or {}
        dh = cfg.get("diffusion_head_config") or {}
        signature = {
            "decoder_hidden_size": dec.get("hidden_size"),
            "decoder_num_hidden_layers": dec.get("num_hidden_layers"),
            "decoder_num_attention_heads": dec.get("num_attention_heads"),
            "decoder_num_key_value_heads": dec.get("num_key_value_heads"),
            "decoder_vocab_size": dec.get("vocab_size"),
            "tts_backbone_num_hidden_layers": cfg.get("tts_backbone_num_hidden_layers"),
            "acoustic_vae_dim": cfg.get("acoustic_vae_dim"),
            "diffusion_latent_size": dh.get("latent_size"),
        }
        if signature == SUPPORTED_KUGELAUDIO_SIGNATURE:
            return "kugelaudio-0-open"
        raise ValueError(
            "unsupported KugelAudio checkpoint: only kugelaudio/kugelaudio-0-open "
            f"is supported in v1 (got signature={signature})"
        )

    archs = cfg.get("architectures") or []
    arch = archs[0] if archs else ""
    has_lm_head = any(k.startswith("lm_head") for k in remapped_tensor_names)
    has_prediction_hd = any(k.startswith("dh.") for k in remapped_tensor_names)
    has_at_decoder = any(k.startswith("at.dec.") for k in remapped_tensor_names)
    if "Streaming" in arch or cfg.get("tts_backbone_num_hidden_layers"):
        return "realtime-0.5b"
    if "ASR" in arch:
        return "asr-7b"
    if has_prediction_hd and has_at_decoder:
        return "1.5b"
    if has_lm_head:
        return "asr-7b"
    return cfg.get("model_type", "vibevoice")


def _append_qwen_stack_requirements(out: list[str], prefix: str, n_layers: int) -> None:
    out.append(f"{prefix}.tok_embd.weight")
    for i in range(n_layers):
        base = f"{prefix}.blk.{i}."
        out.extend([
            base + "attn_q.weight",
            base + "attn_q.bias",
            base + "attn_k.weight",
            base + "attn_k.bias",
            base + "attn_v.weight",
            base + "attn_v.bias",
            base + "attn_o.weight",
            base + "attn_norm.weight",
            base + "ffn_norm.weight",
            base + "ffn_gate.weight",
            base + "ffn_up.weight",
            base + "ffn_down.weight",
        ])


def _append_block1d_requirements(out: list[str], prefix: str) -> None:
    out.extend([
        prefix + ".weight.norm",
        prefix + ".weight.ffn_norm",
        prefix + ".weight.mixer_weight",
        prefix + ".weight.mixer_bias",
        prefix + ".weight.gamma",
        prefix + ".weight.ffn_gamma",
        prefix + ".weight.ffn_linear1",
        prefix + ".weight.ffn_linear1_bias",
        prefix + ".weight.ffn_linear2",
        prefix + ".weight.ffn_linear2_bias",
    ])


def _append_encoder_requirements(out: list[str], prefix: str, depths: list[int], n_downs: int) -> None:
    out.extend([f"{prefix}.stem.weight", f"{prefix}.stem.bias"])
    for i in range(1, n_downs + 1):
        out.extend([f"{prefix}.down_{i}.weight", f"{prefix}.down_{i}.bias"])
    for stage_i, depth in enumerate(depths):
        for block_i in range(depth):
            _append_block1d_requirements(out, f"{prefix}.stage_{stage_i}_block_{block_i}")
    out.extend([f"{prefix}.head.weight", f"{prefix}.head.bias"])


def _append_decoder_requirements(out: list[str], prefix: str, depths: list[int], n_ups: int) -> None:
    out.extend([f"{prefix}.stem.weight", f"{prefix}.stem.bias"])
    for i in range(1, n_ups + 1):
        out.extend([f"{prefix}.up_{i}.weight", f"{prefix}.up_{i}.bias"])
    for stage_i, depth in enumerate(depths):
        for block_i in range(depth):
            _append_block1d_requirements(out, f"{prefix}.stage_{stage_i}_block_{block_i}")
    out.extend([f"{prefix}.head.weight", f"{prefix}.head.bias"])


def _append_diffusion_head_requirements(out: list[str], head_layers: int) -> None:
    out.extend([
        "dh.noisy_proj",
        "dh.cond_proj",
        "dh.t_embed_lin1",
        "dh.t_embed_lin2",
        "dh.final.proj",
        "dh.final.adaln",
    ])
    for i in range(head_layers):
        out.extend([
            f"dh.layer_{i}.norm",
            f"dh.layer_{i}.adaln",
            f"dh.layer_{i}.ffn_gate",
            f"dh.layer_{i}.ffn_up",
            f"dh.layer_{i}.ffn_down",
        ])


def required_tensor_names_for_variant(cfg: dict[str, Any], variant: str) -> list[str]:
    if variant != "kugelaudio-0-open":
        return []

    dec = cfg["decoder_config"]
    ac = cfg["acoustic_tokenizer_config"]
    sm = resolve_semantic_config(cfg)
    dh = cfg.get("diffusion_head_config") or {}

    n_total = dec["num_hidden_layers"]
    n_tts_layers = cfg.get("tts_backbone_num_hidden_layers", 0) or 0
    n_lm_layers = n_total - n_tts_layers
    ac_enc_depths = _split_depths(ac["encoder_depths"])
    ac_dec_depths = _split_depths(ac.get("decoder_depths") or list(reversed(ac_enc_depths)))
    sm_enc_depths = _split_depths(sm.get("encoder_depths") or ac_enc_depths)
    head_layers = int(dh.get("head_layers", 4))
    n_downs = len(ac["encoder_ratios"])
    n_sem_downs = len(sm["encoder_ratios"])
    n_ups = len(ac["encoder_ratios"])

    required: list[str] = []
    _append_qwen_stack_requirements(required, "lm", n_lm_layers)
    required.extend([
        "lm.output_norm.weight",
        "lm_head.weight",
        "speech.scaling",
        "speech.bias",
        "ac.fc1.weight",
        "ac.fc1.bias",
        "ac.norm.weight",
        "ac.fc2.weight",
        "ac.fc2.bias",
        "sc.fc1.weight",
        "sc.fc1.bias",
        "sc.norm.weight",
        "sc.fc2.weight",
        "sc.fc2.bias",
    ])
    _append_diffusion_head_requirements(required, head_layers)
    _append_decoder_requirements(required, "at.dec", ac_dec_depths, n_ups)
    _append_encoder_requirements(required, "at.enc", ac_enc_depths, n_downs)
    _append_encoder_requirements(required, "st.enc", sm_enc_depths, n_sem_downs)
    return required


def _tensor_family(name: str) -> str:
    if name.startswith("lm.") or name == "lm_head.weight":
        return "language_model"
    if name.startswith("dh."):
        return "diffusion_head"
    if name.startswith("at.dec."):
        return "acoustic_decoder"
    if name.startswith("at.enc."):
        return "acoustic_encoder"
    if name.startswith("st.enc."):
        return "semantic_encoder"
    if name.startswith("sc."):
        return "semantic_connector"
    if name.startswith("ac."):
        return "acoustic_connector"
    if name.startswith("speech."):
        return "speech_scalars"
    return "other"


def format_missing_tensor_diagnostics(variant: str, missing: list[str]) -> str:
    families: dict[str, list[str]] = {}
    for name in missing:
        families.setdefault(_tensor_family(name), []).append(name)

    family_summary = ", ".join(
        f"{family}={len(names)}" for family, names in sorted(families.items())
    )
    examples = []
    for family, names in sorted(families.items()):
        preview = ", ".join(names[:3])
        suffix = " ..." if len(names) > 3 else ""
        examples.append(f"{family}: {preview}{suffix}")

    return (
        f"missing required tensors for {variant} "
        f"({len(missing)} total; {family_summary}). "
        f"Examples -> {'; '.join(examples)}"
    )


def validate_required_tensors(cfg: dict[str, Any], variant: str, tensor_names: list[str]) -> None:
    required = required_tensor_names_for_variant(cfg, variant)
    if not required:
        return
    present = set(tensor_names)
    missing = [name for name in required if name not in present]
    if missing:
        raise ValueError(format_missing_tensor_diagnostics(variant, missing))


def resolve_semantic_config(cfg: dict[str, Any]) -> dict[str, Any]:
    sm = cfg.get("semantic_tokenizer_config")
    if sm:
        return dict(sm)

    ac = cfg["acoustic_tokenizer_config"]
    return {
        "vae_dim": cfg.get("semantic_vae_dim", ac.get("vae_dim", 64)),
        "encoder_ratios": list(ac["encoder_ratios"]),
        "encoder_depths": ac["encoder_depths"],
    }


def add_metadata(writer: Any, cfg: dict[str, Any], variant: str) -> None:
    dec = cfg["decoder_config"]
    ac = cfg["acoustic_tokenizer_config"]
    sm = resolve_semantic_config(cfg)
    dh = cfg.get("diffusion_head_config") or {}

    n_total = dec["num_hidden_layers"]
    n_tts_layers = cfg.get("tts_backbone_num_hidden_layers", 0) or 0
    n_lm_layers = n_total - n_tts_layers if n_tts_layers > 0 else n_total
    head_dim = dec["hidden_size"] // dec["num_attention_heads"]
    enc_depths = _split_depths(ac["encoder_depths"])
    dec_depths = ac.get("decoder_depths") or list(reversed(enc_depths))
    dec_depths = _split_depths(dec_depths)

    writer.add_uint32("kugelaudio.schema_version", 1)
    writer.add_string("kugelaudio.architecture", "kugelaudio")
    writer.add_string("kugelaudio.checkpoint", variant)
    writer.add_uint32("kugelaudio.decoder.hidden_size", dec["hidden_size"])
    writer.add_uint32("kugelaudio.decoder.num_hidden_layers", n_total)
    writer.add_uint32("kugelaudio.decoder.tts_hidden_layers", n_tts_layers)
    writer.add_uint32("kugelaudio.decoder.num_attention_heads", dec["num_attention_heads"])
    writer.add_uint32("kugelaudio.decoder.num_key_value_heads", dec["num_key_value_heads"])
    writer.add_uint32("kugelaudio.decoder.head_dim", head_dim)
    writer.add_uint32("kugelaudio.decoder.vocab_size", dec["vocab_size"])
    writer.add_float32("kugelaudio.decoder.rope_theta", float(dec["rope_theta"]))
    writer.add_float32("kugelaudio.decoder.rms_norm_eps", float(dec["rms_norm_eps"]))
    writer.add_uint32("kugelaudio.acoustic.vae_dim", ac["vae_dim"])
    writer.add_array("kugelaudio.acoustic.encoder_ratios", list(ac["encoder_ratios"]))
    writer.add_array("kugelaudio.acoustic.encoder_depths", enc_depths)
    writer.add_array("kugelaudio.acoustic.decoder_depths", dec_depths)
    writer.add_float32("kugelaudio.acoustic.fix_std", float(ac.get("fix_std", 0.0)))
    writer.add_string("kugelaudio.acoustic.std_dist_type", str(ac.get("std_dist_type", "none")))
    writer.add_uint32("kugelaudio.semantic.vae_dim", sm.get("vae_dim", cfg.get("semantic_vae_dim", 64)))
    writer.add_array("kugelaudio.semantic.encoder_ratios", list(sm["encoder_ratios"]))
    writer.add_array("kugelaudio.semantic.encoder_depths", _split_depths(sm["encoder_depths"]))
    writer.add_float32("kugelaudio.semantic.fix_std", float(sm.get("fix_std", 0.0)))
    writer.add_string("kugelaudio.semantic.std_dist_type", str(sm.get("std_dist_type", "none")))
    if dh:
        writer.add_uint32("kugelaudio.diffusion.head_layers", dh.get("head_layers", 4))
        writer.add_float32("kugelaudio.diffusion.ffn_ratio", float(dh.get("head_ffn_ratio", 3.0)))
        writer.add_uint32("kugelaudio.diffusion.latent_size", dh.get("latent_size", 64))
    writer.add_uint32("kugelaudio.sample_rate", 24000)

    writer.add_string("vibevoice.variant", variant)
    writer.add_uint32("vibevoice.hidden", dec["hidden_size"])
    writer.add_uint32("vibevoice.n_layers_lm", n_lm_layers)
    writer.add_uint32("vibevoice.n_layers_tlm", n_tts_layers)
    writer.add_uint32("vibevoice.n_heads", dec["num_attention_heads"])
    writer.add_uint32("vibevoice.n_kv_heads", dec["num_key_value_heads"])
    writer.add_uint32("vibevoice.head_dim", head_dim)
    writer.add_uint32("vibevoice.vocab_size", dec["vocab_size"])
    writer.add_float32("vibevoice.rope_theta", float(dec["rope_theta"]))
    writer.add_float32("vibevoice.rms_norm_eps", float(dec["rms_norm_eps"]))
    writer.add_uint32("vibevoice.acoustic.vae_dim", ac["vae_dim"])
    writer.add_array("vibevoice.acoustic.encoder_ratios", list(ac["encoder_ratios"]))
    writer.add_array("vibevoice.acoustic.encoder_depths", enc_depths)
    writer.add_array("vibevoice.acoustic.decoder_depths", dec_depths)
    writer.add_float32("vibevoice.acoustic.fix_std", float(ac.get("fix_std", 0.0)))
    writer.add_string("vibevoice.acoustic.std_dist_type", str(ac.get("std_dist_type", "none")))
    writer.add_uint32("vibevoice.semantic.vae_dim", sm.get("vae_dim", cfg.get("semantic_vae_dim", 64)))
    writer.add_array("vibevoice.semantic.encoder_ratios", list(sm["encoder_ratios"]))
    writer.add_array("vibevoice.semantic.encoder_depths", _split_depths(sm["encoder_depths"]))
    writer.add_float32("vibevoice.semantic.fix_std", float(sm.get("fix_std", 0.0)))
    writer.add_string("vibevoice.semantic.std_dist_type", str(sm.get("std_dist_type", "none")))
    if dh:
        writer.add_uint32("vibevoice.diffusion.head_layers", dh.get("head_layers", 4))
        writer.add_float32("vibevoice.diffusion.ffn_ratio", float(dh.get("head_ffn_ratio", 3.0)))
        writer.add_uint32("vibevoice.diffusion.latent", dh.get("latent_size", 64))
    writer.add_uint32("vibevoice.sample_rate", 24000)


def _tensor_info_nbytes(shape: list[int], out_dtype: np.dtype) -> int:
    """Number of bytes for a tensor in the output GGUF."""
    elem_size = 2 if out_dtype == np.float16 else 4
    count = 1
    for s in shape:
        count *= s
    return count * elem_size


def _output_details(shape: list[int], fmt: str) -> tuple[np.dtype, int]:
    """Return (output_numpy_dtype, nbytes) given --dtype fmt."""
    if fmt == "fp32":
        return np.float32, _tensor_info_nbytes(shape, np.float32)
    elem_count = 1
    for s in shape:
        elem_count *= s
    out_dtype = np.float16 if elem_count > 1 else np.float32
    return out_dtype, _tensor_info_nbytes(shape, out_dtype)


def convert_checkpoint(
    src: Path,
    out: Path,
    *,
    strict: bool,
    dtype: str,
    gguf_module: Any,
    safe_open_fn: Callable[..., Any],
) -> int:
    with open(src / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    files = sorted(src.glob("model*.safetensors"))
    if not files:
        sys.stderr.write(f"error: no model*.safetensors under {src}\n")
        return 1

    # ── Phase 1: scan tensor metadata (names, shapes, nbytes) without loading data ──
    # meta entries are ordered to match safetensors iteration so the data pass
    # (Phase 3) can call write_tensor_data() in lockstep.
    meta: list[tuple[str, list[int], np.dtype, int]] = []
    unmapped: list[str] = []
    for f in files:
        with safe_open_fn(str(f), framework="numpy") as fh:
            for k in fh.keys():
                sl = fh.get_slice(k)
                shape = sl.get_shape()
                new = remap(k)
                if new is None:
                    unmapped.append(k)
                    continue
                out_dtype, nbytes = _output_details(shape, dtype)
                meta.append((new, shape, out_dtype, nbytes))

    if unmapped:
        msg = (f"warning: {len(unmapped)} unmapped keys (first 10):\n"
               + "\n".join(f"  {k}" for k in unmapped[:10]))
        sys.stderr.write(msg + "\n")
        if strict:
            return 2

    tnames = [m[0] for m in meta]
    try:
        variant = detect_variant(cfg, tnames)
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        return 3

    dec = cfg["decoder_config"]
    synthesize_lm_head = False
    if (
        variant == "1.5b"
        and dec.get("tie_word_embeddings", False)
        and not any(k.startswith("lm_head") for k in tnames)
    ):
        tok_meta = next((m for m in meta if m[0] == "lm.tok_embd.weight"), None)
        if tok_meta is None:
            sys.stderr.write("error: 1.5b variant missing lm.tok_embd.weight; cannot synthesise tied lm_head\n")
            return 4
        meta.append(("lm_head.weight", tok_meta[1], tok_meta[2], tok_meta[3]))
        synthesize_lm_head = True

    try:
        validate_required_tensors(cfg, variant, [m[0] for m in meta])
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        return 5

    # ── Phase 2: write GGUF header, KV metadata, and tensor info ──
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf_module.GGUFWriter(str(out), arch="vibevoice")
    add_metadata(writer, cfg, variant)
    for name, shape, out_dtype, nbytes in meta:
        writer.add_tensor_info(name, shape, out_dtype, nbytes)
    writer.open_output_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # ── Phase 3: load + write one tensor at a time ──
    for f in files:
        with safe_open_fn(str(f), framework="pt") as fh:
            for k in fh.keys():
                new = remap(k)
                if new is None:
                    continue
                arr = _tensor_to_numpy(fh.get_tensor(k), dtype)
                writer.write_tensor_data(arr)

    if synthesize_lm_head:
        embd_name = "model.language_model.embed_tokens.weight"
        for f in files:
            with safe_open_fn(str(f), framework="pt") as fh:
                if embd_name in fh.keys():
                    arr = _tensor_to_numpy(fh.get_tensor(embd_name), dtype)
                    break
        else:
            sys.stderr.write("error: cannot find embed_tokens.weight for tied lm_head\n")
            return 6
        writer.write_tensor_data(arr)

    writer.close()

    n_total = dec["num_hidden_layers"]
    n_tts_layers = cfg.get("tts_backbone_num_hidden_layers", 0) or 0
    is_kugelaudio = variant.startswith("kugelaudio")
    mode = "kugelaudio" if is_kugelaudio else "vibevoice"
    sys.stderr.write(
        f"convert: mode={mode} checkpoint={variant} src={src}\n"
        f"convert: wrote {out}: {len(meta)} tensors  (unmapped={len(unmapped)})  "
        f"hidden={dec['hidden_size']} lm_layers={n_total - n_tts_layers}+{n_tts_layers} "
        f"vocab={dec['vocab_size']} dtype={dtype}\n"
    )
    if is_kugelaudio:
        sem = cfg.get("semantic_tokenizer_config")
        sys.stderr.write(
            f"convert: conditioning=acoustic+semantic semantic_config={'present' if sem else 'derived_from_acoustic'}\n"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="model dir with config.json + model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict", action="store_true", help="fail on any unmapped source key")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = ap.parse_args(argv)

    gguf_module, safe_open_fn = _import_runtime_deps()
    return convert_checkpoint(
        Path(args.src),
        Path(args.out),
        strict=args.strict,
        dtype=args.dtype,
        gguf_module=gguf_module,
        safe_open_fn=safe_open_fn,
    )


if __name__ == "__main__":
    raise SystemExit(main())
