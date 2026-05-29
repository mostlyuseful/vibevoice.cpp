#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["gguf", "numpy"]
# ///
import argparse
import json
from pathlib import Path

import gguf
import numpy as np


def add_tensor(w, name):
    w.add_tensor(name, np.zeros((1,), dtype=np.float32))


def add_qwen_stack(w, prefix, n_layers):
    add_tensor(w, f"{prefix}.tok_embd.weight")
    for i in range(n_layers):
        base = f"{prefix}.blk.{i}."
        for suffix in [
            "attn_q.weight", "attn_q.bias",
            "attn_k.weight", "attn_k.bias",
            "attn_v.weight", "attn_v.bias",
            "attn_o.weight",
            "attn_norm.weight", "ffn_norm.weight",
            "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
        ]:
            add_tensor(w, base + suffix)


def add_block1d(w, prefix):
    for suffix in [
        "weight.norm",
        "weight.ffn_norm",
        "weight.mixer_weight",
        "weight.mixer_bias",
        "weight.gamma",
        "weight.ffn_gamma",
        "weight.ffn_linear1",
        "weight.ffn_linear1_bias",
        "weight.ffn_linear2",
        "weight.ffn_linear2_bias",
    ]:
        add_tensor(w, f"{prefix}.{suffix}")


def add_encoder(w, prefix, depths, n_downs):
    add_tensor(w, f"{prefix}.stem.weight")
    add_tensor(w, f"{prefix}.stem.bias")
    for i in range(1, n_downs + 1):
        add_tensor(w, f"{prefix}.down_{i}.weight")
        add_tensor(w, f"{prefix}.down_{i}.bias")
    for stage_i, depth in enumerate(depths):
        for block_i in range(depth):
            add_block1d(w, f"{prefix}.stage_{stage_i}_block_{block_i}")
    add_tensor(w, f"{prefix}.head.weight")
    add_tensor(w, f"{prefix}.head.bias")


def add_decoder(w, prefix, depths, n_ups):
    add_tensor(w, f"{prefix}.stem.weight")
    add_tensor(w, f"{prefix}.stem.bias")
    for i in range(1, n_ups + 1):
        add_tensor(w, f"{prefix}.up_{i}.weight")
        add_tensor(w, f"{prefix}.up_{i}.bias")
    for stage_i, depth in enumerate(depths):
        for block_i in range(depth):
            add_block1d(w, f"{prefix}.stage_{stage_i}_block_{block_i}")
    add_tensor(w, f"{prefix}.head.weight")
    add_tensor(w, f"{prefix}.head.bias")


def add_diffusion_head(w, head_layers):
    for name in [
        "dh.noisy_proj", "dh.cond_proj",
        "dh.t_embed_lin1", "dh.t_embed_lin2",
        "dh.final.proj", "dh.final.adaln",
    ]:
        add_tensor(w, name)
    for i in range(head_layers):
        for suffix in ["norm", "adaln", "ffn_gate", "ffn_up", "ffn_down"]:
            add_tensor(w, f"dh.layer_{i}.{suffix}")


def write_fixture(path: Path, *, schema_version: int, checkpoint: str, omit_prefixes=(), omit_keys=()):
    omitted = tuple(omit_prefixes)
    omitted_keys = set(omit_keys)

    def keep(name: str) -> bool:
        return not any(name.startswith(prefix) for prefix in omitted)

    def add_u32(key: str, value: int):
        if key not in omitted_keys:
            w.add_uint32(key, value)

    def add_f32(key: str, value: float):
        if key not in omitted_keys:
            w.add_float32(key, value)

    def add_str(key: str, value: str):
        if key not in omitted_keys:
            w.add_string(key, value)

    def add_arr(key: str, value):
        if key not in omitted_keys:
            w.add_array(key, value)

    w = gguf.GGUFWriter(str(path), arch="vibevoice")
    add_u32("kugelaudio.schema_version", schema_version)
    add_str("kugelaudio.architecture", "kugelaudio")
    add_str("kugelaudio.checkpoint", checkpoint)
    add_u32("kugelaudio.decoder.hidden_size", 3584)
    add_u32("kugelaudio.decoder.num_hidden_layers", 28)
    add_u32("kugelaudio.decoder.tts_hidden_layers", 20)
    add_u32("kugelaudio.decoder.num_attention_heads", 28)
    add_u32("kugelaudio.decoder.num_key_value_heads", 4)
    add_u32("kugelaudio.decoder.head_dim", 128)
    add_u32("kugelaudio.decoder.vocab_size", 152064)
    add_f32("kugelaudio.decoder.rope_theta", 1000000.0)
    add_f32("kugelaudio.decoder.rms_norm_eps", 1e-6)
    add_u32("kugelaudio.acoustic.vae_dim", 64)
    add_arr("kugelaudio.acoustic.encoder_ratios", [8, 5, 5, 4, 2, 2])
    add_arr("kugelaudio.acoustic.encoder_depths", [3, 3, 3, 3, 3, 3, 8])
    add_arr("kugelaudio.acoustic.decoder_depths", [8, 3, 3, 3, 3, 3, 3])
    add_u32("kugelaudio.semantic.vae_dim", 64)
    add_arr("kugelaudio.semantic.encoder_ratios", [8, 5, 5, 4, 2, 2])
    add_arr("kugelaudio.semantic.encoder_depths", [3, 3, 3, 3, 3, 3, 8])
    add_u32("kugelaudio.diffusion.head_layers", 4)
    add_f32("kugelaudio.diffusion.ffn_ratio", 3.0)
    add_u32("kugelaudio.diffusion.latent_size", 64)
    add_u32("kugelaudio.sample_rate", 24000)

    add_qwen_stack(w, "lm", 8)
    for name in ["lm.output_norm.weight", "lm_head.weight", "speech.scaling", "speech.bias"]:
        if keep(name):
            add_tensor(w, name)
    for name in [
        "ac.fc1.weight", "ac.fc1.bias", "ac.norm.weight", "ac.fc2.weight", "ac.fc2.bias",
        "sc.fc1.weight", "sc.fc1.bias", "sc.norm.weight", "sc.fc2.weight", "sc.fc2.bias",
    ]:
        if keep(name):
            add_tensor(w, name)
    if keep("dh."):
        add_diffusion_head(w, 4)
    if keep("at.dec."):
        add_decoder(w, "at.dec", [8, 3, 3, 3, 3, 3, 3], 6)
    if keep("at.enc."):
        add_encoder(w, "at.enc", [3, 3, 3, 3, 3, 3, 8], 6)
    if keep("st.enc."):
        add_encoder(w, "st.enc", [3, 3, 3, 3, 3, 3, 8], 6)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_fixture(out_dir / "kugelaudio_loader_ok.gguf", schema_version=1, checkpoint="kugelaudio-0-open")
    write_fixture(out_dir / "kugelaudio_loader_bad_schema.gguf", schema_version=2, checkpoint="kugelaudio-0-open")
    write_fixture(out_dir / "kugelaudio_loader_bad_checkpoint.gguf", schema_version=1, checkpoint="kugelaudio-1-open")
    write_fixture(out_dir / "kugelaudio_loader_missing_semantic.gguf", schema_version=1, checkpoint="kugelaudio-0-open", omit_prefixes=("st.enc.", "sc."))
    write_fixture(out_dir / "kugelaudio_loader_missing_acoustic.gguf", schema_version=1, checkpoint="kugelaudio-0-open", omit_prefixes=("at.dec.",))
    write_fixture(out_dir / "kugelaudio_loader_missing_metadata.gguf", schema_version=1, checkpoint="kugelaudio-0-open", omit_keys=("kugelaudio.diffusion.latent_size",))
    print(json.dumps({
        "ok": str(out_dir / "kugelaudio_loader_ok.gguf"),
        "bad_schema": str(out_dir / "kugelaudio_loader_bad_schema.gguf"),
        "bad_checkpoint": str(out_dir / "kugelaudio_loader_bad_checkpoint.gguf"),
        "missing_semantic": str(out_dir / "kugelaudio_loader_missing_semantic.gguf"),
        "missing_acoustic": str(out_dir / "kugelaudio_loader_missing_acoustic.gguf"),
        "missing_metadata": str(out_dir / "kugelaudio_loader_missing_metadata.gguf"),
    }))


if __name__ == "__main__":
    main()
