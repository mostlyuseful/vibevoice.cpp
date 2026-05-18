#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "torch",
#   "transformers",
#   "librosa",
#   "soundfile",
# ]
# ///
"""Dump canonical KugelAudio conditioning/prompt/prefill/first-diffusion tensors.

This script intentionally mirrors the current canonical PyTorch path for the
supported v1 raw-reference single-speaker flow and writes normalized row-major
float32 / int32 dumps that can be compared to the ggml runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-repo", default="../kugelaudio-open")
    p.add_argument("--model", required=True)
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--dump-dir", default=None)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--speech-end-penalty", type=float, default=1.5)
    p.add_argument("--acoustic-mode", choices=["sample", "mean"], default="sample")
    p.add_argument("--stop-after", choices=["conditioning", "prefill", "first_diffusion"], default="first_diffusion")
    return p.parse_args()


def ensure_imports(canonical_repo: Path) -> None:
    src = canonical_repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def dump_meta(path: Path, stage: str, shape: list[int], dtype: str, semantic: str, source: str) -> None:
    path.write_text(json.dumps({
        "stage": stage,
        "shape": shape,
        "dtype": dtype,
        "layout": "row_major",
        "semantic": semantic,
        "source": source,
    }, indent=2) + "\n", encoding="utf-8")


def dump_f32(dir_path: Path, stage: str, tensor: torch.Tensor | np.ndarray, semantic: str) -> None:
    arr = tensor.detach().cpu().float().contiguous().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor, dtype=np.float32)
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    (dir_path / f"{stage}.bin").write_bytes(arr.tobytes(order="C"))
    dump_meta(dir_path / f"{stage}.json", stage, list(arr.shape), "float32", semantic, "canonical")


def dump_i32(dir_path: Path, stage: str, tensor: torch.Tensor | np.ndarray | list[int], semantic: str) -> None:
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().to(dtype=torch.int32).contiguous().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.int32)
    arr = np.ascontiguousarray(arr)
    (dir_path / f"{stage}.bin").write_bytes(arr.tobytes(order="C"))
    dump_meta(dir_path / f"{stage}.json", stage, list(arr.shape), "int32", semantic, "canonical")


def to_row_major_tc(x: torch.Tensor) -> torch.Tensor:
    # Canonical feature tensors are generally [B, T, C]. Normalize to [T, C].
    if x.dim() == 3:
        x = x[0]
    if x.dim() == 2:
        return x.contiguous()
    if x.dim() == 1:
        return x.contiguous()
    raise ValueError(f"unsupported tensor rank for row-major dump: {tuple(x.shape)}")


def main() -> int:
    args = parse_args()
    dump_dir = Path(args.dump_dir or Path.cwd() / "canonical-kugelaudio-dumps").resolve()
    dump_dir.mkdir(parents=True, exist_ok=True)

    ensure_imports(Path(args.canonical_repo).resolve())

    from kugelaudio_open.models.kugelaudio_inference import (
        KugelAudioTokenConstraintProcessor,
        _get_cache_tensors,
    )
    from kugelaudio_open.utils.generation import load_model_and_processor

    torch.manual_seed(args.seed)
    model, processor = load_model_and_processor(
        model_name_or_path=args.model,
        device="cpu",
        torch_dtype=torch.float32,
        use_flash_attention=False,
    )
    model.eval()

    inputs = processor(
        text=args.text,
        voice_prompt=args.ref_audio,
        return_tensors="pt",
    )

    text_ids = inputs["text_ids"]
    speech_input_mask = inputs["speech_input_mask"]
    speech_tensors = inputs["speech_tensors"]
    speech_masks = inputs["speech_masks"]

    # Processor-level normalized audio.
    dump_f32(dump_dir, "00_ref_audio_post_norm", speech_tensors.squeeze(0).squeeze(0),
             "processor-normalized 24kHz mono reference audio")
    dump_i32(dump_dir, "09_prompt_input_ids", text_ids.squeeze(0),
             "canonical processor text ids")
    dump_i32(dump_dir, "10_speech_input_mask", speech_input_mask.squeeze(0).to(dtype=torch.int32),
             "mask of prompt positions replaced by speech embeddings")
    pad_positions = speech_input_mask.squeeze(0).nonzero(as_tuple=False).squeeze(-1)
    dump_i32(dump_dir, "10_pad_positions", pad_positions.to(dtype=torch.int32),
             "canonical prompt positions replaced by speech embeddings")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    speech_tensors = speech_tensors.to(device=device, dtype=dtype)
    speech_masks = speech_masks.to(device=device)
    text_ids = text_ids.to(device)
    speech_input_mask = speech_input_mask.to(device)

    with torch.no_grad():
        acoustic_output = model.acoustic_tokenizer.encode(speech_tensors)
        acoustic_raw = getattr(acoustic_output, "mean", acoustic_output)
        dump_f32(dump_dir, "01_acoustic_encoder_out_raw", to_row_major_tc(acoustic_raw),
                 "canonical acoustic encoder output prior to sampling")

        if args.acoustic_mode == "mean":
            acoustic_features = acoustic_output.mean
        else:
            acoustic_features, _ = model.acoustic_tokenizer.sampling(acoustic_output)
        dump_f32(dump_dir, "02_acoustic_features_after_sampling", to_row_major_tc(acoustic_features),
                 f"canonical acoustic features after tokenizer sampling mode={args.acoustic_mode}")

        semantic_output = model.semantic_tokenizer.encode(speech_tensors)
        semantic_features = semantic_output.mean
        dump_f32(dump_dir, "03_semantic_encoder_mean", to_row_major_tc(semantic_features),
                 "canonical semantic encoder mean features")

        acoustic_len = acoustic_features.shape[1]
        semantic_len = semantic_features.shape[1]
        if semantic_len < acoustic_len:
            pad_size = acoustic_len - semantic_len
            semantic_aligned = torch.nn.functional.pad(
                semantic_features, (0, 0, 0, pad_size), mode="constant", value=0
            )
        elif semantic_len > acoustic_len:
            semantic_aligned = semantic_features[:, :acoustic_len, :]
        else:
            semantic_aligned = semantic_features
        dump_f32(dump_dir, "04_semantic_aligned", to_row_major_tc(semantic_aligned),
                 "canonical semantic features after pad/truncate to acoustic length")

        if not torch.isnan(model.speech_scaling_factor):
            acoustic_scaled = (acoustic_features + model.speech_bias_factor) * model.speech_scaling_factor
        else:
            acoustic_scaled = acoustic_features
        dump_f32(dump_dir, "05_acoustic_after_scale_bias", to_row_major_tc(acoustic_scaled),
                 "canonical acoustic conditioning features after bias/scale")

        acoustic_embed = model.acoustic_connector(acoustic_scaled)
        dump_f32(dump_dir, "06_acoustic_connector_out", to_row_major_tc(acoustic_embed),
                 "canonical acoustic connector output")

        semantic_embed = model.semantic_connector(semantic_aligned)
        dump_f32(dump_dir, "07_semantic_connector_out", to_row_major_tc(semantic_embed),
                 "canonical semantic connector output")

        speech_embeds = (acoustic_embed + semantic_embed)[speech_masks.cpu()]
        dump_f32(dump_dir, "08_fused_speech_embeds", speech_embeds,
                 "canonical fused speech conditioning embeddings")

        if args.stop_after == "conditioning":
            return 0

        current_ids = text_ids
        attention_mask = torch.ones_like(current_ids)
        speech_start_id = getattr(model.config, "speech_start_id", None) or 151652
        speech_end_id = getattr(model.config, "speech_end_id", None) or 151653
        speech_diffusion_id = getattr(model.config, "speech_diffusion_id", None) or 151654
        eos_token_id = getattr(model.config.decoder_config, "eos_token_id", None) or 151643

        inputs_embeds = model.model.get_input_embeddings()(current_ids)
        dump_f32(dump_dir, "11_prompt_embeds_pre_splice", inputs_embeds[0],
                 "canonical prompt embeddings before speech-feature splice")
        inputs_embeds[speech_input_mask] = speech_embeds
        dump_f32(dump_dir, "12_prompt_embeds_post_splice", inputs_embeds[0],
                 "canonical prompt embeddings after speech-feature splice")

        negative_ids = torch.full((current_ids.shape[0], 1), speech_start_id, dtype=torch.long, device=device)
        negative_attention_mask = torch.ones_like(negative_ids)
        negative_inputs_embeds = model.model.get_input_embeddings()(negative_ids)

        token_constraint = KugelAudioTokenConstraintProcessor(
            [speech_start_id, speech_end_id, speech_diffusion_id, eos_token_id], device=device
        )
        finished = torch.zeros(current_ids.shape[0], dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(current_ids.shape[0], dtype=torch.long, device=device)

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        hidden_last_pos = outputs.last_hidden_state[:, -1, :]
        dump_f32(dump_dir, "13_prefill_hidden_last_pos", hidden_last_pos[0],
                 "canonical positive prefill last hidden state")

        negative_past_key_values = None
        if args.cfg_scale != 1.0:
            neg_outputs = model(
                inputs_embeds=negative_inputs_embeds,
                attention_mask=negative_attention_mask,
                use_cache=True,
                return_dict=True,
            )
            hidden_last_neg = neg_outputs.last_hidden_state[:, -1, :]
            dump_f32(dump_dir, "14_prefill_hidden_last_neg", hidden_last_neg[0],
                     "canonical negative/CFG prefill last hidden state")

        logits = outputs.logits[:, -1, :]
        logits = token_constraint(current_ids, logits)
        if args.speech_end_penalty > 0:
            logits[:, speech_end_id] = logits[:, speech_end_id] - args.speech_end_penalty
        dump_f32(dump_dir, "15_first_logits", logits[0],
                 "canonical first constrained/penalized logits before token selection")
        next_tokens = torch.argmax(logits, dim=-1)
        dump_i32(dump_dir, "16_first_selected_token", next_tokens.to(dtype=torch.int32),
                 "canonical selected first speech-path control token")

        if args.stop_after == "prefill":
            return 0

        first_diffusion_done = False
        inputs_embeds_running = inputs_embeds
        negative_inputs_embeds_running = negative_inputs_embeds
        current_ids_running = current_ids
        attention_mask_running = attention_mask
        negative_ids_running = negative_ids
        negative_attention_mask_running = negative_attention_mask

        for _step in range(args.max_new_tokens):
            outputs = model(
                inputs_embeds=inputs_embeds_running[:, -1:] if current_ids_running.shape[1] > text_ids.shape[1] else inputs_embeds_running,
                attention_mask=attention_mask_running,
                past_key_values=None if current_ids_running.shape[1] == text_ids.shape[1] else past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = token_constraint(current_ids_running, outputs.logits[:, -1, :])
            if args.speech_end_penalty > 0:
                logits[:, speech_end_id] = logits[:, speech_end_id] - args.speech_end_penalty
            next_tokens = torch.argmax(logits, dim=-1)
            next_tokens = torch.where(finished, torch.tensor(eos_token_id, device=device), next_tokens)

            current_ids_running = torch.cat([current_ids_running, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask_running = torch.cat(
                [attention_mask_running, torch.ones((current_ids_running.shape[0], 1), device=device, dtype=attention_mask_running.dtype)],
                dim=-1,
            )
            eos_mask = (next_tokens == eos_token_id) & ~finished
            speech_end_mask = (next_tokens == speech_end_id) & ~finished
            finished = finished | eos_mask | speech_end_mask

            speech_start_mask = (next_tokens == speech_start_id) & ~finished
            if speech_start_mask.any() and args.cfg_scale != 1.0 and negative_past_key_values is not None:
                speech_start_indices = speech_start_mask.nonzero(as_tuple=False).squeeze(-1)
                if speech_start_indices.dim() == 0:
                    speech_start_indices = speech_start_indices.unsqueeze(0)
                for sample_idx in speech_start_indices.tolist():
                    negative_attention_mask_running[sample_idx, :] = 0
                    negative_attention_mask_running[sample_idx, -1] = 1
                    key_caches, value_caches = _get_cache_tensors(negative_past_key_values)
                    for k_cache, v_cache in zip(key_caches, value_caches):
                        k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                        v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()
                    negative_ids_running[sample_idx, -1] = speech_start_id

            next_inputs_embeds = model.model.get_input_embeddings()(next_tokens).unsqueeze(1)
            diffusion_mask = (next_tokens == speech_diffusion_id) & ~finished
            if diffusion_mask.any():
                diffusion_indices = diffusion_mask.nonzero(as_tuple=False).squeeze(-1)
                if diffusion_indices.dim() == 0:
                    diffusion_indices = diffusion_indices.unsqueeze(0)

                if args.cfg_scale != 1.0:
                    if negative_past_key_values is None:
                        neg_outputs = model(
                            inputs_embeds=negative_inputs_embeds_running,
                            attention_mask=negative_attention_mask_running,
                            use_cache=True,
                            return_dict=True,
                        )
                    else:
                        neg_outputs = model(
                            inputs_embeds=negative_inputs_embeds_running[:, -1:],
                            attention_mask=negative_attention_mask_running,
                            past_key_values=negative_past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                    negative_past_key_values = neg_outputs.past_key_values

                    non_diffusion_mask = ~diffusion_mask & ~finished
                    if non_diffusion_mask.any():
                        non_diffusion_indices = non_diffusion_mask.nonzero(as_tuple=False).squeeze(-1)
                        if non_diffusion_indices.dim() == 0:
                            non_diffusion_indices = non_diffusion_indices.unsqueeze(0)
                        key_caches, value_caches = _get_cache_tensors(negative_past_key_values)
                        for sample_idx in non_diffusion_indices.tolist():
                            start_idx = correct_cnt[sample_idx].item()
                            seq_len = negative_attention_mask_running.shape[1]
                            if start_idx + 1 < seq_len - 1:
                                negative_attention_mask_running[sample_idx, start_idx + 1:] = (
                                    negative_attention_mask_running[sample_idx, start_idx:-1].clone()
                                )
                            negative_attention_mask_running[sample_idx, start_idx] = 0
                            for k_cache, v_cache in zip(key_caches, value_caches):
                                if start_idx + 1 < k_cache.shape[2] - 1:
                                    k_cache[sample_idx, :, start_idx + 1:, :] = k_cache[sample_idx, :, start_idx:-1, :].clone()
                                    v_cache[sample_idx, :, start_idx + 1:, :] = v_cache[sample_idx, :, start_idx:-1, :].clone()
                            if start_idx + 1 < negative_ids_running.shape[1] - 1:
                                negative_ids_running[sample_idx, start_idx + 1:] = negative_ids_running[sample_idx, start_idx:-1].clone()
                        correct_cnt[non_diffusion_indices] += 1
                    neg_condition = neg_outputs.last_hidden_state[diffusion_indices, -1, :]
                else:
                    neg_condition = torch.zeros(
                        diffusion_indices.shape[0],
                        model.config.decoder_config.hidden_size,
                        device=device,
                        dtype=dtype,
                    )

                condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                dump_f32(dump_dir, "17_first_diffusion_cond_pos", condition[0],
                         "canonical positive conditioning vector for first diffusion sample")
                dump_f32(dump_dir, "18_first_diffusion_cond_neg", neg_condition[0],
                         "canonical negative conditioning vector for first diffusion sample")
                speech_latents = model.sample_speech_tokens(condition, neg_condition, args.cfg_scale)
                dump_f32(dump_dir, "19_first_diffusion_latent", speech_latents[0],
                         "canonical first diffusion-sampled latent before decode unscale")
                acoustic_embed = model.acoustic_connector(speech_latents.unsqueeze(1))
                diffusion_embeds = acoustic_embed.squeeze(1)
                dump_f32(dump_dir, "20_first_step_embed", diffusion_embeds[0],
                         "canonical next-step embedding produced from first diffusion latent")
                first_diffusion_done = True
                break

            inputs_embeds_running = torch.cat([inputs_embeds_running, next_inputs_embeds], dim=1)
            negative_inputs_embeds_running = torch.cat([negative_inputs_embeds_running, next_inputs_embeds], dim=1)
            negative_attention_mask_running = torch.cat(
                [negative_attention_mask_running, torch.ones((current_ids_running.shape[0], 1), device=device, dtype=negative_attention_mask_running.dtype)],
                dim=-1,
            )
            negative_ids_running = torch.cat([negative_ids_running, next_tokens.unsqueeze(-1)], dim=-1)

        if not first_diffusion_done:
            raise RuntimeError("did not reach a diffusion token within max_new_tokens")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
