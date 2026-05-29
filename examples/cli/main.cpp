// kugelaudio-cli — KugelAudio-first text-to-speech CLI.

#include "audio_io.hpp"
#include "backend.hpp"
#include "model_loader.hpp"
#include "tokenizer.hpp"
#include "vibevoice.h"
#include "vibevoice_tts.hpp"

#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void print_usage(const char* argv0) {
    std::printf(
        "usage: %s [--version|--help|options]\n"
        "\n"
        "options:\n"
        "  --model <path>      path to KugelAudio GGUF (required for the supported path)\n"
        "  --tokenizer <path>  path to tokenizer.gguf (required)\n"
        "  --ref-audio <path>  reference WAV (24 kHz mono, ~5 s) — runtime\n"
        "                      voice cloning. For KugelAudio v1, pass exactly\n"
        "                      one --ref-audio and plain untagged text only.\n"
        "  --text <string>     input text\n"
        "  --text-file <path>  read text from file\n"
        "  --out <path>        output WAV path (default: out.wav)\n"
        "  --max-words-per-chunk N\n"
        "                      enable long-text chunking when N > 0\n"
        "  --overlap-sentences N\n"
        "                      reuse the last N completed sentences from the\n"
        "                      previous chunk as prompt context\n"
        "  --chunking-strategy <heuristic|syntax-aware>\n"
        "                      chunk planning strategy (default heuristic)\n"
        "  --pause-mode <none|punctuation|speaker-aware>\n"
        "                      pause insertion mode for chunked output\n"
        "  --crossfade-ms N    crossfade duration between chunk joins\n"
        "                      (default 30, 0 disables)\n"
        "  --chunk-boundary-cleanup <none|trim-fade>\n"
        "                      opt-in cleanup before stitching: trim excess\n"
        "                      generated silence and fade chunk seams\n"
        "  --chunk-boundary-leading-silence-ms N\n"
        "                      leading silence to keep on chunks after first\n"
        "                      when cleanup is enabled (default 200)\n"
        "  --chunk-boundary-trailing-silence-ms N\n"
        "                      trailing silence to keep on chunks before last\n"
        "                      when cleanup is enabled (default 300)\n"
        "  --chunk-boundary-fade-ms N\n"
        "                      seam fade duration when cleanup is enabled\n"
        "                      (default 15)\n"
        "  --chunk-eos-guard-frames N\n"
        "                      opt-in diagnostic: suppress the first N EOS/\n"
        "                      speech_end tokens after a chunk has emitted audio\n"
        "                      (default 0)\n"
        "  --text-end-padding <none|ellipsis>\n"
        "                      opt-in diagnostic: normalize generated text ends\n"
        "                      to terminal ASCII ... while preserving source text\n"
        "  --chunk-continuity <none|single-sequence|segmented-state>\n"
        "                      experimental seam continuity mode\n"
        "  --continuity-tail-ms N\n"
        "                      reserved for retired continuity experiments\n"
        "  --max-frames N      cap speech frames (default 200)\n"
        "  --steps N           DPM-Solver inference steps (default 20)\n"
        "  --cfg X             classifier-free guidance scale (default 1.3,\n"
        "                      1.0 disables CFG)\n"
        "  --seed N            RNG seed for noise (default random)\n"
        "  --threads N         CPU thread count (default auto)\n"
        "  --latent-refine-strength X\n"
        "                      opt-in diagnostic latent img2img refinement strength\n"
        "                      (0 disables; try 0.10-0.40)\n"
        "  --latent-refine-steps N\n"
        "                      optional cap/override for refinement reverse steps\n"
        "  --final-decoder-backend <auto|cpu|stream|active>\n"
        "                      explicit final latent->waveform decode backend.\n"
        "                      `cpu` is a hybrid fallback for GPU runs;\n"
        "                      `stream` keeps the active backend but chunks\n"
        "                      the final decoder; `active` forces one-shot\n"
        "                      decode on the active backend.\n"
        "  --verbose           print per-frame progress\n"
        "\n"
        "%s\n",
        argv0, vv_version());
}

bool slurp(const std::string& path, std::string* out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out->resize(n > 0 ? static_cast<size_t>(n) : 0);
    if (n > 0) std::fread(out->data(), 1, n, f);
    std::fclose(f);
    return true;
}

const char* getenv_compat(const char* kugelaudio_name, const char* vibevoice_name) {
    if (const char* v = std::getenv(kugelaudio_name); v && *v) return v;
    return std::getenv(vibevoice_name);
}

bool env_truthy(const char* v) {
    return v && *v && std::string(v) != "0" && std::string(v) != "false";
}

int cmd_tts(int argc, char** argv) {
    std::string model_path, tok_path, voice_path;
    std::vector<std::string> ref_audio;
    std::string text, text_file, out_path = "out.wav";
    int   max_frames = 200, steps = 20;
    int   max_words_per_chunk = 0;
    int   overlap_sentences = 0;
    int   crossfade_ms = 30;
    bool  chunk_boundary_cleanup = false;
    int   chunk_boundary_leading_silence_ms = 200;
    int   chunk_boundary_trailing_silence_ms = 300;
    int   chunk_boundary_fade_ms = 15;
    int   chunk_eos_guard_frames = 0;
    int   continuity_tail_ms = 1200;
    vv::TextEndPaddingMode text_end_padding = vv::TextEndPaddingMode::None;
    float cfg_scale = 1.3f;
    float latent_refine_strength = 0.0f;
    int   latent_refine_steps = 0;
    uint32_t seed = 0;
    bool  verbose = false;
    int   n_threads = 0;
    vv::ChunkingStrategy chunking_strategy = vv::ChunkingStrategy::Heuristic;
    vv::ChunkPauseMode pause_mode = vv::ChunkPauseMode::Punctuation;
    vv::ChunkContinuityMode chunk_continuity = vv::ChunkContinuityMode::None;
    vv::FinalDecoderBackend final_decoder_backend = vv::FinalDecoderBackend::Auto;
    const bool allow_retired_continuity = env_truthy(getenv_compat("KUGELAUDIO_ENABLE_RETIRED_CONTINUITY", "VIBEVOICE_ENABLE_RETIRED_CONTINUITY"));

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"      && (i + 1 < argc)) { model_path = argv[++i]; }
        else if (a == "--tokenizer"  && (i + 1 < argc)) { tok_path   = argv[++i]; }
        else if (a == "--voice"      && (i + 1 < argc)) { voice_path = argv[++i]; }
        else if (a == "--ref-audio"  && (i + 1 < argc)) { ref_audio.emplace_back(argv[++i]); }
        else if (a == "--text"       && (i + 1 < argc)) { text       = argv[++i]; }
        else if (a == "--text-file"  && (i + 1 < argc)) { text_file  = argv[++i]; }
        else if (a == "--out"        && (i + 1 < argc)) { out_path   = argv[++i]; }
        else if (a == "--max-words-per-chunk" && (i + 1 < argc)) { max_words_per_chunk = std::atoi(argv[++i]); }
        else if (a == "--overlap-sentences" && (i + 1 < argc)) { overlap_sentences = std::atoi(argv[++i]); }
        else if (a == "--chunking-strategy" && (i + 1 < argc)) {
            std::string v = argv[++i];
            if (v == "heuristic") chunking_strategy = vv::ChunkingStrategy::Heuristic;
            else if (v == "syntax-aware") chunking_strategy = vv::ChunkingStrategy::SyntaxAware;
            else {
                std::fprintf(stderr, "tts: invalid --chunking-strategy %s (expected heuristic|syntax-aware)\n", v.c_str());
                return 1;
            }
        }
        else if (a == "--pause-mode" && (i + 1 < argc)) {
            std::string v = argv[++i];
            if (v == "none") pause_mode = vv::ChunkPauseMode::None;
            else if (v == "punctuation") pause_mode = vv::ChunkPauseMode::Punctuation;
            else if (v == "speaker-aware") pause_mode = vv::ChunkPauseMode::SpeakerAware;
            else {
                std::fprintf(stderr, "tts: invalid --pause-mode %s (expected none|punctuation|speaker-aware)\n", v.c_str());
                return 1;
            }
        }
        else if (a == "--crossfade-ms" && (i + 1 < argc)) { crossfade_ms = std::atoi(argv[++i]); }
        else if (a == "--chunk-boundary-cleanup" && (i + 1 < argc)) {
            std::string v = argv[++i];
            if (v == "none") chunk_boundary_cleanup = false;
            else if (v == "trim-fade") chunk_boundary_cleanup = true;
            else {
                std::fprintf(stderr, "tts: invalid --chunk-boundary-cleanup %s (expected none|trim-fade)\n", v.c_str());
                return 1;
            }
        }
        else if (a == "--chunk-boundary-leading-silence-ms" && (i + 1 < argc)) { chunk_boundary_leading_silence_ms = std::atoi(argv[++i]); }
        else if (a == "--chunk-boundary-trailing-silence-ms" && (i + 1 < argc)) { chunk_boundary_trailing_silence_ms = std::atoi(argv[++i]); }
        else if (a == "--chunk-boundary-fade-ms" && (i + 1 < argc)) { chunk_boundary_fade_ms = std::atoi(argv[++i]); }
        else if (a == "--chunk-eos-guard-frames" && (i + 1 < argc)) { chunk_eos_guard_frames = std::atoi(argv[++i]); }
        else if (a == "--text-end-padding" && (i + 1 < argc)) {
            std::string v = argv[++i];
            if (v == "none") text_end_padding = vv::TextEndPaddingMode::None;
            else if (v == "ellipsis") text_end_padding = vv::TextEndPaddingMode::Ellipsis;
            else {
                std::fprintf(stderr, "tts: invalid --text-end-padding %s (expected none|ellipsis)\n", v.c_str());
                return 1;
            }
        }
        else if (a == "--chunk-continuity" && (i + 1 < argc)) {
            std::string v = argv[++i];
            if (v == "none") chunk_continuity = vv::ChunkContinuityMode::None;
            else if (v == "single-sequence") chunk_continuity = vv::ChunkContinuityMode::SingleSequence;
            else if (v == "segmented-state") chunk_continuity = vv::ChunkContinuityMode::SegmentedState;
            else if (v == "tail-reference") {
                if (!allow_retired_continuity) {
                    std::fprintf(stderr, "tts: --chunk-continuity tail-reference is retired because it feeds decoded waveform noise back into later chunks; use none, single-sequence, or segmented-state\n");
                    return 1;
                }
                chunk_continuity = vv::ChunkContinuityMode::TailReference;
            } else if (v == "clean-tail-reference") {
                if (!allow_retired_continuity) {
                    std::fprintf(stderr, "tts: --chunk-continuity clean-tail-reference is retired because sanitized generated waveform feedback still caused noise and speaker drift; use none, single-sequence, or segmented-state\n");
                    return 1;
                }
                chunk_continuity = vv::ChunkContinuityMode::CleanTailReference;
            } else if (v == "latent-prefix") {
                if (!allow_retired_continuity) {
                    std::fprintf(stderr, "tts: --chunk-continuity latent-prefix is retired because listening showed worse noise and intonation drift; use none, single-sequence, or segmented-state\n");
                    return 1;
                }
                chunk_continuity = vv::ChunkContinuityMode::LatentPrefix;
            } else if (v == "prompt-instruction") {
                if (!allow_retired_continuity) {
                    std::fprintf(stderr, "tts: --chunk-continuity prompt-instruction is retired because listening showed speaker identity drift; use none, single-sequence, or segmented-state\n");
                    return 1;
                }
                chunk_continuity = vv::ChunkContinuityMode::PromptInstruction;
            } else {
                std::fprintf(stderr, "tts: invalid --chunk-continuity %s (expected none|single-sequence|segmented-state)\n", v.c_str());
                return 1;
            }
        }
        else if (a == "--continuity-tail-ms" && (i + 1 < argc)) { continuity_tail_ms = std::atoi(argv[++i]); }
        else if (a == "--max-frames" && (i + 1 < argc)) { max_frames = std::atoi(argv[++i]); }
        else if (a == "--steps"      && (i + 1 < argc)) { steps      = std::atoi(argv[++i]); }
        else if (a == "--seed"       && (i + 1 < argc)) { seed       = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10)); }
        else if (a == "--cfg"        && (i + 1 < argc)) { cfg_scale = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--latent-refine-strength" && (i + 1 < argc)) { latent_refine_strength = static_cast<float>(std::atof(argv[++i])); }
        else if (a == "--latent-refine-steps" && (i + 1 < argc)) { latent_refine_steps = std::atoi(argv[++i]); }
        else if (a == "--threads"    && (i + 1 < argc)) { n_threads  = std::atoi(argv[++i]); }
        else if (a == "--final-decoder-backend" && (i + 1 < argc)) {
            std::string v = argv[++i];
            if (v == "auto") final_decoder_backend = vv::FinalDecoderBackend::Auto;
            else if (v == "cpu") final_decoder_backend = vv::FinalDecoderBackend::Cpu;
            else if (v == "stream") final_decoder_backend = vv::FinalDecoderBackend::StreamActive;
            else if (v == "active") final_decoder_backend = vv::FinalDecoderBackend::Active;
            else {
                std::fprintf(stderr, "tts: invalid --final-decoder-backend %s (expected auto|cpu|stream|active)\n", v.c_str());
                return 1;
            }
        }
        else if (a == "--verbose")                       { verbose = true; }
        else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else {
            std::fprintf(stderr, "tts: unknown arg: %s\n", a.c_str());
            return 1;
        }
    }

    if (model_path.empty() || tok_path.empty()) {
        std::fprintf(stderr, "tts: --model and --tokenizer are required\n");
        return 1;
    }

    if (n_threads > 0) {
        if (ggml_backend_is_cpu(vv::backend())) {
            ggml_backend_cpu_set_n_threads(vv::backend(), n_threads);
            std::fprintf(stderr, "tts: cpu threads = %d\n", n_threads);
        } else {
            std::fprintf(stderr,
                         "tts: ignoring --threads=%d because backend=%s is not CPU\n",
                         n_threads, vv::backend_name());
        }
    }
    if (crossfade_ms < 0) {
        std::fprintf(stderr, "tts: --crossfade-ms must be >= 0\n");
        return 1;
    }
    if (overlap_sentences < 0) {
        std::fprintf(stderr, "tts: --overlap-sentences must be >= 0\n");
        return 1;
    }
    if (continuity_tail_ms < 0) {
        std::fprintf(stderr, "tts: --continuity-tail-ms must be >= 0\n");
        return 1;
    }
    if (chunk_boundary_leading_silence_ms < 0 || chunk_boundary_trailing_silence_ms < 0 || chunk_boundary_fade_ms < 0) {
        std::fprintf(stderr, "tts: chunk boundary cleanup values must be >= 0\n");
        return 1;
    }
    if (chunk_eos_guard_frames < 0) {
        std::fprintf(stderr, "tts: --chunk-eos-guard-frames must be >= 0\n");
        return 1;
    }

    if (text.empty()) {
        if (text_file.empty()) {
            std::fprintf(stderr, "tts: provide --text or --text-file\n");
            return 1;
        }
        if (!slurp(text_file, &text)) {
            std::fprintf(stderr, "tts: failed to read %s\n", text_file.c_str());
            return 1;
        }
    }

    std::fprintf(stderr, "cli tts: loading model %s\n", model_path.c_str());
    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_path, &model)) {
        std::fprintf(stderr, "tts: failed to load model\n");
        return 2;
    }

    std::fprintf(stderr, "cli tts: loading tokenizer %s\n", tok_path.c_str());
    if (!model.tokenizer.load_from_file(tok_path)) {
        std::fprintf(stderr, "tts: failed to load tokenizer\n");
        return 3;
    }

    // Validate inputs against the loaded model's variant. The gguf
    // already says which kind of conditioning it expects; the CLI is
    // a thin wrapper around that.
    const bool uses_raw_reference_tts = (model.variant == "1.5b");
    const bool is_kugelaudio = model.loader.has_key("kugelaudio.architecture");

    // Log model/load config for operator diagnostics.
    const char* quant_hint = "unknown";
    if (model_path.find("q8_0") != std::string::npos) quant_hint = "q8_0";
    else if (model_path.find("f16") != std::string::npos) quant_hint = "f16";
    else if (model_path.find("fp32") != std::string::npos) quant_hint = "fp32";
    std::fprintf(stderr, "tts: model_family=%s internal_variant=%s quantization_hint=%s\n",
                 is_kugelaudio ? "kugelaudio" : "legacy", model.variant.c_str(), quant_hint);

    if (!voice_path.empty()) {
        std::fprintf(stderr,
                     "tts: --voice has been removed; pre-baked voice.gguf conditioning is no longer supported on the published CLI. "
                     "Use exactly one --ref-audio with a raw-reference TTS model instead.\n");
        return 1;
    }
    if (!uses_raw_reference_tts) {
        std::fprintf(stderr,
                     "tts: unsupported model for this CLI: pre-baked voice.gguf conditioning has been removed. "
                     "Use a raw-reference TTS model instead.\n");
        return 1;
    }
    if (uses_raw_reference_tts && ref_audio.empty()) {
        if (is_kugelaudio) {
            std::fprintf(stderr,
                         "tts: KugelAudio v1 requires exactly one --ref-audio "
                         "(raw 24 kHz mono WAV) and plain untagged text.\n");
        } else {
            std::fprintf(stderr,
                         "tts: this legacy raw-reference TTS model requires at least one --ref-audio "
                         "(raw 24 kHz mono WAV). Multi-speaker dialog and pre-baked voice flows "
                         "are legacy-only and are not part of the KugelAudio publish surface.\n");
        }
        return 1;
    }
    if (!uses_raw_reference_tts && !ref_audio.empty()) {
        std::fprintf(stderr,
                     "tts: --ref-audio only applies to raw-reference TTS models; this "
                     "model is internal_variant=%s. Use the matching legacy voice-conditioning flow "
                     "instead.\n", model.variant.c_str());
        return 1;
    }

    if (is_kugelaudio) {
        vv::VibeVoiceTTSParams gate;
        gate.ref_audio_paths = ref_audio;
        std::string gate_error;
        if (!vv::detail::validate_kugelaudio_single_speaker_request(text, gate, &gate_error)) {
            std::fprintf(stderr, "tts: %s\n", gate_error.c_str());
            return 1;
        }
    }

    vv::VibeVoiceTTSParams p;
    p.voice             = nullptr;
    p.ref_audio_paths   = ref_audio;
    p.max_speech_frames = max_frames;
    p.n_diffusion_steps = steps;
    p.cfg_scale         = cfg_scale;
    p.seed                = seed;
    p.verbose             = verbose;
    p.max_words_per_chunk = max_words_per_chunk;
    p.overlap_sentences   = overlap_sentences;
    p.chunking_strategy   = chunking_strategy;
    p.pause_mode          = pause_mode;
    p.crossfade_ms        = crossfade_ms;
    p.chunk_boundary_cleanup = chunk_boundary_cleanup;
    p.chunk_boundary_leading_silence_ms = chunk_boundary_leading_silence_ms;
    p.chunk_boundary_trailing_silence_ms = chunk_boundary_trailing_silence_ms;
    p.chunk_boundary_fade_ms = chunk_boundary_fade_ms;
    p.chunk_eos_guard_frames = chunk_eos_guard_frames;
    p.text_end_padding = text_end_padding;
    p.chunk_continuity    = chunk_continuity;
    p.continuity_tail_ms  = continuity_tail_ms;
    p.final_decoder_backend = final_decoder_backend;
    p.latent_refine_strength = latent_refine_strength;
    p.latent_refine_steps = latent_refine_steps;
    if (is_kugelaudio && std::string(quant_hint) == "f16") {
        p.cast_step_embed_f16 = 1;
    }

    std::fprintf(stderr,
                 "tts: generation_settings frames=%d steps=%d cfg=%.2f seed=%u "
                 "conditioning=%s ref_count=%zu chunk_words=%d overlap_sentences=%d chunking_strategy=%s pause_mode=%s crossfade_ms=%d chunk_boundary_cleanup=%s chunk_boundary_leading_ms=%d chunk_boundary_trailing_ms=%d chunk_boundary_fade_ms=%d chunk_eos_guard_frames=%d text_end_padding=%s chunk_continuity=%s continuity_tail_ms=%d final_decoder=%s step_embed_f16=%s latent_refine_strength=%.3f latent_refine_steps=%d\n",
                 max_frames, steps, cfg_scale, seed,
                 "raw_reference",
                 ref_audio.size(),
                 max_words_per_chunk,
                 overlap_sentences,
                 vv::chunking_strategy_name(chunking_strategy),
                 vv::chunk_pause_mode_name(pause_mode),
                 crossfade_ms,
                 chunk_boundary_cleanup ? "trim-fade" : "none",
                 chunk_boundary_leading_silence_ms,
                 chunk_boundary_trailing_silence_ms,
                 chunk_boundary_fade_ms,
                 chunk_eos_guard_frames,
                 text_end_padding == vv::TextEndPaddingMode::Ellipsis ? "ellipsis" : "none",
                 vv::chunk_continuity_mode_name(chunk_continuity),
                 continuity_tail_ms,
                 final_decoder_backend == vv::FinalDecoderBackend::Cpu ? "cpu" :
                 (final_decoder_backend == vv::FinalDecoderBackend::StreamActive ? "stream" :
                 (final_decoder_backend == vv::FinalDecoderBackend::Active ? "active" : "auto")),
                 p.cast_step_embed_f16 > 0 ? "on" : (p.cast_step_embed_f16 == 0 ? "off" : "auto"),
                 static_cast<double>(p.latent_refine_strength), p.latent_refine_steps);

    std::vector<float> samples;
    int rc = vv::vibevoice_tts_generate(&model, text, p, &samples);
    if (rc != 0) {
        std::fprintf(stderr, "tts: generate failed (rc=%d)\n", rc);
        return 4;
    }
    std::fprintf(stderr, "tts: generated %zu samples (%.2fs at %d Hz)\n",
                 samples.size(),
                 static_cast<double>(samples.size()) / model.cfg.sample_rate,
                 model.cfg.sample_rate);

    vv_audio out;
    out.samples     = samples.data();
    out.n_samples   = samples.size();
    out.sample_rate = model.cfg.sample_rate;
    out.channels    = 1;
    if (vv_save_wav(out_path.c_str(), &out) != VV_OK) {
        std::fprintf(stderr, "tts: failed to write %s\n", out_path.c_str());
        return 5;
    }
    std::fprintf(stderr, "tts: wrote %s\n", out_path.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }
    std::string arg1 = argv[1];
    if (arg1 == "-h" || arg1 == "--help") {
        print_usage(argv[0]);
        return 0;
    }
    if (arg1 == "-v" || arg1 == "--version") {
        std::printf("%s\n", vv_version());
        return 0;
    }
    return cmd_tts(argc, argv);
}
