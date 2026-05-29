#ifndef VIBEVOICE_TTS_HPP
#define VIBEVOICE_TTS_HPP

// VibeVoice realtime-TTS orchestrator.
//
// Loads everything from a single .gguf produced by
// scripts/convert_vibevoice_to_gguf.py and exposes a `generate(text)` call
// that returns 24 kHz mono audio.
//
// Architecture (mirrors microsoft/VibeVoice-Realtime-0.5B):
//   text_ids
//     -> embed via lm.tok_embd
//     -> language_model (lower 4 layers, no final norm)         [stack 1]
//     -> spliced into TTS LM tail + tts_input_types[1] (text)
//     -> tts_language_model (upper 20 layers, with final norm)  [stack 2]
//     -> last hidden = condition for diffusion head
//     -> DPM-Solver++ samples a 64-D speech latent
//     -> acoustic decoder produces ~1600 samples per latent
//     -> acoustic_connector(latent) becomes next-step input embed
//     -> stack 2 again with type=speech, then EOS classifier check
//     -> repeat until EOS or max_speech_frames
//
// v1 limitations:
//   - no CFG (cfg_scale = 1.0; only the positive condition is sampled)
//   - no voice prompt (initial KV caches start empty); without a learned
//     voice the generated audio will be incoherent. The wiring is correct;
//     a follow-up will add voice-cache loading.

#include "acoustic_tokenizer.hpp"
#include "diffusion_head.hpp"
#include "dpm_solver.hpp"
#include "kugelaudio_chunking.hpp"
#include "model_loader.hpp"
#include "qwen2.hpp"
#include "tokenizer.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vv {

struct VibeVoiceConfig {
    // LM (Qwen2)
    int     hidden        = 0;
    int     n_layers_lm   = 0;
    int     n_layers_tlm  = 0;
    int     n_heads       = 0;
    int     n_kv_heads    = 0;
    int     head_dim      = 0;
    int     vocab_size    = 0;
    float   rope_theta    = 1.0e6f;
    float   rms_norm_eps  = 1.0e-6f;
    // Diffusion head
    int     latent        = 64;
    int     head_layers   = 4;
    float   ffn_ratio     = 3.0f;
    int     freq_size     = 256;
    // Acoustic decoder / conditioning
    int     vae_dim       = 64;
    AcousticConfig acoustic;
    float   acoustic_fix_std = 0.0f;
    std::string acoustic_std_dist_type = "none";
    // Audio
    int     sample_rate   = 24000;
    // Speech latent normalization
    float   speech_scaling = 1.0f;
    float   speech_bias    = 0.0f;
};

enum class FinalDecoderBackend {
    Auto,
    Cpu,
    StreamActive,
    Active,
};

struct CpuDecoderShadow {
    struct ggml_context*  ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    DecoderWeights        at_dec;

    CpuDecoderShadow() = default;
    CpuDecoderShadow(const CpuDecoderShadow&) = delete;
    CpuDecoderShadow& operator=(const CpuDecoderShadow&) = delete;
    ~CpuDecoderShadow();
    void free();
};

struct VibeVoiceWeights {
    // ---- LM ----
    struct ggml_tensor*               lm_tok_embd  = nullptr;
    std::vector<Qwen2LayerWeights>    lm_layers;             // size n_layers_lm
    std::vector<Qwen2LayerWeights>    tlm_layers;            // size n_layers_tlm
    struct ggml_tensor*               tlm_output_norm = nullptr;
    struct ggml_tensor*               tts_input_types = nullptr;  // [hidden, 2]

    // ---- connector / EOS ----
    struct ggml_tensor* ac_fc1_w = nullptr, *ac_fc1_b = nullptr;
    struct ggml_tensor* ac_norm  = nullptr;
    struct ggml_tensor* ac_fc2_w = nullptr, *ac_fc2_b = nullptr;
    struct ggml_tensor* eos_fc1_w = nullptr, *eos_fc1_b = nullptr;
    struct ggml_tensor* eos_fc2_w = nullptr, *eos_fc2_b = nullptr;

    // ---- diffusion head ----
    DiffusionHeadWeights dh;

    // ---- acoustic decoder ----
    DecoderWeights at_dec;
};

// Per-layer KV cache stored on the CPU as raw float buffers. Each layer
// holds k, v of shape [head_dim, n_kv_heads, past_len, B=1] in ggml's
// convention. Shared by both the TTS and ASR orchestrators.
struct LayerKV {
    std::vector<float> k;
    std::vector<float> v;
    int                past_len = 0;
};

struct VibeVoiceModel {
    ModelLoader      loader;
    VibeVoiceConfig  cfg;
    VibeVoiceWeights w;
    Tokenizer        tokenizer;        // optional, set externally

    // Set during vibevoice_load. Some legacy/internal compatibility variants
    // still use historical names even though the published KugelAudio surface
    // is now described by runtime_path logs rather than variant labels.
    std::string      variant;

    // ---- ASR-specific weights (only populated when variant == "asr-7b") ----
    EncoderWeights   at_enc;
    EncoderWeights   st_enc;
    AcousticConfig   semantic_cfg;
    int              semantic_vae_dim = 128;
    struct ggml_tensor* sc_fc1_w  = nullptr; struct ggml_tensor* sc_fc1_b = nullptr;
    struct ggml_tensor* sc_norm   = nullptr;
    struct ggml_tensor* sc_fc2_w  = nullptr; struct ggml_tensor* sc_fc2_b = nullptr;
    struct ggml_tensor* lm_head   = nullptr;

    // Optional CPU shadow for explicit hybrid Vulkan->CPU final decode.
    CpuDecoderShadow cpu_decoder_shadow;
};

bool vibevoice_load(const std::string& gguf_path, VibeVoiceModel* out);

// A pre-baked voice prompt (system prompt + voice audio prefix) loaded from
// `convert_voice_to_gguf.py` output. Without this the orchestrator runs
// without speaker context and produces low-amplitude/incoherent audio.
struct VibeVoiceVoice {
    int                  seq_lm  = 0;
    int                  seq_tlm = 0;
    std::vector<LayerKV> kv_lm;
    std::vector<LayerKV> kv_tlm;
    std::vector<float>   tlm_last_hidden;       // [hidden]

    // Negative branch (for classifier-free guidance). Optional — older
    // voice .gguf files won't have these populated.
    bool                 has_neg     = false;
    int                  seq_neg_lm  = 0;
    int                  seq_neg_tlm = 0;
    std::vector<LayerKV> kv_neg_lm;
    std::vector<LayerKV> kv_neg_tlm;
    std::vector<float>   neg_tlm_last_hidden;   // [hidden]
};

bool vibevoice_voice_load(const std::string&     path,
                          const VibeVoiceModel&  model,
                          VibeVoiceVoice*        out);

struct VibeVoiceTTSParams {
    // Legacy compatibility conditioning: a pre-baked voice gguf wrapped in
    // VibeVoiceVoice (load with vibevoice_voice_load). Ignored when the
    // model uses raw-reference conditioning.
    const VibeVoiceVoice* voice = nullptr;

    // Raw-reference conditioning surface.
    //
    // The legacy raw-reference compatibility path supports one reference WAV
    // per speaker and speaker-tagged dialog text. That behavior is retained
    // for legacy non-KugelAudio paths only.
    //
    // KugelAudio v1 acceptance is narrower by design: pass exactly one raw
    // reference WAV and plain untagged text. Wider request shapes (for
    // example multi-speaker or alternative conditioning layouts) are deferred
    // and should widen through `detail::KugelAudioRequestPolicy`, not by
    // changing this v1 call surface ad hoc.
    std::vector<std::string> ref_audio_paths;

    int      max_speech_frames = 200;
    int      min_speech_frames = 0;
    float    cfg_scale         = 1.3f;
    int      n_diffusion_steps = 20;
    uint32_t seed              = 0;
    bool     verbose             = false;
    int      max_words_per_chunk = 0;
    int      overlap_sentences   = 0;
    ChunkingStrategy chunking_strategy = ChunkingStrategy::Heuristic;
    ChunkPauseMode pause_mode    = ChunkPauseMode::Punctuation;
    int      crossfade_ms        = 30;
    bool     chunk_boundary_cleanup = false;
    int      chunk_boundary_leading_silence_ms = 200;
    int      chunk_boundary_trailing_silence_ms = 300;
    int      chunk_boundary_fade_ms = 15;
    int      chunk_eos_guard_frames = 0;
    TextEndPaddingMode text_end_padding = TextEndPaddingMode::None;
    ChunkContinuityMode chunk_continuity = ChunkContinuityMode::None;
    int      continuity_tail_ms  = 1200;
    std::string prompt_continuity_instruction;
    FinalDecoderBackend final_decoder_backend = FinalDecoderBackend::Auto;

    // -1 = runtime/env default, 0 = disabled, 1 = enabled. The CLI enables
    // this for f16 KugelAudio artifacts because canonical feedback appears to
    // round generated connector embeddings before feeding them back to the LM.
    int      cast_step_embed_f16 = -1;

    // Opt-in diagnostic: img2img-style second pass over generated speech
    // latents. 0 disables. Strength maps to the tail fraction of the existing
    // DPM schedule; steps can cap/override the number of reverse steps.
    float    latent_refine_strength = 0.0f;
    int      latent_refine_steps    = 0;
};

// Generate audio for `text`. Dispatches on `model->variant`:
//   * legacy pre-baked-voice compatibility path -> uses `p.voice`
//     (pre-baked voice gguf state; not KugelAudio v1 acceptance).
//   * legacy/raw-reference compatibility path -> uses `p.ref_audio_paths`
//     (raw reference WAV(s)). For KugelAudio v1, exactly one raw reference
//     audio input is accepted; wider shapes remain deferred behind the
//     request-policy seam.
// Output samples are 24 kHz mono float32. Returns 0 on success.
int vibevoice_tts_generate(VibeVoiceModel*           model,
                           const std::string&        text,
                           const VibeVoiceTTSParams& p,
                           std::vector<float>*       samples);

namespace detail {
// Policy seam for deferred KugelAudio features. V1 keeps the profile narrow,
// but future work (multi-reference, speaker-tagged dialog, alternative
// conditioning shapes) should widen behavior by introducing a new profile,
// not by scattering ad hoc conditionals across CLI/runtime/CAPI callsites.
struct KugelAudioRequestPolicy {
    bool        allow_pre_baked_voice      = false;
    std::size_t min_ref_audio_inputs       = 1;
    std::size_t max_ref_audio_inputs       = 1;
    bool        allow_speaker_tagged_dialog = false;
    const char* supported_shape            = nullptr;
};

const KugelAudioRequestPolicy& kugelaudio_v1_request_policy();
bool validate_kugelaudio_request(const std::string& text,
                                 const VibeVoiceTTSParams& p,
                                 const KugelAudioRequestPolicy& policy,
                                 std::string* error);
bool validate_kugelaudio_single_speaker_request(const std::string& text,
                                                const VibeVoiceTTSParams& p,
                                                std::string* error);
std::string build_kugelaudio_prompt_single_speaker_for_test(int vae_tok_len,
                                                            const std::string& text);
std::vector<int32_t> build_kugelaudio_inserted_speech_tokens_for_test(int vae_tok_len);
std::vector<int32_t> build_kugelaudio_negative_seed_tokens_for_test();
void build_kugelaudio_prompt_input_ids_for_test(const Tokenizer& tokenizer,
                                                int vae_tok_len,
                                                const std::string& text,
                                                std::vector<int32_t>* input_ids,
                                                std::vector<int>* pad_positions);
std::vector<int32_t> kugelaudio_valid_speech_token_ids_for_test();
float kugelaudio_speech_end_penalty_for_test();
void apply_kugelaudio_speech_end_penalty_for_test(std::vector<float>* logits);
std::vector<float> run_speech_connector_for_test(const VibeVoiceConfig& cfg,
                                                 const VibeVoiceWeights& w,
                                                 const float* x,
                                                 int batch);
DPMSolverConfig kugelaudio_solver_config_for_test(int requested_steps);
int select_kugelaudio_speech_token_from_logits_for_test(const std::vector<float>& logits);
bool kugelaudio_token_requires_cfg_reset_for_test(int32_t token_id);
bool kugelaudio_token_stops_generation_for_test(int32_t token_id);
int kugelaudio_speech_start_id_for_test();
int kugelaudio_pause_duration_ms_for_test(const std::string& left_text,
                                          const std::string& right_text,
                                          ChunkPauseMode pause_mode);
int kugelaudio_speech_end_id_for_test();
int kugelaudio_speech_diffusion_id_for_test();
int kugelaudio_eos_id_for_test();
int kugelaudio_image_pad_id_for_test();
}

}  // namespace vv

#endif  // VIBEVOICE_TTS_HPP
