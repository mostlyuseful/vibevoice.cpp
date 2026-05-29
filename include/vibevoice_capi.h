// vibevoice_capi.h - flat C ABI for purego / dlopen integration.
//
// This is a *separate* header from vibevoice.h on purpose: it exposes a
// stateless, file-path-oriented surface that matches what
// LocalAI's go-purego backends expect (see backend/go/qwen3-tts-cpp/cpp/).
//
// Lifetime model: a single global engine, one load_model() per process,
// many tts() calls. Legacy/internal ASR entrypoints still exist for
// compatibility, but they are not part of this branch's published KugelAudio
// surface. Mirrors qwen3-tts-cpp closely enough that a purego dlsym lookup and
// `purego.RegisterLibFunc` finds these symbols by name.
//
// Why a separate flat ABI instead of the existing vibevoice.h:
//   * No opaque pointers — purego pinning lifetimes is fiddly.
//   * No callee-allocated buffers — output via WAV path or caller-owned
//     char buffer.
//   * All return codes are int with 0 = success — every purego backend
//     in LocalAI expects exactly that.

#ifndef VIBEVOICE_CAPI_H
#define VIBEVOICE_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Loads the engine. Published KugelAudio use should provide a raw-reference
// TTS model + tokenizer only. Legacy/internal compatibility fields remain:
//   tts_model_path  - raw-reference TTS gguf, required for vv_capi_tts.
//   asr_model_path  - legacy/internal asr-7b gguf, only for vv_capi_asr.
//   tokenizer_path  - tokenizer gguf, required for either loaded path.
//   voice_path      - legacy pre-baked voice gguf compatibility input; not
//                     part of the published KugelAudio surface.
//   n_threads       - 0 → auto-detect.
// Returns 0 on success, non-zero error code otherwise. Idempotent —
// calling twice replaces the engine.
int vv_capi_load(const char* tts_model_path,
                 const char* asr_model_path,
                 const char* tokenizer_path,
                 const char* voice_path,
                 int         n_threads);

// Synthesize `text` into a 24 kHz mono WAV at `dst_wav_path`. The TTS
// path is selected by the loaded model's internal compatibility variant:
//
//   * legacy pre-baked-voice path -> uses `voice_path` (a pre-baked voice
//     gguf). `ref_audio_paths` must be NULL / n_ref_audio == 0.
//   * raw-reference path -> uses `ref_audio_paths` — one WAV per speaker,
//     24 kHz mono. `n_ref_audio_paths` is the number of distinct speakers
//     (>= 1; the dialog in `text` can reference Speaker 0 .. n-1).
//     `voice_path` must be NULL.
//
// `text` is either a plain sentence (single-speaker convenience —
// auto-wrapped as "Speaker 0: ...") or speaker-tagged dialog with one
// "Speaker N:" line per turn (passed through verbatim).
//
// `voice_path` may be NULL if already supplied to vv_capi_load.
// n_diffusion_steps == 0 -> 20, cfg_scale == 0 -> 1.3 (1.0 disables
// CFG), max_speech_frames == 0 -> 200, seed == 0 -> random.
int vv_capi_tts(const char*        text,
                const char*        voice_path,
                const char* const* ref_audio_paths,
                int                n_ref_audio_paths,
                const char*        dst_wav_path,
                int                n_diffusion_steps,
                float              cfg_scale,
                int                max_speech_frames,
                uint32_t           seed);

// Legacy/internal compatibility entrypoint.
// Transcribe `src_wav_path` into a JSON string written into the caller-
// owned `out_json` buffer of size `out_capacity`. The JSON is the same
// shape the model produces, e.g.
//   [{"Start":0.0,"End":2.8,"Speaker":0,"Content":"…"}]
// Returns:
//    > 0 on success — the number of bytes written (excluding the NUL
//                     terminator). The buffer is always NUL-terminated.
//      0 if no transcription was produced.
//    < 0 on error (see vv_status in vibevoice.h for the enum).
//   If out_capacity is smaller than the produced JSON, returns the
//   negative of the required size and writes the prefix that fit.
int vv_capi_asr(const char* src_wav_path,
                char*       out_json,
                size_t      out_capacity,
                int         max_new_tokens);

// Free engine state. Optional — process exit also frees it.
void vv_capi_unload(void);

// Build / version info. Returns a pointer to a static string; do not free.
const char* vv_capi_version(void);

// Deprecated legacy compatibility entrypoint. Voice cloning via the older
// pre-baked-voice + ASR path is not supported; use raw-reference TTS through
// vv_capi_tts instead.
int vv_capi_voice_clone(const char* src_wav_path,
                        const char* dst_voice_gguf_path,
                        int         with_cfg);

#ifdef __cplusplus
}
#endif

#endif  // VIBEVOICE_CAPI_H
