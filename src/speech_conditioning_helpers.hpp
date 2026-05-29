#ifndef SPEECH_CONDITIONING_HELPERS_HPP
#define SPEECH_CONDITIONING_HELPERS_HPP

// Shared speech-conditioning helpers reused by the raw-reference TTS path and
// legacy/internal compatibility code. Implementations currently live in
// src/vibevoice_asr.cpp; this header keeps the reusable declarations in one
// neutral place so the TTS path can depend on them without presenting them as
// an ASR product surface.

#include "acoustic_tokenizer.hpp"

#include <vector>

struct ggml_tensor;

namespace vv::detail {

// Long-form encoder forward, using the same chunking and streaming-cache
// behaviour as the legacy compatibility implementation. Returns false on
// failure; on success `*latents` holds
// `vae_dim * (*T_compressed)` floats and `*T_compressed` is the number of
// compressed frames (= ceil(samples / 3200)).
bool run_encoder_buf(const EncoderWeights& w, const AcousticConfig& cfg,
                     const std::vector<float>& audio,
                     std::vector<float>* latents,
                     int* T_compressed);

// Speech connector: y = fc2(rmsnorm(fc1(x))).
//   x: [latent_dim, T] (column-major / ggml ne[0]=latent_dim, ne[1]=T)
//   y: [hidden,    T]
std::vector<float> run_connector(struct ggml_tensor* fc1_w,
                                 struct ggml_tensor* fc1_b,
                                 struct ggml_tensor* norm_w,
                                 struct ggml_tensor* fc2_w,
                                 struct ggml_tensor* fc2_b,
                                 const std::vector<float>& x,
                                 int latent_dim, int T, int hidden);

// LM head over the final hidden state only. Returns the `vocab_size`-long
// logits vector.
std::vector<float> lm_head_logits_last(struct ggml_tensor* lm_head_w,
                                       const std::vector<float>& hidden_last,
                                       int hidden, int vocab);

// Combine acoustic and semantic conditioning features frame-wise.
// Both inputs must be shaped [hidden * T]. Returns false on mismatch.
bool fuse_conditioning_features(const std::vector<float>& acoustic,
                                const std::vector<float>& semantic,
                                int hidden,
                                int T,
                                std::vector<float>* out);

}  // namespace vv::detail

#endif  // SPEECH_CONDITIONING_HELPERS_HPP
