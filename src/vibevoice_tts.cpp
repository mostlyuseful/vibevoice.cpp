#include "vibevoice_tts.hpp"
#include "speech_conditioning_helpers.hpp"
#include "audio_io.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "rms_norm.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <random>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace vv {

CpuDecoderShadow::~CpuDecoderShadow() { free(); }

void CpuDecoderShadow::free() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    at_dec = DecoderWeights{};
}

// ============================================================================
//  Loaders
// ============================================================================

namespace {

bool clone_tensor_to_cpu(struct ggml_context* ctx,
                         struct ggml_tensor* src,
                         struct ggml_tensor** dst,
                         std::vector<std::pair<const struct ggml_tensor*, struct ggml_tensor*>>* copies) {
    if (!src) {
        *dst = nullptr;
        return true;
    }
    *dst = ggml_dup_tensor(ctx, src);
    if (!*dst) return false;
    copies->emplace_back(src, *dst);
    return true;
}

bool clone_strided_to_cpu(struct ggml_context* ctx,
                          const StridedConvWeights& src,
                          StridedConvWeights* dst,
                          std::vector<std::pair<const struct ggml_tensor*, struct ggml_tensor*>>* copies) {
    dst->stride = src.stride;
    return clone_tensor_to_cpu(ctx, src.kernel, &dst->kernel, copies) &&
           clone_tensor_to_cpu(ctx, src.bias,   &dst->bias,   copies);
}

bool clone_block1d_to_cpu(struct ggml_context* ctx,
                          const Block1DWeights& src,
                          Block1DWeights* dst,
                          std::vector<std::pair<const struct ggml_tensor*, struct ggml_tensor*>>* copies) {
    return clone_tensor_to_cpu(ctx, src.norm,          &dst->norm,          copies) &&
           clone_tensor_to_cpu(ctx, src.mixer_kernel,  &dst->mixer_kernel,  copies) &&
           clone_tensor_to_cpu(ctx, src.mixer_bias,    &dst->mixer_bias,    copies) &&
           clone_tensor_to_cpu(ctx, src.gamma,         &dst->gamma,         copies) &&
           clone_tensor_to_cpu(ctx, src.ffn_norm,      &dst->ffn_norm,      copies) &&
           clone_tensor_to_cpu(ctx, src.ffn_linear1,   &dst->ffn_linear1,   copies) &&
           clone_tensor_to_cpu(ctx, src.ffn_linear1_b, &dst->ffn_linear1_b, copies) &&
           clone_tensor_to_cpu(ctx, src.ffn_linear2,   &dst->ffn_linear2,   copies) &&
           clone_tensor_to_cpu(ctx, src.ffn_linear2_b, &dst->ffn_linear2_b, copies) &&
           clone_tensor_to_cpu(ctx, src.ffn_gamma,     &dst->ffn_gamma,     copies);
}

bool ensure_cpu_decoder_shadow(VibeVoiceModel* model) {
    if (!model) return false;
    auto& shadow = model->cpu_decoder_shadow;
    if (shadow.ctx && shadow.buffer) return true;

    shadow.free();

    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 4096;
    p.no_alloc = true;
    shadow.ctx = ggml_init(p);
    if (!shadow.ctx) {
        VV_LOG_ERROR("hybrid-final-decoder: failed to create CPU shadow ctx");
        return false;
    }

    std::vector<std::pair<const struct ggml_tensor*, struct ggml_tensor*>> copies;
    copies.reserve(512);

    if (!clone_strided_to_cpu(shadow.ctx, model->w.at_dec.stem, &shadow.at_dec.stem, &copies)) {
        shadow.free();
        VV_LOG_ERROR("hybrid-final-decoder: failed to clone decoder stem");
        return false;
    }
    shadow.at_dec.ups.resize(model->w.at_dec.ups.size());
    for (size_t i = 0; i < model->w.at_dec.ups.size(); ++i) {
        if (!clone_strided_to_cpu(shadow.ctx, model->w.at_dec.ups[i], &shadow.at_dec.ups[i], &copies)) {
            shadow.free();
            VV_LOG_ERROR("hybrid-final-decoder: failed to clone decoder up layer %zu", i);
            return false;
        }
    }
    shadow.at_dec.stages.resize(model->w.at_dec.stages.size());
    for (size_t i = 0; i < model->w.at_dec.stages.size(); ++i) {
        shadow.at_dec.stages[i].resize(model->w.at_dec.stages[i].size());
        for (size_t j = 0; j < model->w.at_dec.stages[i].size(); ++j) {
            if (!clone_block1d_to_cpu(shadow.ctx, model->w.at_dec.stages[i][j], &shadow.at_dec.stages[i][j], &copies)) {
                shadow.free();
                VV_LOG_ERROR("hybrid-final-decoder: failed to clone decoder stage %zu block %zu", i, j);
                return false;
            }
        }
    }
    if (!clone_tensor_to_cpu(shadow.ctx, model->w.at_dec.final_norm, &shadow.at_dec.final_norm, &copies) ||
        !clone_strided_to_cpu(shadow.ctx, model->w.at_dec.head, &shadow.at_dec.head, &copies)) {
        shadow.free();
        VV_LOG_ERROR("hybrid-final-decoder: failed to clone decoder tail");
        return false;
    }

    shadow.buffer = ggml_backend_alloc_ctx_tensors_from_buft(shadow.ctx, ggml_backend_cpu_buffer_type());
    if (!shadow.buffer) {
        shadow.free();
        VV_LOG_ERROR("hybrid-final-decoder: failed to allocate CPU shadow decoder buffer");
        return false;
    }
    for (const auto& cp : copies) {
        ggml_backend_tensor_copy(cp.first, cp.second);
    }
    VV_LOG_INFO("hybrid-final-decoder: prepared CPU decoder shadow (%zu tensors)", copies.size());
    return true;
}

int get_meta_i32_any(const ModelLoader& m,
                     std::initializer_list<const char*> keys,
                     int def = 0) {
    for (const char* key : keys) {
        if (m.get_i64(key, INT64_MIN) != INT64_MIN) return m.get_i32(key, def);
    }
    return def;
}

float get_meta_f32_any(const ModelLoader& m,
                       std::initializer_list<const char*> keys,
                       float def = 0.0f) {
    for (const char* key : keys) {
        if (m.get_f32(key, std::numeric_limits<float>::quiet_NaN()) ==
            m.get_f32(key, std::numeric_limits<float>::quiet_NaN())) {
            return m.get_f32(key, def);
        }
    }
    return def;
}

std::vector<int32_t> get_meta_i32_array_any(const ModelLoader& m,
                                            std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        auto values = m.get_i32_array(key);
        if (!values.empty()) return values;
    }
    return {};
}

std::string get_meta_str_any(const ModelLoader& m,
                             std::initializer_list<const char*> keys,
                             const std::string& def = {}) {
    for (const char* key : keys) {
        auto value = m.get_str(key, "");
        if (!value.empty()) return value;
    }
    return def;
}

bool has_meta_i32(const ModelLoader& m, const char* key) {
    return m.get_i64(key, INT64_MIN) != INT64_MIN;
}

bool require_named_tensors(const ModelLoader& m,
                           std::initializer_list<const char*> names,
                           const char* component) {
    std::vector<std::string> missing;
    for (const char* name : names) {
        if (!m.has(name)) missing.emplace_back(name);
    }
    if (missing.empty()) return true;
    std::string joined;
    for (size_t i = 0; i < missing.size(); ++i) {
        if (i) joined += ", ";
        joined += missing[i];
    }
    VV_LOG_ERROR("vibevoice_load: %s missing required tensors: %s",
                 component, joined.c_str());
    return false;
}

bool require_metadata_keys(const ModelLoader& m,
                           std::initializer_list<const char*> keys,
                           const char* component) {
    std::vector<std::string> missing;
    for (const char* key : keys) {
        if (!m.has_key(key)) missing.emplace_back(key);
    }
    if (missing.empty()) return true;
    std::string joined;
    for (size_t i = 0; i < missing.size(); ++i) {
        if (i) joined += ", ";
        joined += missing[i];
    }
    VV_LOG_ERROR("vibevoice_load: unsupported KugelAudio schema/config: %s missing required metadata: %s",
                 component, joined.c_str());
    return false;
}

bool load_qwen2_layer(const ModelLoader& m, const std::string& prefix,
                      Qwen2LayerWeights* out) {
    auto get = [&](const char* n) { return m.tensor(prefix + n); };
    out->attn_norm   = get("attn_norm.weight");
    out->attn_q      = get("attn_q.weight");
    out->attn_q_bias = get("attn_q.bias");
    out->attn_k      = get("attn_k.weight");
    out->attn_k_bias = get("attn_k.bias");
    out->attn_v      = get("attn_v.weight");
    out->attn_v_bias = get("attn_v.bias");
    out->attn_o      = get("attn_o.weight");
    out->ffn_norm    = get("ffn_norm.weight");
    out->ffn_gate    = get("ffn_gate.weight");
    out->ffn_up      = get("ffn_up.weight");
    out->ffn_down    = get("ffn_down.weight");
    return out->attn_norm && out->attn_q && out->attn_k && out->attn_v &&
           out->attn_o && out->ffn_norm && out->ffn_gate && out->ffn_up &&
           out->ffn_down;
}

bool load_qwen2_stack(const ModelLoader& m, const std::string& prefix,
                      int n_layers, std::vector<Qwen2LayerWeights>* out) {
    out->assign(n_layers, {});
    for (int i = 0; i < n_layers; ++i) {
        char buf[64]; std::snprintf(buf, sizeof(buf), "%sblk.%d.", prefix.c_str(), i);
        if (!load_qwen2_layer(m, buf, &(*out)[i])) {
            VV_LOG_ERROR("load_qwen2_stack: layer %d incomplete (prefix %s)", i, prefix.c_str());
            return false;
        }
    }
    return true;
}

const char* kugelaudio_getenv(const char* legacy_name) {
    std::string preferred = legacy_name ? legacy_name : "";
    if (preferred.rfind("VIBEVOICE_KUGELAUDIO_", 0) == 0) {
        preferred.replace(0, std::strlen("VIBEVOICE_KUGELAUDIO_"), "KUGELAUDIO_");
    } else if (preferred.rfind("VIBEVOICE_", 0) == 0) {
        preferred.replace(0, std::strlen("VIBEVOICE_"), "KUGELAUDIO_");
    }
    if (!preferred.empty()) {
        if (const char* v = std::getenv(preferred.c_str()); v && *v) return v;
    }
    return std::getenv(legacy_name);
}

const char* kugelaudio_dump_dir_env() {
    return kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_DUMP_DIR");
}

bool kugelaudio_dump_enabled() {
    const char* dir = kugelaudio_dump_dir_env();
    return dir && *dir;
}

bool kugelaudio_dump_stage_allowed(const std::string& stage) {
    const char* filter = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_DUMP_FILTER");
    if (!filter || !*filter) return true;
    std::string f(filter);
    size_t start = 0;
    while (start <= f.size()) {
        size_t end = f.find(',', start);
        if (end == std::string::npos) end = f.size();
        std::string token = f.substr(start, end - start);
        const size_t left = token.find_first_not_of(" \t\n\r");
        if (left == std::string::npos) {
            token.clear();
        } else {
            const size_t right = token.find_last_not_of(" \t\n\r");
            token = token.substr(left, right - left + 1);
        }
        if (!token.empty() && stage.find(token) != std::string::npos) return true;
        if (end == f.size()) break;
        start = end + 1;
    }
    return false;
}

std::string kugelaudio_dump_escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

void kugelaudio_dump_write_meta(const std::filesystem::path& meta_path,
                                const std::string& stage,
                                const std::vector<size_t>& shape,
                                const char* dtype,
                                const char* semantic,
                                const char* source = "ggml") {
    std::ofstream meta(meta_path, std::ios::binary);
    if (!meta) return;
    meta << "{\n";
    meta << "  \"stage\": \"" << kugelaudio_dump_escape_json(stage) << "\",\n";
    meta << "  \"shape\": [";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) meta << ", ";
        meta << shape[i];
    }
    meta << "],\n";
    meta << "  \"dtype\": \"" << dtype << "\",\n";
    meta << "  \"layout\": \"row_major\",\n";
    meta << "  \"semantic\": \"" << kugelaudio_dump_escape_json(semantic ? semantic : "") << "\",\n";
    meta << "  \"source\": \"" << source << "\"\n";
    meta << "}\n";
}

void kugelaudio_dump_blob(const std::string& stage,
                          const void* data,
                          size_t bytes,
                          const std::vector<size_t>& shape,
                          const char* dtype,
                          const char* semantic) {
    if (!kugelaudio_dump_enabled() || !kugelaudio_dump_stage_allowed(stage)) return;
    const std::filesystem::path dir(kugelaudio_dump_dir_env());
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto bin_path  = dir / (stage + ".bin");
    const auto meta_path = dir / (stage + ".json");
    std::ofstream bin(bin_path, std::ios::binary);
    if (!bin) return;
    bin.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    bin.close();
    kugelaudio_dump_write_meta(meta_path, stage, shape, dtype, semantic);
}

void kugelaudio_dump_f32_matrix(const std::string& stage,
                                const std::vector<float>& data,
                                size_t rows,
                                size_t cols,
                                const char* semantic) {
    kugelaudio_dump_blob(stage, data.data(), data.size() * sizeof(float), {rows, cols}, "float32", semantic);
}

void kugelaudio_dump_f32_vector(const std::string& stage,
                                const std::vector<float>& data,
                                const char* semantic) {
    kugelaudio_dump_blob(stage, data.data(), data.size() * sizeof(float), {data.size()}, "float32", semantic);
}

void kugelaudio_dump_i32_vector(const std::string& stage,
                                const std::vector<int32_t>& data,
                                const char* semantic) {
    kugelaudio_dump_blob(stage, data.data(), data.size() * sizeof(int32_t), {data.size()}, "int32", semantic);
}

void kugelaudio_dump_int_vector(const std::string& stage,
                                const std::vector<int>& data,
                                const char* semantic) {
    std::vector<int32_t> tmp(data.begin(), data.end());
    kugelaudio_dump_i32_vector(stage, tmp, semantic);
}

void kugelaudio_dump_i32_scalar(const std::string& stage,
                                int32_t value,
                                const char* semantic) {
    kugelaudio_dump_blob(stage, &value, sizeof(value), {1}, "int32", semantic);
}

int kugelaudio_dump_frame_limit() {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_DUMP_FRAMES");
    if (!v || !*v) return 0;
    char* end = nullptr;
    long n = std::strtol(v, &end, 10);
    if (end == v || n < 0) return 0;
    return static_cast<int>(std::min<long>(n, 100000));
}

bool kugelaudio_should_dump_frame(int frame_index) {
    return kugelaudio_dump_enabled() && frame_index >= 0 && frame_index < kugelaudio_dump_frame_limit();
}

const char* kugelaudio_noise_dir_env() {
    return kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_NOISE_DIR");
}

bool kugelaudio_noise_load_enabled() {
    const char* dir = kugelaudio_noise_dir_env();
    return dir && *dir;
}

bool kugelaudio_teacher_force_step_embeds() {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_TEACHER_STEP_EMBEDS");
    return v && *v && std::string(v) != "0" && std::string(v) != "false";
}

bool kugelaudio_step_embed_blend_alpha(float* alpha) {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_BLEND_STEP_EMBEDS");
    if (!v || !*v) return false;
    char* end = nullptr;
    const float parsed = std::strtof(v, &end);
    if (end == v || !std::isfinite(parsed)) return false;
    if (alpha) *alpha = std::clamp(parsed, 0.0f, 1.0f);
    return true;
}

bool kugelaudio_temporal_step_embed_alpha(float* alpha) {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_TEMPORAL_STEP_EMBEDS");
    if (!v || !*v) return false;
    char* end = nullptr;
    const float parsed = std::strtof(v, &end);
    if (end == v || !std::isfinite(parsed)) return false;
    if (alpha) *alpha = std::clamp(parsed, 0.0f, 1.0f);
    return true;
}

bool kugelaudio_env_truthy(const char* v) {
    return v && *v && std::string(v) != "0" && std::string(v) != "false";
}

bool kugelaudio_model_prefers_step_embed_f16(const VibeVoiceModel* model) {
    if (!model || !model->loader.has_key("kugelaudio.architecture") || model->w.lm_layers.empty()) return false;
    const ggml_tensor* representative_weight = model->w.lm_layers.front().attn_q;
    return representative_weight && representative_weight->type == GGML_TYPE_F16;
}

bool kugelaudio_cast_step_embed_f16(const VibeVoiceModel* model, const VibeVoiceTTSParams& p) {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_CAST_STEP_EMBED_F16");
    if (v && *v) return kugelaudio_env_truthy(v);
    if (p.cast_step_embed_f16 >= 0) return p.cast_step_embed_f16 != 0;
    return kugelaudio_model_prefers_step_embed_f16(model);
}

bool kugelaudio_cast_diffusion_cond_f16() {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_CAST_DIFFUSION_COND_F16");
    return v && *v && std::string(v) != "0" && std::string(v) != "false";
}

float kugelaudio_single_sequence_min_ratio() {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_SINGLE_SEQUENCE_MIN_RATIO");
    if (!v || !*v) return 0.0f;
    char* end = nullptr;
    const float parsed = std::strtof(v, &end);
    if (end == v || !std::isfinite(parsed)) return 0.0f;
    return std::clamp(parsed, 0.0f, 1.0f);
}

bool kugelaudio_teacher_force_diffusion_cond() {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_TEACHER_DIFFUSION_COND");
    return v && *v && std::string(v) != "0" && std::string(v) != "false";
}

bool kugelaudio_diffusion_cond_blend_alpha(float* alpha) {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_BLEND_DIFFUSION_COND");
    if (!v || !*v) return false;
    char* end = nullptr;
    const float parsed = std::strtof(v, &end);
    if (end == v || !std::isfinite(parsed)) return false;
    if (alpha) *alpha = std::clamp(parsed, 0.0f, 1.0f);
    return true;
}

bool kugelaudio_temporal_diffusion_cond_alpha(float* alpha) {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_TEMPORAL_DIFFUSION_COND");
    if (!v || !*v) return false;
    char* end = nullptr;
    const float parsed = std::strtof(v, &end);
    if (end == v || !std::isfinite(parsed)) return false;
    if (alpha) *alpha = std::clamp(parsed, 0.0f, 1.0f);
    return true;
}

int kugelaudio_rolling_kv_frames() {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_ROLLING_KV_FRAMES");
    if (!v || !*v) return -1;
    char* end = nullptr;
    long n = std::strtol(v, &end, 10);
    if (end == v || n < 0) return -1;
    return static_cast<int>(std::min<long>(n, 100000));
}

std::string kugelaudio_frame_stage(int frame_index, const char* suffix) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "frame_%04d_%s", frame_index, suffix);
    return std::string(buf);
}

bool kugelaudio_load_f32_vector_from_dump(const std::filesystem::path& dir,
                                          const std::string& stage,
                                          size_t expected_count,
                                          std::vector<float>* out) {
    if (!out || expected_count == 0) return false;
    const auto bin_path = dir / (stage + ".bin");
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize bytes = f.tellg();
    if (bytes != static_cast<std::streamsize>(expected_count * sizeof(float))) {
        VV_LOG_ERROR("tts_15b: noise dump %s has %lld bytes, expected %zu",
                     bin_path.string().c_str(), static_cast<long long>(bytes), expected_count * sizeof(float));
        return false;
    }
    f.seekg(0, std::ios::beg);
    out->resize(expected_count);
    if (!f.read(reinterpret_cast<char*>(out->data()), bytes)) {
        VV_LOG_ERROR("tts_15b: failed to read noise dump %s", bin_path.string().c_str());
        return false;
    }
    return true;
}

std::vector<float> kugelaudio_align_semantic_features(const std::vector<float>& semantic,
                                                      int semantic_dim,
                                                      int semantic_T,
                                                      int acoustic_T) {
    if (semantic_dim <= 0 || acoustic_T <= 0) return {};
    std::vector<float> out(static_cast<size_t>(semantic_dim) * static_cast<size_t>(acoustic_T), 0.0f);
    const int copy_T = std::min(semantic_T, acoustic_T);
    if (copy_T > 0) {
        std::memcpy(out.data(), semantic.data(), sizeof(float) * static_cast<size_t>(semantic_dim) * static_cast<size_t>(copy_T));
    }
    return out;
}

std::string kugelaudio_acoustic_sampling_mode() {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_ACOUSTIC_MODE");
    if (!v || !*v) return "sample";
    return std::string(v);
}

std::vector<float> kugelaudio_sample_acoustic_features(const std::vector<float>& mean,
                                                        float fix_std,
                                                        const std::string& dist_type,
                                                        std::mt19937& rng) {
    std::vector<float> out = mean;
    if (out.empty()) return out;

    const std::string mode = kugelaudio_acoustic_sampling_mode();
    if (mode == "mean") return out;
    if (mode != "sample") {
        VV_LOG_WARN("tts_15b: unknown VIBEVOICE_KUGELAUDIO_ACOUSTIC_MODE=%s, falling back to sample",
                    mode.c_str());
    }

    if (dist_type == "none" || fix_std == 0.0f) return out;

    std::normal_distribution<float> norm(0.0f, 1.0f);
    if (dist_type == "fix") {
        for (float& v : out) {
            v += fix_std * norm(rng);
        }
        return out;
    }

    if (dist_type == "gaussian") {
        const float scalar_std = fix_std / 0.8f;
        const float sampled_std = scalar_std * norm(rng);
        for (float& v : out) {
            v += sampled_std * norm(rng);
        }
        return out;
    }

    return out;
}

std::vector<float> kugelaudio_apply_acoustic_scale_bias(const std::vector<float>& acoustic,
                                                        const VibeVoiceConfig& cfg) {
    std::vector<float> out = acoustic;
    if (std::isnan(cfg.speech_scaling)) return out;
    for (float& v : out) {
        v = (v + cfg.speech_bias) * cfg.speech_scaling;
    }
    return out;
}

}  // namespace

bool vibevoice_load(const std::string& path, VibeVoiceModel* out) {
    if (!out->loader.load(path)) return false;
    auto& m = out->loader;
    // ggml's element-wise ops don't accept mixed f32/f16; promote all small
    // tensors up front so RMSNorm scales, biases, gammas, scaling buffers and
    // tiny embeddings work in fp32 even when the gguf is fp16. Matmul weights
    // (large 2-D tensors) stay f16.
    m.promote_small_f16_to_f32();

    const bool is_kugelaudio = !m.get_str("kugelaudio.architecture", {}).empty() ||
                               m.get_i32("kugelaudio.schema_version", 0) != 0;
    if (is_kugelaudio) {
        if (!require_metadata_keys(m,
                {
                    "kugelaudio.schema_version",
                    "kugelaudio.architecture",
                    "kugelaudio.checkpoint",
                    "kugelaudio.decoder.hidden_size",
                    "kugelaudio.decoder.num_hidden_layers",
                    "kugelaudio.decoder.tts_hidden_layers",
                    "kugelaudio.decoder.num_attention_heads",
                    "kugelaudio.decoder.num_key_value_heads",
                    "kugelaudio.decoder.head_dim",
                    "kugelaudio.decoder.vocab_size",
                    "kugelaudio.decoder.rope_theta",
                    "kugelaudio.decoder.rms_norm_eps",
                    "kugelaudio.acoustic.vae_dim",
                    "kugelaudio.acoustic.encoder_ratios",
                    "kugelaudio.acoustic.encoder_depths",
                    "kugelaudio.acoustic.decoder_depths",
                    "kugelaudio.semantic.vae_dim",
                    "kugelaudio.semantic.encoder_ratios",
                    "kugelaudio.semantic.encoder_depths",
                    "kugelaudio.diffusion.head_layers",
                    "kugelaudio.diffusion.ffn_ratio",
                    "kugelaudio.diffusion.latent_size",
                    "kugelaudio.sample_rate",
                },
                "KugelAudio loader contract")) {
            return false;
        }
        const int schema = m.get_i32("kugelaudio.schema_version", 0);
        if (schema != 1) {
            VV_LOG_ERROR("vibevoice_load: unsupported KugelAudio schema/config: kugelaudio.schema_version=%d (want 1)", schema);
            return false;
        }
        const std::string arch = m.get_str("kugelaudio.architecture", {});
        if (arch != "kugelaudio") {
            VV_LOG_ERROR("vibevoice_load: unsupported KugelAudio schema/config: kugelaudio.architecture=%s (want kugelaudio)",
                         arch.empty() ? "<missing>" : arch.c_str());
            return false;
        }
        const std::string checkpoint = m.get_str("kugelaudio.checkpoint", {});
        if (checkpoint != "kugelaudio-0-open") {
            VV_LOG_ERROR("vibevoice_load: unsupported KugelAudio schema/config: kugelaudio.checkpoint=%s (want kugelaudio-0-open)",
                         checkpoint.empty() ? "<missing>" : checkpoint.c_str());
            return false;
        }
    }

    out->variant = m.get_str("vibevoice.variant", "");
    if (is_kugelaudio) {
        if (out->variant.empty()) {
            const std::string checkpoint = m.get_str("kugelaudio.checkpoint", "");
            if (checkpoint == "kugelaudio-0-open") out->variant = "1.5b";
        }
        if (out->variant == "kugelaudio-0-open") out->variant = "1.5b";
    } else if (out->variant.empty()) {
        out->variant = "realtime-0.5b";
    }
    const bool is_asr      = (out->variant == "asr-7b");
    const bool is_realtime = (out->variant == "realtime-0.5b");
    const bool is_15b      = (out->variant == "1.5b");

    auto& c = out->cfg;
    c.hidden       = get_meta_i32_any(m, {"kugelaudio.decoder.hidden_size", "vibevoice.hidden"});
    c.n_layers_lm  = get_meta_i32_any(m, {"kugelaudio.decoder.num_hidden_layers", "vibevoice.n_layers_lm"});
    c.n_layers_tlm = get_meta_i32_any(m, {"kugelaudio.decoder.tts_hidden_layers", "vibevoice.n_layers_tlm"});
    if (is_kugelaudio && !has_meta_i32(m, "vibevoice.n_layers_lm") && c.n_layers_lm > 0) {
        c.n_layers_lm -= c.n_layers_tlm;
    }
    c.n_heads      = get_meta_i32_any(m, {"kugelaudio.decoder.num_attention_heads", "vibevoice.n_heads"});
    c.n_kv_heads   = get_meta_i32_any(m, {"kugelaudio.decoder.num_key_value_heads", "vibevoice.n_kv_heads"});
    c.head_dim     = get_meta_i32_any(m, {"kugelaudio.decoder.head_dim", "vibevoice.head_dim"});
    c.vocab_size   = get_meta_i32_any(m, {"kugelaudio.decoder.vocab_size", "vibevoice.vocab_size"});
    c.rope_theta   = get_meta_f32_any(m, {"kugelaudio.decoder.rope_theta", "vibevoice.rope_theta"}, 1.0e6f);
    c.rms_norm_eps = get_meta_f32_any(m, {"kugelaudio.decoder.rms_norm_eps", "vibevoice.rms_norm_eps"}, 1.0e-6f);
    c.latent       = get_meta_i32_any(m, {"kugelaudio.diffusion.latent_size", "vibevoice.diffusion.latent"}, 64);
    c.head_layers  = get_meta_i32_any(m, {"kugelaudio.diffusion.head_layers", "vibevoice.diffusion.head_layers"}, 4);
    c.ffn_ratio    = get_meta_f32_any(m, {"kugelaudio.diffusion.ffn_ratio", "vibevoice.diffusion.ffn_ratio"}, 3.0f);
    c.vae_dim      = get_meta_i32_any(m, {"kugelaudio.acoustic.vae_dim", "vibevoice.acoustic.vae_dim"}, 64);
    c.acoustic_fix_std = get_meta_f32_any(m, {"kugelaudio.acoustic.fix_std", "vibevoice.acoustic.fix_std"}, is_kugelaudio ? 0.5f : 0.0f);
    c.acoustic_std_dist_type = get_meta_str_any(m, {"kugelaudio.acoustic.std_dist_type", "vibevoice.acoustic.std_dist_type"}, is_kugelaudio ? "gaussian" : "none");
    c.sample_rate  = get_meta_i32_any(m, {"kugelaudio.sample_rate", "vibevoice.sample_rate"}, 24000);

    // Acoustic decoder config (we only use the decoder side here)
    c.acoustic.channels   = 1;
    c.acoustic.vae_dim    = c.vae_dim;
    c.acoustic.eps        = get_meta_f32_any(m, {"kugelaudio.acoustic.eps", "vibevoice.acoustic.eps"}, 1e-5f);
    auto ratios = get_meta_i32_array_any(m, {"kugelaudio.acoustic.encoder_ratios", "vibevoice.acoustic.encoder_ratios"});
    auto depths = get_meta_i32_array_any(m, {"kugelaudio.acoustic.decoder_depths", "vibevoice.acoustic.decoder_depths"});
    if (ratios.empty() || depths.empty()) {
        VV_LOG_ERROR("vibevoice_load: acoustic ratios/depths missing");
        return false;
    }
    c.acoustic.ratios.assign(ratios.begin(), ratios.end());
    c.acoustic.depths.assign(depths.begin(), depths.end());
    c.acoustic.kernel_stem = 7;
    c.acoustic.kernel_head = 7;
    c.acoustic.ffn_mult    = 4;

    // ---- LM stacks ----
    auto& w = out->w;
    w.lm_tok_embd = m.tensor("lm.tok_embd.weight");
    if (!w.lm_tok_embd) {
        VV_LOG_ERROR("vibevoice_load: missing lm.tok_embd.weight");
        return false;
    }
    if (!load_qwen2_stack(m, "lm.",  c.n_layers_lm,  &w.lm_layers))  return false;
    if (is_realtime) {
        if (!load_qwen2_stack(m, "tlm.", c.n_layers_tlm, &w.tlm_layers)) return false;
        w.tlm_output_norm = m.tensor("tlm.output_norm.weight");
        w.tts_input_types = m.tensor("tts.input_types.weight");
        if (!w.tlm_output_norm || !w.tts_input_types) {
            VV_LOG_ERROR("vibevoice_load: missing tlm.output_norm or tts.input_types");
            return false;
        }
    }

    // ---- acoustic connector (always present) ----
    if (!require_named_tensors(m,
            {"ac.fc1.weight", "ac.fc2.weight", "ac.norm.weight"},
            "acoustic connector (ac.*)")) {
        return false;
    }
    w.ac_fc1_w = m.tensor("ac.fc1.weight");
    w.ac_fc1_b = m.tensor("ac.fc1.bias");
    w.ac_norm  = m.tensor("ac.norm.weight");
    w.ac_fc2_w = m.tensor("ac.fc2.weight");
    w.ac_fc2_b = m.tensor("ac.fc2.bias");

    if (is_realtime) {
        // ---- TTS-specific: EOS classifier + diffusion head + acoustic decoder ----
        if (!require_named_tensors(m,
                {"at.dec.stem.weight", "at.dec.head.weight", "dh.cond_proj"},
                "TTS acoustic decoder / diffusion head")) {
            return false;
        }
        w.eos_fc1_w = m.tensor("eos.fc1.weight");
        w.eos_fc1_b = m.tensor("eos.fc1.bias");
        w.eos_fc2_w = m.tensor("eos.fc2.weight");
        w.eos_fc2_b = m.tensor("eos.fc2.bias");
        if (!w.eos_fc1_w || !w.eos_fc2_w) {
            VV_LOG_ERROR("vibevoice_load: eos classifier missing");
            return false;
        }
        DiffusionHeadConfig dhc;
        dhc.hidden      = c.hidden;
        dhc.latent      = c.latent;
        dhc.head_layers = c.head_layers;
        dhc.ffn_ratio   = c.ffn_ratio;
        dhc.eps         = c.rms_norm_eps;
        dhc.freq_size   = 256;
        if (!load_diffusion_head(m, "dh.", dhc, &w.dh)) return false;
        if (!load_decoder(m, "at.dec", c.acoustic, &w.at_dec)) return false;
    }

    if (is_15b) {
        // ---- 1.5B: single-stack LM + diffusion head + decoder + encoders ----
        // No EOS classifier (uses LM logits + speech_end token instead).
        if (!require_named_tensors(m,
                {"at.dec.stem.weight", "at.dec.head.weight", "dh.cond_proj"},
                "raw-reference TTS acoustic decoder / diffusion head")) {
            return false;
        }
        DiffusionHeadConfig dhc;
        dhc.hidden      = c.hidden;
        dhc.latent      = c.latent;
        dhc.head_layers = c.head_layers;
        dhc.ffn_ratio   = c.ffn_ratio;
        dhc.eps         = c.rms_norm_eps;
        dhc.freq_size   = 256;
        if (!load_diffusion_head(m, "dh.", dhc, &w.dh)) return false;
        if (!load_decoder(m, "at.dec", c.acoustic, &w.at_dec)) return false;
        // Single LM stack: stash its final RMSNorm in tlm_output_norm so the
        // run_qwen2_stack call site can reuse the same hook the realtime path
        // uses for the upper stack.
        w.tlm_output_norm = m.tensor("lm.output_norm.weight");
        if (!w.tlm_output_norm) {
            VV_LOG_ERROR("vibevoice_load: missing lm.output_norm.weight");
            return false;
        }
        out->lm_head = m.tensor("lm_head.weight");
        if (!out->lm_head) {
            VV_LOG_ERROR("vibevoice_load: missing lm_head.weight");
            return false;
        }
    }

    if (is_asr || is_15b) {
        // ---- ASR-specific: encoders + semantic connector + lm_head ----
        // Encoder depths are forward order (3,3,3,3,3,3,8) — different from
        // the (reversed) decoder depths the TTS path uses.
        AcousticConfig enc_cfg = c.acoustic;
        auto enc_depths = get_meta_i32_array_any(m, {"kugelaudio.acoustic.encoder_depths", "vibevoice.acoustic.encoder_depths"});
        if (!enc_depths.empty()) enc_cfg.depths.assign(enc_depths.begin(), enc_depths.end());

        if (!require_named_tensors(m,
                {"at.enc.stem.weight", "at.enc.head.weight"},
                "acoustic encoder (at.enc.*)")) {
            return false;
        }
        if (!load_encoder(m, "at.enc", enc_cfg, &out->at_enc)) {
            VV_LOG_ERROR("vibevoice_load: acoustic encoder load failed");
            return false;
        }
        // Update the model's acoustic config so callers (e.g. the test) get the
        // right depths when running encoder_forward.
        c.acoustic = enc_cfg;
        // Semantic config (separate ratios/depths possible, but in practice same)
        out->semantic_vae_dim = get_meta_i32_any(m, {"kugelaudio.semantic.vae_dim", "vibevoice.semantic.vae_dim"}, 128);
        out->semantic_cfg.channels = 1;
        out->semantic_cfg.vae_dim  = out->semantic_vae_dim;
        out->semantic_cfg.eps      = get_meta_f32_any(m, {"kugelaudio.acoustic.eps", "vibevoice.acoustic.eps"}, 1e-5f);
        auto sm_ratios = get_meta_i32_array_any(m, {"kugelaudio.semantic.encoder_ratios", "vibevoice.semantic.encoder_ratios"});
        auto sm_depths = get_meta_i32_array_any(m, {"kugelaudio.semantic.encoder_depths", "vibevoice.semantic.encoder_depths"});
        if (sm_ratios.empty()) sm_ratios = ratios;
        if (sm_depths.empty()) sm_depths.assign(c.acoustic.depths.begin(), c.acoustic.depths.end());
        out->semantic_cfg.ratios.assign(sm_ratios.begin(), sm_ratios.end());
        out->semantic_cfg.depths.assign(sm_depths.begin(), sm_depths.end());
        out->semantic_cfg.kernel_stem = 7;
        out->semantic_cfg.kernel_head = 7;
        out->semantic_cfg.ffn_mult    = 4;
        if (!require_named_tensors(m,
                {"st.enc.stem.weight", "st.enc.head.weight"},
                "semantic encoder (st.enc.*)")) {
            return false;
        }
        if (!load_encoder(m, "st.enc", out->semantic_cfg, &out->st_enc)) {
            VV_LOG_ERROR("vibevoice_load: semantic encoder load failed");
            return false;
        }
        if (!require_named_tensors(m,
                {"sc.fc1.weight", "sc.fc2.weight", "sc.norm.weight", "lm_head.weight"},
                "semantic connector (sc.*) or lm_head")) {
            return false;
        }
        out->sc_fc1_w = m.tensor("sc.fc1.weight");
        out->sc_fc1_b = m.tensor("sc.fc1.bias");
        out->sc_norm  = m.tensor("sc.norm.weight");
        out->sc_fc2_w = m.tensor("sc.fc2.weight");
        out->sc_fc2_b = m.tensor("sc.fc2.bias");
        out->lm_head  = m.tensor("lm_head.weight");
        // ASR also has full output_norm
        if (!w.tlm_output_norm) w.tlm_output_norm = m.tensor("lm.output_norm.weight");
    }

    // ---- speech scaling buffers (handle fp32 or fp16) ----
    // Tensors live on the active backend's buffer; read the first scalar
    // via ggml_backend_tensor_get (memcpy on CPU, DtoH on GPU).
    auto load_scalar = [](struct ggml_tensor* t) -> float {
        if (!t) return 0.0f;
        if (t->type == GGML_TYPE_F32) {
            float v = 0.0f;
            ggml_backend_tensor_get(t, &v, 0, sizeof(float));
            return v;
        }
        if (t->type == GGML_TYPE_F16) {
            ggml_fp16_t h = 0;
            ggml_backend_tensor_get(t, &h, 0, sizeof(ggml_fp16_t));
            return ggml_fp16_to_fp32(h);
        }
        return 0.0f;
    };
    c.speech_scaling = load_scalar(m.tensor("speech.scaling"));
    c.speech_bias    = load_scalar(m.tensor("speech.bias"));

    VV_LOG_INFO("vibevoice_load: hidden=%d  layers=%d+%d  vocab=%d  scaling=%.4f bias=%.4f",
                c.hidden, c.n_layers_lm, c.n_layers_tlm, c.vocab_size,
                static_cast<double>(c.speech_scaling),
                static_cast<double>(c.speech_bias));
    if (is_kugelaudio) {
        VV_LOG_INFO("vibevoice_load: detected kugelaudio checkpoint=%s schema=%d runtime_path=kugelaudio_raw_ref_tts enabled={raw_ref_single_speaker:on semantic_conditioning:on pre_baked_voice:off multi_speaker_dialog:off}",
                    m.get_str("kugelaudio.checkpoint", "<missing>").c_str(),
                    m.get_i32("kugelaudio.schema_version", 0));
    }
    return true;
}

bool vibevoice_voice_load(const std::string&    path,
                          const VibeVoiceModel& model,
                          VibeVoiceVoice*       out) {
    if (!out) return false;
    static thread_local ModelLoader v_loader;
    if (!v_loader.load(path)) {
        VV_LOG_ERROR("voice_load: failed to open %s", path.c_str());
        return false;
    }
    auto& m = v_loader;

    const int hidden    = m.get_i32("voice.hidden");
    const int head_dim  = m.get_i32("voice.head_dim");
    const int n_kv      = m.get_i32("voice.n_kv_heads");
    const int n_lm_l    = m.get_i32("voice.lm.n_layers");
    const int n_tlm_l   = m.get_i32("voice.tts_lm.n_layers");
    out->seq_lm  = m.get_i32("voice.lm.seq_len");
    out->seq_tlm = m.get_i32("voice.tts_lm.seq_len");

    if (hidden != model.cfg.hidden || head_dim != model.cfg.head_dim ||
        n_kv   != model.cfg.n_kv_heads ||
        n_lm_l  != model.cfg.n_layers_lm ||
        n_tlm_l != model.cfg.n_layers_tlm) {
        VV_LOG_ERROR("voice_load: shape mismatch with model "
                     "(hidden %d/%d head_dim %d/%d n_kv %d/%d "
                     "lm_layers %d/%d tlm_layers %d/%d)",
                     hidden, model.cfg.hidden, head_dim, model.cfg.head_dim,
                     n_kv, model.cfg.n_kv_heads,
                     n_lm_l, model.cfg.n_layers_lm,
                     n_tlm_l, model.cfg.n_layers_tlm);
        return false;
    }

    auto load_stack = [&](const char* prefix, int n_layers, int seq_len,
                          std::vector<LayerKV>* dst) {
        dst->assign(n_layers, {});
        const size_t n = static_cast<size_t>(head_dim) * n_kv * seq_len;
        for (int i = 0; i < n_layers; ++i) {
            char nk[64], nv[64];
            std::snprintf(nk, sizeof(nk), "%s.k.%d", prefix, i);
            std::snprintf(nv, sizeof(nv), "%s.v.%d", prefix, i);
            struct ggml_tensor* k = m.tensor(nk);
            struct ggml_tensor* v = m.tensor(nv);
            if (!k || !v) {
                VV_LOG_ERROR("voice_load: missing %s or %s", nk, nv);
                return false;
            }
            (*dst)[i].k.assign(n, 0.0f);
            (*dst)[i].v.assign(n, 0.0f);
            ggml_backend_tensor_get(k, (*dst)[i].k.data(), 0, sizeof(float) * n);
            ggml_backend_tensor_get(v, (*dst)[i].v.data(), 0, sizeof(float) * n);
            (*dst)[i].past_len = seq_len;
        }
        return true;
    };
    if (!load_stack("voice.lm",     n_lm_l,  out->seq_lm,  &out->kv_lm))  return false;
    if (!load_stack("voice.tts_lm", n_tlm_l, out->seq_tlm, &out->kv_tlm)) return false;

    struct ggml_tensor* tlm_h = m.tensor("voice.tts_lm.last_hidden");
    if (!tlm_h) {
        VV_LOG_ERROR("voice_load: missing voice.tts_lm.last_hidden");
        return false;
    }
    auto last_col = [&](struct ggml_tensor* t, int seq, std::vector<float>* dst) {
        dst->resize(hidden);
        ggml_backend_tensor_get(t, dst->data(),
                                sizeof(float) * static_cast<size_t>(hidden) * (seq - 1),
                                sizeof(float) * hidden);
    };
    last_col(tlm_h, out->seq_tlm, &out->tlm_last_hidden);

    // Optional negative branch (CFG).
    out->has_neg = m.get_bool("voice.has_neg", false);
    if (out->has_neg) {
        out->seq_neg_lm  = m.get_i32("voice.neg_lm.seq_len");
        out->seq_neg_tlm = m.get_i32("voice.neg_tts_lm.seq_len");
        if (!load_stack("voice.neg_lm",     n_lm_l,  out->seq_neg_lm,  &out->kv_neg_lm))  out->has_neg = false;
        if (!load_stack("voice.neg_tts_lm", n_tlm_l, out->seq_neg_tlm, &out->kv_neg_tlm)) out->has_neg = false;
        struct ggml_tensor* nh = m.tensor("voice.neg_tts_lm.last_hidden");
        if (out->has_neg && nh) {
            last_col(nh, out->seq_neg_tlm, &out->neg_tlm_last_hidden);
        }
    }

    VV_LOG_INFO("voice_load: %s  lm=%dx%d  tlm=%dx%d  neg=%s",
                path.c_str(), n_lm_l, out->seq_lm, n_tlm_l, out->seq_tlm,
                out->has_neg ? "on" : "off");
    return true;
}

// ============================================================================
//  Inference
// ============================================================================

namespace {


// Build & run one forward pass through a Qwen2 stack.
// `inputs_embeds`: [hidden, n_new_tokens, B=1]
// `pos_start`:     absolute position of the first new token
// `kvs`:           resident K/V cache; new K/V written via ggml_cpy at
//                  offset kvs.past_len, then kvs.past_len is advanced.
// `out_hidden`:    if non-null, all output hidden states ([hidden * n_new_tokens])
// `out_hidden_last`: if non-null, only the last token's hidden state ([hidden])
bool run_qwen2_stack(struct ggml_context* /*ctx_ext*/,
                     const VibeVoiceConfig& cfg,
                     const std::vector<Qwen2LayerWeights>& layers,
                     struct ggml_tensor*  output_norm,
                     int                  pos_start,
                     int                  n_new_tokens,
                     const float*         inputs_embeds,
                     ResidentKV*           kvs,
                     std::vector<float>*   out_hidden,            // optional
                     std::vector<float>*   out_hidden_last) {     // optional
    const int hidden = cfg.hidden;

    Qwen2Hparams hp;
    hp.hidden_size       = hidden;
    hp.n_heads           = cfg.n_heads;
    hp.n_kv_heads        = cfg.n_kv_heads;
    hp.head_dim          = cfg.head_dim;
    hp.intermediate_size = 0;
    hp.rope_theta        = cfg.rope_theta;
    hp.rms_norm_eps      = cfg.rms_norm_eps;
    hp.use_flash_attn    = vv::backend_supports_flash_attn();

    const int kv_len_old = kvs->past_len;
    const int kv_len_new = kv_len_old + n_new_tokens;

    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 16384
               + ggml_graph_overhead_custom(16384, false);
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return false;

    struct ggml_tensor* x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, n_new_tokens, 1);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_new_tokens);
    struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len_new, n_new_tokens);

    struct ggml_tensor* h = x;
    std::vector<struct ggml_tensor*> k_writes(layers.size()),
                                     v_writes(layers.size());
    for (size_t li = 0; li < layers.size(); ++li) {
        auto out = qwen2_layer_forward_resident(ctx, h, pos, mask,
                                                *kvs, static_cast<int>(li),
                                                layers[li], hp);
        h            = out.y;
        k_writes[li] = out.k_write;
        v_writes[li] = out.v_write;
    }
    if (output_norm) {
        h = rms_norm(ctx, h, output_norm, cfg.rms_norm_eps);
    }

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
    // Interleave per-layer cpys (cpy_v_li MUST be in the graph before
    // attention_(li+1) is added through k_write[li+1]'s expansion - same
    // ordering rule as the ASR prefill, see test_qwen2_resident chain28
    // one-graph case).
    for (size_t li = 0; li < layers.size(); ++li) {
        ggml_build_forward_expand(gf, k_writes[li]);
        ggml_build_forward_expand(gf, v_writes[li]);
    }
    ggml_build_forward_expand(gf, h);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return false; }

    vv::backend_tensor_set(x, inputs_embeds, 0, sizeof(float) * hidden * n_new_tokens);
    std::vector<int32_t> pos_v(n_new_tokens);
    for (int i = 0; i < n_new_tokens; ++i) pos_v[i] = pos_start + i;
    vv::backend_tensor_set(pos, pos_v.data(), 0, sizeof(int32_t) * n_new_tokens);
    std::vector<ggml_fp16_t> mask_v(static_cast<size_t>(kv_len_new) * n_new_tokens);
    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n_new_tokens; ++i) {
        const int qabs = pos_start + i;
        for (int j = 0; j < kv_len_new; ++j) {
            mask_v[i * kv_len_new + j] = (j > qabs) ? f16_ninf : f16_zero;
        }
    }
    vv::backend_tensor_set(mask, mask_v.data(), 0, sizeof(ggml_fp16_t) * mask_v.size());

    if (!vv::compute_graph(gf)) {
        ggml_backend_buffer_free(in_buf); ggml_free(ctx); return false;
    }

    if (out_hidden) {
        out_hidden->resize(static_cast<size_t>(hidden) * n_new_tokens);
        ggml_backend_tensor_get(h, out_hidden->data(), 0,
                                sizeof(float) * hidden * n_new_tokens);
    }
    if (out_hidden_last) {
        out_hidden_last->resize(hidden);
        ggml_backend_tensor_get(h, out_hidden_last->data(),
                                sizeof(float) * hidden * (n_new_tokens - 1),
                                sizeof(float) * hidden);
    }

    kvs->past_len = kv_len_new;

    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return true;
}

// SpeechConnector: y = fc2(rmsnorm_last_dim(fc1(x)))
// `x` is [latent, B], output is [hidden, B].
// KugelAudio parity note: this helper is intentionally reused for the
// generated-latent path because the canonical implementation feeds generated
// speech latents through the acoustic connector only (no semantic re-encode)
// before the next LM step; see
// ../kugelaudio-open/src/kugelaudio_open/models/kugelaudio_inference.py.
std::vector<float> run_speech_connector(const VibeVoiceConfig&  cfg,
                                        const VibeVoiceWeights& w,
                                        const float*            x,
                                        int                     batch) {
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead();
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);

    const int latent = cfg.latent;
    const int hidden = cfg.hidden;
    struct ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, batch);

    struct ggml_tensor* h = ggml_mul_mat(ctx, w.ac_fc1_w, in);
    if (w.ac_fc1_b) h = ggml_add(ctx, h, w.ac_fc1_b);
    h = ggml_rms_norm(ctx, h, 1e-6f);
    h = ggml_mul(ctx, h, w.ac_norm);
    h = ggml_mul_mat(ctx, w.ac_fc2_w, h);
    if (w.ac_fc2_b) h = ggml_add(ctx, h, w.ac_fc2_b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, h);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return {}; }
    vv::backend_tensor_set(in, x, 0, sizeof(float) * latent * batch);

    std::vector<float> out;
    if (vv::compute_graph(gf)) {
        out.assign(static_cast<size_t>(hidden) * batch, 0.0f);
        ggml_backend_tensor_get(h, out.data(), 0, sizeof(float) * out.size());
    }
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return out;
}

// EOS classifier: sigmoid(fc2(relu(fc1(x))))
float run_eos_classifier(const VibeVoiceConfig&  /*cfg*/,
                         const VibeVoiceWeights& w,
                         const float*            x_hidden) {
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);

    const int hidden = w.eos_fc1_w->ne[0];
    struct ggml_tensor* in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);

    struct ggml_tensor* h = ggml_mul_mat(ctx, w.eos_fc1_w, in);
    if (w.eos_fc1_b) h = ggml_add(ctx, h, w.eos_fc1_b);
    h = ggml_relu(ctx, h);
    h = ggml_mul_mat(ctx, w.eos_fc2_w, h);
    if (w.eos_fc2_b) h = ggml_add(ctx, h, w.eos_fc2_b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, h);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return 0.0f; }
    vv::backend_tensor_set(in, x_hidden, 0, sizeof(float) * hidden);

    float logit = 0.0f;
    if (vv::compute_graph(gf)) {
        ggml_backend_tensor_get(h, &logit, 0, sizeof(float));
    }
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return 1.0f / (1.0f + std::exp(-logit));
}

// Look up the tts_input_types embedding for a given mask (0=speech, 1=text)
// and add it (broadcast across n_tokens) into x [hidden * n_tokens].
void add_input_type_embedding(const VibeVoiceConfig& cfg,
                              const VibeVoiceWeights& w,
                              int n_tokens,
                              int type_id,                // 0 or 1
                              float* x) {
    const int hidden = cfg.hidden;
    // tts_input_types lives on the active backend's buffer; pull the row
    // once via ggml_backend_tensor_get into a CPU staging buffer.
    const size_t row_bytes = (w.tts_input_types->type == GGML_TYPE_F32)
                             ? sizeof(float) * hidden
                             : sizeof(ggml_fp16_t) * hidden;
    std::vector<uint8_t> row(row_bytes);
    ggml_backend_tensor_get(w.tts_input_types, row.data(),
                            row_bytes * static_cast<size_t>(type_id),
                            row_bytes);
    if (w.tts_input_types->type == GGML_TYPE_F32) {
        const float* emb = reinterpret_cast<const float*>(row.data());
        for (int t = 0; t < n_tokens; ++t)
            for (int i = 0; i < hidden; ++i)
                x[t * hidden + i] += emb[i];
    } else if (w.tts_input_types->type == GGML_TYPE_F16) {
        const ggml_fp16_t* emb = reinterpret_cast<const ggml_fp16_t*>(row.data());
        for (int t = 0; t < n_tokens; ++t)
            for (int i = 0; i < hidden; ++i)
                x[t * hidden + i] += ggml_fp16_to_fp32(emb[i]);
    }
}

// Decode a SEQUENCE of N speech latents into audio samples in a single
// decoder pass. The decoder is causal — its convolutions need to see the
// full latent trajectory to produce coherent audio. Decoding frame-by-frame
// independently zeros out the receptive field across frames and yields
// "lyric"-style noise instead of intelligible speech.
// KugelAudio parity note: this matches the canonical non-streaming final
// decode, which stores latent chunks and decodes once at the end so the
// decoder tail is preserved; see
// ../kugelaudio-open/src/kugelaudio_open/models/kugelaudio_inference.py.
//
// `scaled_latents` has shape [vae_dim * n_frames] in row-major (latent
// fastest), matching what `ggml_new_tensor_3d(ctx, F32, n_frames, vae_dim, 1)`
// expects when ne[0] = n_frames is the contiguous dim.
std::vector<float> decode_latent_sequence(const VibeVoiceConfig&  cfg,
                                          const VibeVoiceWeights& w,
                                          const float*            scaled_latents,
                                          int                     n_frames) {
    if (n_frames <= 0) return {};
    // Backend-aware compute: build the graph in a no_alloc ctx, allocate
    // leaf tensors on the active backend's buffer, upload input via
    // ggml_backend_tensor_set, and let vv::compute_graph allocate the
    // intermediates and dispatch.
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);

    struct ggml_tensor* z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_frames, cfg.vae_dim, 1);
    ggml_set_name(z, "decode_z");

    struct ggml_tensor* y = decoder_forward(ctx, z, w.at_dec, cfg.acoustic);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return {}; }
    vv::backend_tensor_set(z, scaled_latents, 0,
                            sizeof(float) * cfg.vae_dim * n_frames);

    if (!vv::compute_graph(gf)) {
        ggml_backend_buffer_free(in_buf);
        ggml_free(ctx);
        return {};
    }
    const int T_full = static_cast<int>(y->ne[0]);
    std::vector<float> samples(T_full);
    ggml_backend_tensor_get(y, samples.data(), 0, sizeof(float) * T_full);
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return samples;
}

int streamed_decoder_chunk_frames() {
    const char* env = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_STREAM_DECODER_FRAMES");
    if (!env || !*env) return 8;
    const int v = std::atoi(env);
    return v > 0 ? v : 8;
}

std::vector<float> decode_latent_sequence_streaming(const VibeVoiceConfig&  cfg,
                                                    const VibeVoiceWeights& w,
                                                    const float*            scaled_latents,
                                                    int                     n_frames) {
    if (n_frames <= 0) return {};
    const int chunk_frames = std::max(1, streamed_decoder_chunk_frames());
    StreamingCache cache;
    cache.is_first_chunk = true;

    std::vector<float> all;
    for (int off = 0; off < n_frames; off += chunk_frames) {
        const int seg_T = std::min(chunk_frames, n_frames - off);
        cache.is_final_chunk = (off + seg_T == n_frames);

        std::vector<float> seg(static_cast<size_t>(cfg.vae_dim) * seg_T);
        for (int d = 0; d < cfg.vae_dim; ++d) {
            for (int t = 0; t < seg_T; ++t) {
                seg[static_cast<size_t>(d) * seg_T + t] =
                    scaled_latents[static_cast<size_t>(d) * n_frames + (off + t)];
            }
        }

        struct ggml_init_params p {};
        p.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
        p.no_alloc = true;
        struct ggml_context* ctx = ggml_init(p);
        if (!ctx) return {};

        struct ggml_tensor* z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seg_T, cfg.vae_dim, 1);
        ggml_set_name(z, "decode_z_stream");
        struct ggml_tensor* y = decoder_forward_streaming(ctx, z, w.at_dec, cfg.acoustic, cache);
        struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(gf, y);
        for (auto& kv : cache) {
            if (kv.second.next_view) ggml_build_forward_expand(gf, kv.second.next_view);
        }

        ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
        if (!in_buf) { ggml_free(ctx); return {}; }
        vv::backend_tensor_set(z, seg.data(), 0, sizeof(float) * seg.size());
        for (auto& kv : cache) {
            StreamingCacheEntry& e = kv.second;
            if (!e.prefix || e.T == 0) continue;
            const size_t need = static_cast<size_t>(e.T) * e.C;
            if (cache.is_first_chunk || e.data.size() != need) {
                std::vector<float> zeros(need, 0.0f);
                vv::backend_tensor_set(e.prefix, zeros.data(), 0, sizeof(float) * need);
            } else {
                vv::backend_tensor_set(e.prefix, e.data.data(), 0, sizeof(float) * need);
            }
        }

        if (!vv::compute_graph(gf)) {
            ggml_backend_buffer_free(in_buf);
            ggml_free(ctx);
            return {};
        }
        const size_t n = static_cast<size_t>(y->ne[0]) * y->ne[1] * y->ne[2];
        const size_t old = all.size();
        all.resize(old + n);
        ggml_backend_tensor_get(y, all.data() + old, 0, sizeof(float) * n);
        for (auto& kv : cache) {
            StreamingCacheEntry& e = kv.second;
            if (!e.next_view || e.T == 0) continue;
            const size_t need = static_cast<size_t>(e.T) * e.C;
            e.data.assign(need, 0.0f);
            ggml_backend_tensor_get(e.next_view, e.data.data(), 0, sizeof(float) * need);
            e.next_view = nullptr;
            e.prefix    = nullptr;
        }
        cache.is_first_chunk = false;
        ggml_backend_buffer_free(in_buf);
        ggml_free(ctx);
    }
    return all;
}

std::vector<float> decode_latent_sequence_cpu(const VibeVoiceConfig& cfg,
                                              const CpuDecoderShadow& shadow,
                                              const float* scaled_latents,
                                              int n_frames) {
    if (n_frames <= 0 || !shadow.ctx || !shadow.buffer) return {};

    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);
    if (!ctx) return {};

    struct ggml_tensor* z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_frames, cfg.vae_dim, 1);
    ggml_set_name(z, "decode_z_cpu");

    struct ggml_tensor* y = decoder_forward(ctx, z, shadow.at_dec, cfg.acoustic);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, y);

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        ggml_free(ctx);
        return {};
    }
    ggml_backend_cpu_set_n_threads(cpu_backend, 0);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!allocr) {
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return {};
    }
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        ggml_gallocr_free(allocr);
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return {};
    }
    ggml_backend_tensor_set(z, scaled_latents, 0, sizeof(float) * cfg.vae_dim * n_frames);
    if (ggml_backend_graph_compute(cpu_backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(allocr);
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return {};
    }

    const int T_full = static_cast<int>(y->ne[0]);
    std::vector<float> samples(T_full);
    ggml_backend_tensor_get(y, samples.data(), 0, sizeof(float) * T_full);
    ggml_gallocr_free(allocr);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx);
    return samples;
}

}  // namespace

// Forward declaration — implementation later in this file.
namespace {
struct PreparedSpeechConditioning {
    std::vector<int> per_speaker_Tc;
    std::vector<size_t> ref_sample_counts;
    std::vector<size_t> acoustic_feature_counts;
    std::vector<float> speech_features;
    int total_Tc = 0;
};

struct PreparedSpeechConditioningResult {
    int rc = 0;
    bool stopped_after_preprocessing = false;
    bool stopped_after_connectors = false;
};

struct ChunkLatentContinuityState {
    std::vector<float> latents;
    int frames = 0;
};

bool should_short_circuit_after_conditioning_for_test();
bool should_short_circuit_after_connectors_for_test();

PreparedSpeechConditioningResult prepare_speech_conditioning(VibeVoiceModel* model,
                                                             const VibeVoiceTTSParams& p,
                                                             bool is_kugelaudio,
                                                             std::mt19937& rng,
                                                             PreparedSpeechConditioning* out) {
    PreparedSpeechConditioningResult result;
    if (!model || !out) {
        result.rc = -1;
        return result;
    }
    const auto& cfg = model->cfg;
    const auto& w = model->w;
    const int hidden = cfg.hidden;

    out->per_speaker_Tc.clear();
    out->ref_sample_counts.clear();
    out->acoustic_feature_counts.clear();
    out->speech_features.clear();
    out->total_Tc = 0;
    out->per_speaker_Tc.reserve(p.ref_audio_paths.size());
    out->ref_sample_counts.reserve(p.ref_audio_paths.size());
    out->acoustic_feature_counts.reserve(p.ref_audio_paths.size());

    for (size_t spk = 0; spk < p.ref_audio_paths.size(); ++spk) {
        const std::string& ref_wav_path = p.ref_audio_paths[spk];

        std::vector<float> ref_audio;
        if (load_wav_24k_mono(ref_wav_path.c_str(), &ref_audio) != 0 || ref_audio.empty()) {
            VV_LOG_ERROR("tts_15b: failed to load reference WAV %zu: %s", spk, ref_wav_path.c_str());
            result.rc = -3;
            return result;
        }

        normalize_dbfs(&ref_audio);
        if (is_kugelaudio) {
            kugelaudio_dump_f32_vector("00_ref_audio_post_norm", ref_audio,
                                       "normalized 24kHz mono reference audio");
        }

        std::vector<float> ac_lat, sm_lat;
        int Tc_a = 0, Tc_s = 0;
        if (!detail::run_encoder_buf(model->at_enc, cfg.acoustic, ref_audio, &ac_lat, &Tc_a)) {
            result.rc = -4;
            return result;
        }
        if (!detail::run_encoder_buf(model->st_enc, model->semantic_cfg, ref_audio, &sm_lat, &Tc_s)) {
            result.rc = -4;
            return result;
        }
        if (is_kugelaudio) {
            kugelaudio_dump_f32_matrix("01_acoustic_encoder_out", ac_lat,
                                       static_cast<size_t>(Tc_a), static_cast<size_t>(cfg.vae_dim),
                                       "C++ acoustic encoder output before any canonical sampling stage");
            kugelaudio_dump_f32_matrix("02_semantic_encoder_out", sm_lat,
                                       static_cast<size_t>(Tc_s), static_cast<size_t>(model->semantic_vae_dim),
                                       "C++ semantic encoder output prior to canonical length alignment");
        }

        std::vector<float> sm_lat_aligned;
        int Tc = Tc_a;
        if (is_kugelaudio) {
            sm_lat_aligned = kugelaudio_align_semantic_features(sm_lat,
                                                                model->semantic_vae_dim,
                                                                Tc_s,
                                                                Tc_a);
            kugelaudio_dump_f32_matrix("03_semantic_aligned", sm_lat_aligned,
                                       static_cast<size_t>(Tc_a), static_cast<size_t>(model->semantic_vae_dim),
                                       "semantic features after canonical pad/truncate to acoustic length");
        } else {
            if (Tc_a != Tc_s) {
                VV_LOG_ERROR("tts_15b: encoder frame mismatch on speaker %zu (%d vs %d)", spk, Tc_a, Tc_s);
                result.rc = -5;
                return result;
            }
            sm_lat_aligned = sm_lat;
        }

        std::vector<float> acoustic_features = ac_lat;
        if (is_kugelaudio) {
            acoustic_features = kugelaudio_sample_acoustic_features(ac_lat,
                                                                    cfg.acoustic_fix_std,
                                                                    cfg.acoustic_std_dist_type,
                                                                    rng);
            kugelaudio_dump_f32_matrix("01b_acoustic_features_after_sampling", acoustic_features,
                                       static_cast<size_t>(Tc), static_cast<size_t>(cfg.vae_dim),
                                       "acoustic conditioning features after canonical tokenizer sampling");
        }

        std::vector<float> ac_cond = acoustic_features;
        if (is_kugelaudio) {
            ac_cond = kugelaudio_apply_acoustic_scale_bias(acoustic_features, cfg);
            kugelaudio_dump_f32_matrix("04_acoustic_after_scale_bias", ac_cond,
                                       static_cast<size_t>(Tc), static_cast<size_t>(cfg.vae_dim),
                                       "acoustic conditioning features after canonical bias/scale step");
        }

        if (is_kugelaudio && should_short_circuit_after_conditioning_for_test()) {
            VV_LOG_INFO("tts_15b: test hook stopping after preprocessing+conditioning encoders (single_ref_frames=%d)", Tc);
            result.stopped_after_preprocessing = true;
            return result;
        }

        auto ac_emb = detail::run_connector(w.ac_fc1_w, w.ac_fc1_b, w.ac_norm,
                                             w.ac_fc2_w, w.ac_fc2_b,
                                             ac_cond, cfg.vae_dim, Tc, hidden);
        auto sm_emb = detail::run_connector(model->sc_fc1_w, model->sc_fc1_b, model->sc_norm,
                                             model->sc_fc2_w, model->sc_fc2_b,
                                             sm_lat_aligned, model->semantic_vae_dim, Tc, hidden);
        if (is_kugelaudio) {
            kugelaudio_dump_f32_matrix("05_acoustic_connector_out", ac_emb,
                                       static_cast<size_t>(Tc), static_cast<size_t>(hidden),
                                       "acoustic connector output in [T, hidden]");
            kugelaudio_dump_f32_matrix("06_semantic_connector_out", sm_emb,
                                       static_cast<size_t>(Tc), static_cast<size_t>(hidden),
                                       "semantic connector output in [T, hidden]");
        }
        if (ac_emb.size() != sm_emb.size() || ac_emb.size() != static_cast<size_t>(hidden) * Tc) {
            VV_LOG_ERROR("tts_15b: connector output size mismatch on speaker %zu", spk);
            result.rc = -6;
            return result;
        }

        std::vector<float> fused_features;
        if (!detail::fuse_conditioning_features(ac_emb, sm_emb, hidden, Tc, &fused_features)) {
            VV_LOG_ERROR("tts_15b: failed to fuse acoustic + semantic conditioning on speaker %zu", spk);
            result.rc = -6;
            return result;
        }
        if (is_kugelaudio) {
            kugelaudio_dump_f32_matrix("07_fused_speech_features", fused_features,
                                       static_cast<size_t>(Tc), static_cast<size_t>(hidden),
                                       "fused acoustic + semantic speech conditioning features");
        }
        if (is_kugelaudio && should_short_circuit_after_connectors_for_test()) {
            VV_LOG_INFO("tts_15b: test hook stopping after conditioning connectors (single_ref_frames=%d hidden=%d)", Tc, hidden);
            result.stopped_after_connectors = true;
            return result;
        }

        const size_t base = out->speech_features.size();
        out->speech_features.resize(base + fused_features.size());
        std::memcpy(out->speech_features.data() + base,
                    fused_features.data(),
                    sizeof(float) * fused_features.size());

        out->per_speaker_Tc.push_back(Tc);
        out->ref_sample_counts.push_back(ref_audio.size());
        out->acoustic_feature_counts.push_back(ac_lat.size());
        out->total_Tc += Tc;
    }

    return result;
}

int tts_15b_generate_single_pass(VibeVoiceModel*            model,
                                 const std::string&         text,
                                 const VibeVoiceTTSParams&  p,
                                 std::vector<float>*        samples,
                                 const PreparedSpeechConditioning* prepared_conditioning = nullptr,
                                 const ChunkLatentContinuityState* latent_prefix = nullptr,
                                 ChunkLatentContinuityState* latent_tail_out = nullptr);

int tts_15b_generate_segmented_state(VibeVoiceModel*                         model,
                                     const detail::KugelAudioChunkPlan&      plan,
                                     const VibeVoiceTTSParams&               p,
                                     std::vector<float>*                     samples);

std::string describe_chunk_boundary_types(const std::vector<detail::KugelAudioChunkBoundaryType>& boundary_types) {
    std::string out;
    for (size_t i = 0; i < boundary_types.size(); ++i) {
        if (!out.empty()) out += ",";
        out += detail::kugelaudio_chunk_boundary_type_name(boundary_types[i]);
    }
    return out;
}

std::string build_prompt_continuity_instruction() {
    return "Use only the voice from Voice input Speaker 0. Keep the same speaker identity, accent, loudness, pacing, and intonation across chunks.";
}

std::string apply_text_end_padding(const std::string& text, TextEndPaddingMode mode) {
    if (mode == TextEndPaddingMode::None) return text;
    std::string out = text;
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) out.pop_back();
    size_t begin = 0;
    while (begin < out.size() && std::isspace(static_cast<unsigned char>(out[begin]))) ++begin;
    if (begin > 0) out.erase(0, begin);
    if (out.empty()) return text;
    if (mode == TextEndPaddingMode::Ellipsis) {
        if (out.size() >= 3 && out.compare(out.size() - 3, 3, "...") == 0) return out;
        while (!out.empty() && out.back() == '.') out.pop_back();
        out += "...";
        return out;
    }
    return text;
}

double rms_range(const std::vector<float>& xs, size_t begin, size_t end) {
    begin = std::min(begin, xs.size());
    end = std::min(end, xs.size());
    if (end <= begin) return 0.0;
    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) sum += static_cast<double>(xs[i]) * xs[i];
    return std::sqrt(sum / static_cast<double>(end - begin));
}

size_t leading_quiet_samples(const std::vector<float>& xs, int sample_rate, double threshold) {
    if (xs.empty() || sample_rate <= 0) return 0;
    const size_t win = std::max<size_t>(1, static_cast<size_t>(sample_rate) / 50);   // 20 ms
    const size_t step = std::max<size_t>(1, static_cast<size_t>(sample_rate) / 100); // 10 ms
    for (size_t pos = 0; pos + win <= xs.size(); pos += step) {
        if (rms_range(xs, pos, pos + win) > threshold) return pos;
    }
    return xs.size();
}

size_t trailing_quiet_samples(const std::vector<float>& xs, int sample_rate, double threshold) {
    if (xs.empty() || sample_rate <= 0) return 0;
    const size_t win = std::max<size_t>(1, static_cast<size_t>(sample_rate) / 50);   // 20 ms
    const size_t step = std::max<size_t>(1, static_cast<size_t>(sample_rate) / 100); // 10 ms
    for (size_t end = xs.size(); end >= win;) {
        const size_t begin = end - win;
        if (rms_range(xs, begin, end) > threshold) return xs.size() - end;
        if (begin < step) break;
        end = begin - step;
    }
    return xs.size();
}

size_t ms_to_samples(int ms, int sample_rate) {
    if (ms <= 0 || sample_rate <= 0) return 0;
    return static_cast<size_t>((static_cast<int64_t>(ms) * sample_rate) / 1000);
}

void fade_in_prefix(std::vector<float>& xs, size_t fade_samples) {
    if (xs.empty() || fade_samples == 0) return;
    const size_t n = std::min(fade_samples, xs.size());
    for (size_t i = 0; i < n; ++i) {
        const float gain = n <= 1 ? 1.0f : static_cast<float>(i) / static_cast<float>(n - 1);
        xs[i] *= gain;
    }
}

void fade_out_suffix(std::vector<float>& xs, size_t fade_samples) {
    if (xs.empty() || fade_samples == 0) return;
    const size_t n = std::min(fade_samples, xs.size());
    const size_t begin = xs.size() - n;
    for (size_t i = 0; i < n; ++i) {
        const float gain = n <= 1 ? 0.0f : 1.0f - static_cast<float>(i) / static_cast<float>(n - 1);
        xs[begin + i] *= gain;
    }
}

void apply_chunk_boundary_cleanup(std::vector<std::vector<float>>& chunks,
                                  int sample_rate,
                                  int keep_leading_ms,
                                  int keep_trailing_ms,
                                  int fade_ms,
                                  bool verbose) {
    if (chunks.empty() || sample_rate <= 0) return;
    const double quiet_threshold = 0.003;
    const size_t keep_leading = ms_to_samples(keep_leading_ms, sample_rate);
    const size_t keep_trailing = ms_to_samples(keep_trailing_ms, sample_rate);
    const size_t fade_samples = ms_to_samples(fade_ms, sample_rate);
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto& chunk = chunks[i];
        const size_t before = chunk.size();
        size_t trim_head = 0;
        size_t trim_tail = 0;
        const size_t head_quiet = leading_quiet_samples(chunk, sample_rate, quiet_threshold);
        if (i > 0 && head_quiet > keep_leading) {
            trim_head = head_quiet - keep_leading;
            chunk.erase(chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(trim_head));
        }
        const size_t tail_quiet = trailing_quiet_samples(chunk, sample_rate, quiet_threshold);
        if (i + 1 < chunks.size() && tail_quiet > keep_trailing) {
            trim_tail = tail_quiet - keep_trailing;
            chunk.resize(chunk.size() - trim_tail);
        }
        if (i > 0) fade_in_prefix(chunk, fade_samples);
        if (i + 1 < chunks.size()) fade_out_suffix(chunk, fade_samples);
        if (verbose) {
            std::fprintf(stderr,
                         "[chunking] cleanup chunk %zu trim_head_ms=%.1f trim_tail_ms=%.1f fade_in_ms=%d fade_out_ms=%d samples_before=%zu samples_after=%zu\n",
                         i + 1,
                         1000.0 * static_cast<double>(trim_head) / static_cast<double>(sample_rate),
                         1000.0 * static_cast<double>(trim_tail) / static_cast<double>(sample_rate),
                         i > 0 ? fade_ms : 0,
                         i + 1 < chunks.size() ? fade_ms : 0,
                         before,
                         chunk.size());
        }
    }
}

std::vector<float> extract_continuity_tail(const std::vector<float>& previous_chunk,
                                           int sample_rate,
                                           int tail_ms,
                                           double* tail_rms_out) {
    if (tail_rms_out) *tail_rms_out = 0.0;
    if (previous_chunk.empty() || sample_rate <= 0 || tail_ms <= 0) return {};

    const size_t requested = std::max<size_t>(1, static_cast<size_t>(sample_rate) *
                                                    static_cast<size_t>(tail_ms) / 1000);
    const size_t win = std::max<size_t>(1, static_cast<size_t>(sample_rate) / 20); // 50 ms
    const double whole_rms = rms_range(previous_chunk, 0, previous_chunk.size());
    const double gate = std::max(0.001, whole_rms * 0.10);

    size_t voiced_end = previous_chunk.size();
    bool found = false;
    for (size_t end = previous_chunk.size(); end > 0;) {
        const size_t begin = end > win ? end - win : 0;
        if (rms_range(previous_chunk, begin, end) >= gate) {
            voiced_end = end;
            found = true;
            break;
        }
        if (begin == 0) break;
        end = begin;
    }
    if (!found) voiced_end = previous_chunk.size();

    const size_t min_tail = std::max<size_t>(1, static_cast<size_t>(sample_rate) / 5); // 200 ms
    const size_t begin = voiced_end > requested ? voiced_end - requested : 0;
    if (voiced_end <= begin || voiced_end - begin < min_tail) return {};

    std::vector<float> tail(previous_chunk.begin() + static_cast<std::ptrdiff_t>(begin),
                            previous_chunk.begin() + static_cast<std::ptrdiff_t>(voiced_end));
    const double tail_rms = rms_range(tail, 0, tail.size());
    if (tail_rms < 0.0005) return {};
    if (tail_rms_out) *tail_rms_out = tail_rms;
    return tail;
}

std::string continuity_temp_ref_path(const VibeVoiceModel* model, size_t chunk_index) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const uint32_t nonce = std::random_device{}();
    return (dir / (std::string("kugelaudio-continuity-ref-") +
                   std::to_string(reinterpret_cast<std::uintptr_t>(model)) + "-" +
                   std::to_string(chunk_index) + "-" + std::to_string(nonce) + ".wav")).string();
}

bool write_tail_reference_wav(const std::string& original_ref_path,
                              const std::vector<float>& previous_chunk,
                              int sample_rate,
                              int tail_ms,
                              const VibeVoiceModel* model,
                              size_t chunk_index,
                              std::string* path_out) {
    if (!path_out) return false;
    path_out->clear();
    std::vector<float> original_ref;
    if (load_wav_24k_mono(original_ref_path, &original_ref) != 0 || original_ref.empty()) {
        VV_LOG_WARN("tts_15b: tail-reference continuity could not load original reference audio");
        return false;
    }

    double tail_rms = 0.0;
    std::vector<float> tail = extract_continuity_tail(previous_chunk, sample_rate, tail_ms, &tail_rms);
    if (tail.empty()) {
        VV_LOG_WARN("tts_15b: tail-reference continuity skipped: no voiced previous-tail segment found");
        return false;
    }

    const double ref_rms = rms_range(original_ref, 0, original_ref.size());
    if (ref_rms > 1.0e-6 && tail_rms > 1.0e-6) {
        const double gain = std::clamp((ref_rms * 0.8) / tail_rms, 0.25, 3.0);
        for (float& v : tail) v = static_cast<float>(std::clamp(static_cast<double>(v) * gain, -1.0, 1.0));
    }

    std::vector<float> combined;
    combined.reserve(original_ref.size() + static_cast<size_t>(sample_rate / 10) + tail.size());
    combined.insert(combined.end(), original_ref.begin(), original_ref.end());
    combined.insert(combined.end(), static_cast<size_t>(sample_rate / 10), 0.0f);
    combined.insert(combined.end(), tail.begin(), tail.end());

    const std::string tmp_path = continuity_temp_ref_path(model, chunk_index);
    vv_audio audio{};
    audio.samples = combined.data();
    audio.n_samples = combined.size();
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    if (save_wav_pcm16(tmp_path, audio) != 0) {
        VV_LOG_WARN("tts_15b: tail-reference continuity could not write temporary reference wav: %s", tmp_path.c_str());
        return false;
    }
    *path_out = tmp_path;
    VV_LOG_INFO("tts_15b: tail-reference continuity chunk=%zu tail_ms=%d tail_samples=%zu ref_rms=%.5f tail_rms=%.5f",
                chunk_index + 1, tail_ms, tail.size(), ref_rms, tail_rms);
    return true;
}

struct CleanTailStats {
    size_t begin = 0;
    size_t end = 0;
    double rms = 0.0;
    double mean = 0.0;
    double zcr = 0.0;
    double diff_ratio = 0.0;
    double peak = 0.0;
    double gain = 1.0;
};

double mean_range(const std::vector<float>& xs, size_t begin, size_t end) {
    begin = std::min(begin, xs.size());
    end = std::min(end, xs.size());
    if (end <= begin) return 0.0;
    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) sum += xs[i];
    return sum / static_cast<double>(end - begin);
}

double peak_range(const std::vector<float>& xs, size_t begin, size_t end) {
    begin = std::min(begin, xs.size());
    end = std::min(end, xs.size());
    double peak = 0.0;
    for (size_t i = begin; i < end; ++i) peak = std::max(peak, std::abs(static_cast<double>(xs[i])));
    return peak;
}

double zcr_range(const std::vector<float>& xs, size_t begin, size_t end) {
    begin = std::min(begin, xs.size());
    end = std::min(end, xs.size());
    if (end <= begin + 1) return 0.0;
    size_t crossings = 0;
    float prev = xs[begin];
    for (size_t i = begin + 1; i < end; ++i) {
        const float cur = xs[i];
        if ((prev < 0.0f && cur >= 0.0f) || (prev >= 0.0f && cur < 0.0f)) ++crossings;
        prev = cur;
    }
    return static_cast<double>(crossings) / static_cast<double>(end - begin - 1);
}

double diff_ratio_range(const std::vector<float>& xs, size_t begin, size_t end) {
    begin = std::min(begin, xs.size());
    end = std::min(end, xs.size());
    if (end <= begin + 1) return 0.0;
    double diff_sq = 0.0;
    for (size_t i = begin + 1; i < end; ++i) {
        const double d = static_cast<double>(xs[i]) - xs[i - 1];
        diff_sq += d * d;
    }
    const double diff_rms = std::sqrt(diff_sq / static_cast<double>(end - begin - 1));
    const double rms = rms_range(xs, begin, end);
    return rms > 1.0e-8 ? diff_rms / rms : 0.0;
}

double voiced_rms_reference(const std::vector<float>& ref, int sample_rate) {
    if (ref.empty() || sample_rate <= 0) return rms_range(ref, 0, ref.size());
    const size_t win = std::max<size_t>(1, static_cast<size_t>(sample_rate) / 20);
    const double whole = rms_range(ref, 0, ref.size());
    const double gate = std::max(0.001, whole * 0.35);
    double sum_sq = 0.0;
    size_t count = 0;
    for (size_t begin = 0; begin < ref.size(); begin += win) {
        const size_t end = std::min(ref.size(), begin + win);
        if (rms_range(ref, begin, end) < gate) continue;
        for (size_t i = begin; i < end; ++i) {
            sum_sq += static_cast<double>(ref[i]) * ref[i];
            ++count;
        }
    }
    return count > 0 ? std::sqrt(sum_sq / static_cast<double>(count)) : whole;
}

std::vector<float> extract_clean_continuity_tail(const std::vector<float>& previous_chunk,
                                                 int sample_rate,
                                                 int tail_ms,
                                                 CleanTailStats* stats_out) {
    if (stats_out) *stats_out = CleanTailStats{};
    if (previous_chunk.empty() || sample_rate <= 0 || tail_ms <= 0) return {};

    const size_t win = std::max<size_t>(1, static_cast<size_t>(sample_rate) / 20); // 50 ms
    const size_t target = std::max<size_t>(win, static_cast<size_t>(sample_rate) *
                                                static_cast<size_t>(tail_ms) / 1000);
    const size_t min_tail = std::max<size_t>(win * 4, static_cast<size_t>(sample_rate) / 5); // >= 200 ms
    const size_t n_win = (previous_chunk.size() + win - 1) / win;
    if (n_win == 0) return {};

    const double whole_rms = rms_range(previous_chunk, 0, previous_chunk.size());
    const double gate = std::max(0.003, whole_rms * 0.30);
    const size_t search_windows = std::max<size_t>(1, static_cast<size_t>(sample_rate) * 6000 / 1000 / win);
    const size_t first_win = n_win > search_windows ? n_win - search_windows : 0;

    struct WindowMetric { double rms, peak, zcr, diff; bool voiced; };
    std::vector<WindowMetric> metrics(n_win);
    for (size_t wi = 0; wi < n_win; ++wi) {
        const size_t b = wi * win;
        const size_t e = std::min(previous_chunk.size(), b + win);
        metrics[wi].rms = rms_range(previous_chunk, b, e);
        metrics[wi].peak = peak_range(previous_chunk, b, e);
        metrics[wi].zcr = zcr_range(previous_chunk, b, e);
        metrics[wi].diff = diff_ratio_range(previous_chunk, b, e);
        metrics[wi].voiced = metrics[wi].rms >= gate && metrics[wi].peak < 0.98;
    }

    double best_score = -std::numeric_limits<double>::infinity();
    size_t best_begin = 0;
    size_t best_end = 0;
    for (size_t run_begin = first_win; run_begin < n_win;) {
        while (run_begin < n_win && !metrics[run_begin].voiced) ++run_begin;
        if (run_begin >= n_win) break;
        size_t run_end = run_begin;
        while (run_end < n_win && metrics[run_end].voiced) ++run_end;

        for (size_t end_win = run_end; end_win > run_begin; --end_win) {
            const size_t end_sample = std::min(previous_chunk.size(), end_win * win);
            const size_t begin_min = end_sample > target ? end_sample - target : 0;
            const size_t begin_win = std::max(run_begin, begin_min / win);
            const size_t begin_sample = begin_win * win;
            if (end_sample <= begin_sample || end_sample - begin_sample < min_tail) continue;

            double rms_sum = 0.0;
            double zcr_sum = 0.0;
            double diff_sum = 0.0;
            double peak_max = 0.0;
            size_t used = 0;
            for (size_t wi = begin_win; wi < end_win; ++wi) {
                rms_sum += metrics[wi].rms;
                zcr_sum += metrics[wi].zcr;
                diff_sum += metrics[wi].diff;
                peak_max = std::max(peak_max, metrics[wi].peak);
                ++used;
            }
            if (used == 0) continue;
            const double avg_rms = rms_sum / used;
            const double avg_zcr = zcr_sum / used;
            const double avg_diff = diff_sum / used;
            const double recency = static_cast<double>(end_sample) / static_cast<double>(previous_chunk.size());
            const double clip_penalty = peak_max > 0.90 ? (peak_max - 0.90) * 4.0 : 0.0;
            const double score = std::log(avg_rms + 1.0e-6) - avg_diff * 0.35 - avg_zcr * 0.75 - clip_penalty + recency * 0.20;
            if (score > best_score) {
                best_score = score;
                best_begin = begin_sample;
                best_end = end_sample;
            }
        }
        run_begin = run_end;
    }

    if (best_end <= best_begin || best_end - best_begin < min_tail) return {};
    std::vector<float> tail(previous_chunk.begin() + static_cast<std::ptrdiff_t>(best_begin),
                            previous_chunk.begin() + static_cast<std::ptrdiff_t>(best_end));
    if (stats_out) {
        stats_out->begin = best_begin;
        stats_out->end = best_end;
        stats_out->mean = mean_range(tail, 0, tail.size());
        stats_out->rms = rms_range(tail, 0, tail.size());
        stats_out->peak = peak_range(tail, 0, tail.size());
        stats_out->zcr = zcr_range(tail, 0, tail.size());
        stats_out->diff_ratio = diff_ratio_range(tail, 0, tail.size());
    }
    return tail;
}

void sanitize_clean_tail(std::vector<float>* tail,
                         const std::vector<float>& original_ref,
                         int sample_rate,
                         CleanTailStats* stats) {
    if (!tail || tail->empty()) return;
    const double mean = mean_range(*tail, 0, tail->size());
    for (float& v : *tail) v = static_cast<float>(static_cast<double>(v) - mean);

    const double tail_rms = rms_range(*tail, 0, tail->size());
    const double ref_rms = voiced_rms_reference(original_ref, sample_rate);
    double gain = 1.0;
    if (ref_rms > 1.0e-6 && tail_rms > 1.0e-6) {
        gain = std::clamp(ref_rms / tail_rms, 0.50, 1.25);
    }
    for (float& v : *tail) v = static_cast<float>(std::clamp(static_cast<double>(v) * gain, -0.95, 0.95));

    const size_t fade = std::min(tail->size() / 2, std::max<size_t>(1, static_cast<size_t>(sample_rate) / 100)); // 10 ms
    for (size_t i = 0; i < fade; ++i) {
        const float g = static_cast<float>(i + 1) / static_cast<float>(fade + 1);
        (*tail)[i] *= g;
        (*tail)[tail->size() - 1 - i] *= g;
    }
    if (stats) {
        stats->gain = gain;
        stats->rms = rms_range(*tail, 0, tail->size());
        stats->peak = peak_range(*tail, 0, tail->size());
    }
}

int count_words_for_log(const std::string& text) {
    static const std::regex word_re(R"([A-Za-z0-9]+(?:'[A-Za-z0-9]+)?)");
    return static_cast<int>(std::distance(std::sregex_iterator(text.begin(), text.end(), word_re),
                                          std::sregex_iterator()));
}

bool write_clean_tail_reference_wav(const std::string& original_ref_path,
                                    const std::vector<float>& previous_chunk,
                                    int sample_rate,
                                    int tail_ms,
                                    const VibeVoiceModel* model,
                                    size_t chunk_index,
                                    std::string* path_out) {
    if (!path_out) return false;
    path_out->clear();
    std::vector<float> original_ref;
    if (load_wav_24k_mono(original_ref_path, &original_ref) != 0 || original_ref.empty()) {
        VV_LOG_WARN("tts_15b: clean-tail-reference continuity could not load original reference audio");
        return false;
    }

    CleanTailStats stats;
    std::vector<float> tail = extract_clean_continuity_tail(previous_chunk, sample_rate, tail_ms, &stats);
    if (tail.empty()) {
        VV_LOG_WARN("tts_15b: clean-tail-reference continuity skipped: no clean voiced previous-tail island found");
        return false;
    }
    const double raw_rms = stats.rms;
    const double raw_peak = stats.peak;
    const double raw_zcr = stats.zcr;
    const double raw_diff = stats.diff_ratio;
    sanitize_clean_tail(&tail, original_ref, sample_rate, &stats);

    std::vector<float> combined;
    combined.reserve(original_ref.size() + static_cast<size_t>(sample_rate / 10) + tail.size());
    combined.insert(combined.end(), original_ref.begin(), original_ref.end());
    combined.insert(combined.end(), static_cast<size_t>(sample_rate / 10), 0.0f);
    combined.insert(combined.end(), tail.begin(), tail.end());

    const std::string tmp_path = continuity_temp_ref_path(model, chunk_index);
    vv_audio audio{};
    audio.samples = combined.data();
    audio.n_samples = combined.size();
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    if (save_wav_pcm16(tmp_path, audio) != 0) {
        VV_LOG_WARN("tts_15b: clean-tail-reference continuity could not write temporary reference wav: %s", tmp_path.c_str());
        return false;
    }
    *path_out = tmp_path;
    VV_LOG_INFO("tts_15b: clean-tail-reference chunk=%zu tail_ms=%d tail_samples=%zu source_ms=%.1f..%.1f raw_rms=%.5f raw_peak=%.5f zcr=%.3f diff=%.3f gain=%.3f aligned_rms=%.5f aligned_peak=%.5f",
                chunk_index + 1, tail_ms, tail.size(),
                1000.0 * static_cast<double>(stats.begin) / sample_rate,
                1000.0 * static_cast<double>(stats.end) / sample_rate,
                raw_rms, raw_peak, raw_zcr, raw_diff, stats.gain, stats.rms, stats.peak);
    return true;
}

int tts_15b_generate(VibeVoiceModel*            model,
                     const std::string&         text,
                     const VibeVoiceTTSParams&  p,
                     std::vector<float>*        samples) {
    if (!model || !samples) return -1;

    const bool is_kugelaudio = model->loader.has_key("kugelaudio.architecture");
    const std::string generation_text = is_kugelaudio ? apply_text_end_padding(text, p.text_end_padding) : text;
    if (!is_kugelaudio || p.max_words_per_chunk <= 0) {
        if (is_kugelaudio && p.verbose && generation_text != text) {
            std::fprintf(stderr, "[tts_15b] text-end-padding generation_text: %s\n", generation_text.c_str());
        }
        return tts_15b_generate_single_pass(model, generation_text, p, samples);
    }
    if (p.crossfade_ms < 0) {
        VV_LOG_ERROR("tts_15b: crossfade_ms must be >= 0 (got %d)", p.crossfade_ms);
        return -23;
    }
    if (p.overlap_sentences < 0) {
        VV_LOG_ERROR("tts_15b: overlap_sentences must be >= 0 (got %d)", p.overlap_sentences);
        return -26;
    }
    if (p.continuity_tail_ms < 0) {
        VV_LOG_ERROR("tts_15b: continuity_tail_ms must be >= 0 (got %d)", p.continuity_tail_ms);
        return -27;
    }
    if (p.chunk_boundary_leading_silence_ms < 0 ||
        p.chunk_boundary_trailing_silence_ms < 0 ||
        p.chunk_boundary_fade_ms < 0) {
        VV_LOG_ERROR("tts_15b: chunk boundary cleanup values must be >= 0 (leading=%d trailing=%d fade=%d)",
                     p.chunk_boundary_leading_silence_ms,
                     p.chunk_boundary_trailing_silence_ms,
                     p.chunk_boundary_fade_ms);
        return -33;
    }
    if (p.chunk_eos_guard_frames < 0) {
        VV_LOG_ERROR("tts_15b: chunk_eos_guard_frames must be >= 0 (got %d)", p.chunk_eos_guard_frames);
        return -34;
    }
    const bool allow_retired_continuity = kugelaudio_env_truthy(kugelaudio_getenv("VIBEVOICE_ENABLE_RETIRED_CONTINUITY"));
    if (p.chunk_continuity == ChunkContinuityMode::TailReference) {
        if (!allow_retired_continuity) {
            VV_LOG_ERROR("tts_15b: tail-reference chunk continuity is retired because it feeds decoded waveform noise back into later chunks; use none or single-sequence");
            return -28;
        }
        VV_LOG_WARN("tts_15b: using retired tail-reference continuity because VIBEVOICE_ENABLE_RETIRED_CONTINUITY=1");
    }
    if (p.chunk_continuity == ChunkContinuityMode::CleanTailReference) {
        if (!allow_retired_continuity) {
            VV_LOG_ERROR("tts_15b: clean-tail-reference chunk continuity is retired because sanitized generated waveform feedback still caused noise and speaker drift; use none or single-sequence");
            return -30;
        }
        VV_LOG_WARN("tts_15b: using retired clean-tail-reference continuity because VIBEVOICE_ENABLE_RETIRED_CONTINUITY=1");
    }
    if (p.chunk_continuity == ChunkContinuityMode::LatentPrefix) {
        if (!allow_retired_continuity) {
            VV_LOG_ERROR("tts_15b: latent-prefix chunk continuity is retired because listening showed worse noise and intonation drift; use none or single-sequence");
            return -29;
        }
        VV_LOG_WARN("tts_15b: using retired latent-prefix continuity because VIBEVOICE_ENABLE_RETIRED_CONTINUITY=1");
    }
    if (p.chunk_continuity == ChunkContinuityMode::PromptInstruction) {
        if (!allow_retired_continuity) {
            VV_LOG_ERROR("tts_15b: prompt-instruction chunk continuity is retired because listening showed speaker identity drift; use none, single-sequence, or segmented-state");
            return -31;
        }
        VV_LOG_WARN("tts_15b: using retired prompt-instruction continuity because VIBEVOICE_ENABLE_RETIRED_CONTINUITY=1");
    }

    detail::KugelAudioChunkPlan plan;
    std::string plan_error;
    if (!detail::split_kugelaudio_text_into_chunks(text, p.max_words_per_chunk, p.overlap_sentences,
                                                   p.chunking_strategy, &plan, &plan_error)) {
        VV_LOG_ERROR("tts_15b: chunking plan failed: %s", plan_error.c_str());
        return -24;
    }
    if (plan.chunks.size() <= 1) {
        VV_LOG_INFO("tts_15b: chunking enabled but skipped: single chunk (max_words_per_chunk=%d)",
                    p.max_words_per_chunk);
        if (p.verbose && generation_text != text) {
            std::fprintf(stderr, "[tts_15b] text-end-padding generation_text: %s\n", generation_text.c_str());
        }
        return tts_15b_generate_single_pass(model, generation_text, p, samples);
    }

    if (p.chunk_continuity == ChunkContinuityMode::SegmentedState) {
        return tts_15b_generate_segmented_state(model, plan, p, samples);
    }

    if (p.chunk_continuity == ChunkContinuityMode::SingleSequence) {
        VibeVoiceTTSParams single_params = p;
        single_params.max_words_per_chunk = 0;
        single_params.overlap_sentences = 0;
        single_params.chunk_continuity = ChunkContinuityMode::None;
        const int64_t requested_frames = static_cast<int64_t>(std::max(1, p.max_speech_frames)) *
                                         static_cast<int64_t>(plan.chunks.size());
        single_params.max_speech_frames = static_cast<int>(std::min<int64_t>(requested_frames, std::numeric_limits<int>::max()));
        const float min_ratio = kugelaudio_single_sequence_min_ratio();
        single_params.min_speech_frames = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(std::floor(static_cast<double>(requested_frames) * static_cast<double>(min_ratio))),
            static_cast<int64_t>(single_params.max_speech_frames)));
        VV_LOG_INFO("tts_15b: single-sequence continuity: prefill full text once chunks=%zu per_chunk_frames=%d total_frame_budget=%d min_frames=%d min_ratio=%.3f",
                    plan.chunks.size(), p.max_speech_frames, single_params.max_speech_frames,
                    single_params.min_speech_frames, static_cast<double>(min_ratio));
        if (p.verbose && generation_text != text) {
            std::fprintf(stderr, "[tts_15b] text-end-padding single_sequence_generation_text: %s\n", generation_text.c_str());
        }
        return tts_15b_generate_single_pass(model, generation_text, single_params, samples);
    }

    std::vector<std::string> generation_chunks = plan.chunks;
    if (p.text_end_padding != TextEndPaddingMode::None && !generation_chunks.empty()) {
        generation_chunks.back() = apply_text_end_padding(generation_chunks.back(), p.text_end_padding);
    }

    VV_LOG_INFO("tts_15b: chunking enabled chunks=%zu boundary_types=%s strategy=%s overlap_sentences=%d pause_mode=%s crossfade_ms=%d max_words_per_chunk=%d continuity=%s continuity_tail_ms=%d",
                plan.chunks.size(),
                describe_chunk_boundary_types(plan.boundary_types).c_str(),
                vv::chunking_strategy_name(p.chunking_strategy),
                p.overlap_sentences,
                vv::chunk_pause_mode_name(p.pause_mode),
                p.crossfade_ms,
                p.max_words_per_chunk,
                vv::chunk_continuity_mode_name(p.chunk_continuity),
                p.continuity_tail_ms);
    if (p.verbose) {
        for (size_t i = 0; i < plan.chunks.size(); ++i) {
            const std::string overlap = i < plan.overlap_prefixes.size() ? plan.overlap_prefixes[i] : std::string();
            const std::string new_text = i < plan.new_texts.size() ? plan.new_texts[i] : plan.chunks[i];
            const int overlap_words = count_words_for_log(overlap);
            const int new_words = count_words_for_log(new_text);
            std::fprintf(stderr,
                         "[chunking] chunk %zu/%zu budget overlap_words=%d new_words=%d total_words=%d max_words=%d boundary=%s\n",
                         i + 1, plan.chunks.size(), overlap_words, new_words,
                         overlap_words + new_words, p.max_words_per_chunk,
                         i < plan.boundary_types.size() ? detail::kugelaudio_chunk_boundary_type_name(plan.boundary_types[i]) : "unknown");
            if (!overlap.empty()) std::fprintf(stderr, "[chunking] chunk %zu overlap: %s\n", i + 1, overlap.c_str());
            std::fprintf(stderr, "[chunking] chunk %zu new: %s\n", i + 1, new_text.c_str());
            std::fprintf(stderr, "[chunking] chunk %zu prompt: %s\n", i + 1, plan.chunks[i].c_str());
            if (generation_chunks[i] != plan.chunks[i]) {
                std::fprintf(stderr, "[chunking] chunk %zu generation_prompt: %s\n", i + 1, generation_chunks[i].c_str());
            }
        }
    }

    VibeVoiceTTSParams chunk_params = p;
    chunk_params.max_words_per_chunk = 0;
    chunk_params.overlap_sentences = 0;
    chunk_params.chunking_strategy = ChunkingStrategy::Heuristic;

    PreparedSpeechConditioning cached_conditioning;
    const PreparedSpeechConditioning* prepared_conditioning = nullptr;
    if ((p.chunk_continuity == ChunkContinuityMode::None ||
         p.chunk_continuity == ChunkContinuityMode::PromptInstruction) &&
        !should_short_circuit_after_conditioning_for_test() &&
        !should_short_circuit_after_connectors_for_test()) {
        std::mt19937 conditioning_rng(p.seed ? p.seed : std::random_device{}());
        PreparedSpeechConditioningResult prep = prepare_speech_conditioning(model,
                                                                            p,
                                                                            is_kugelaudio,
                                                                            conditioning_rng,
                                                                            &cached_conditioning);
        if (prep.rc != 0) return prep.rc;
        prepared_conditioning = &cached_conditioning;
        VV_LOG_INFO("tts_15b: reusing cached raw-reference conditioning across %zu chunks (frames=%d)",
                    plan.chunks.size(), cached_conditioning.total_Tc);
    }

    ChunkLatentContinuityState latent_carry;
    std::vector<std::vector<float>> chunk_outputs;
    chunk_outputs.reserve(plan.chunks.size());
    for (size_t i = 0; i < plan.chunks.size(); ++i) {
        if (p.verbose) {
            std::fprintf(stderr, "[chunking] generating chunk %zu/%zu\n", i + 1, plan.chunks.size());
        }

        VibeVoiceTTSParams this_chunk_params = chunk_params;
        if (p.chunk_continuity == ChunkContinuityMode::PromptInstruction) {
            this_chunk_params.prompt_continuity_instruction = build_prompt_continuity_instruction();
        }
        const PreparedSpeechConditioning* this_conditioning = prepared_conditioning;
        std::string temp_ref_path;
        if (p.chunk_continuity == ChunkContinuityMode::TailReference && i > 0 && !chunk_outputs.empty()) {
            if (write_tail_reference_wav(p.ref_audio_paths.front(),
                                         chunk_outputs.back(),
                                         model->cfg.sample_rate,
                                         p.continuity_tail_ms,
                                         model,
                                         i,
                                         &temp_ref_path)) {
                this_chunk_params.ref_audio_paths = {temp_ref_path};
                this_conditioning = nullptr;
            }
        } else if (p.chunk_continuity == ChunkContinuityMode::CleanTailReference && i > 0 && !chunk_outputs.empty()) {
            if (write_clean_tail_reference_wav(p.ref_audio_paths.front(),
                                               chunk_outputs.back(),
                                               model->cfg.sample_rate,
                                               p.continuity_tail_ms,
                                               model,
                                               i,
                                               &temp_ref_path)) {
                this_chunk_params.ref_audio_paths = {temp_ref_path};
                this_conditioning = nullptr;
            }
        }

        const ChunkLatentContinuityState* latent_prefix =
            (p.chunk_continuity == ChunkContinuityMode::LatentPrefix && i > 0 && latent_carry.frames > 0)
                ? &latent_carry
                : nullptr;
        ChunkLatentContinuityState next_latent_carry;
        std::vector<float> chunk_samples;
        const int rc = tts_15b_generate_single_pass(model, generation_chunks[i], this_chunk_params, &chunk_samples,
                                                    this_conditioning,
                                                    latent_prefix,
                                                    p.chunk_continuity == ChunkContinuityMode::LatentPrefix ? &next_latent_carry : nullptr);
        if (!temp_ref_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(temp_ref_path, ec);
        }
        if (rc != 0) return rc;
        if (p.chunk_continuity == ChunkContinuityMode::LatentPrefix) {
            latent_carry = std::move(next_latent_carry);
        }
        if (p.verbose) {
            const int sr = model->cfg.sample_rate;
            const double quiet_threshold = 0.003;
            const size_t win50 = sr > 0 ? static_cast<size_t>(sr) / 20 : 0;
            const size_t win200 = sr > 0 ? static_cast<size_t>(sr) / 5 : 0;
            const size_t n = chunk_samples.size();
            const double duration_s = sr > 0 ? static_cast<double>(n) / static_cast<double>(sr) : 0.0;
            const size_t head_quiet = leading_quiet_samples(chunk_samples, sr, quiet_threshold);
            const size_t tail_quiet = trailing_quiet_samples(chunk_samples, sr, quiet_threshold);
            std::fprintf(stderr,
                         "[chunking] chunk %zu audio duration=%.3fs samples=%zu head_quiet_ms=%.1f tail_quiet_ms=%.1f head50_rms=%.5f tail50_rms=%.5f tail200_rms=%.5f tail50_peak=%.5f\n",
                         i + 1,
                         duration_s,
                         n,
                         sr > 0 ? 1000.0 * static_cast<double>(head_quiet) / static_cast<double>(sr) : 0.0,
                         sr > 0 ? 1000.0 * static_cast<double>(tail_quiet) / static_cast<double>(sr) : 0.0,
                         rms_range(chunk_samples, 0, win50),
                         n > win50 ? rms_range(chunk_samples, n - win50, n) : rms_range(chunk_samples, 0, n),
                         n > win200 ? rms_range(chunk_samples, n - win200, n) : rms_range(chunk_samples, 0, n),
                         n > win50 ? peak_range(chunk_samples, n - win50, n) : peak_range(chunk_samples, 0, n));
        }
        chunk_outputs.push_back(std::move(chunk_samples));
    }

    if (p.chunk_boundary_cleanup) {
        if (p.verbose) {
            std::fprintf(stderr,
                         "[chunking] cleanup enabled leading_keep_ms=%d trailing_keep_ms=%d fade_ms=%d\n",
                         p.chunk_boundary_leading_silence_ms,
                         p.chunk_boundary_trailing_silence_ms,
                         p.chunk_boundary_fade_ms);
        }
        apply_chunk_boundary_cleanup(chunk_outputs,
                                     model->cfg.sample_rate,
                                     p.chunk_boundary_leading_silence_ms,
                                     p.chunk_boundary_trailing_silence_ms,
                                     p.chunk_boundary_fade_ms,
                                     p.verbose);
    }

    if (p.verbose && chunk_outputs.size() > 1) {
        const int sr = model->cfg.sample_rate;
        const double quiet_threshold = 0.003;
        for (size_t i = 1; i < chunk_outputs.size(); ++i) {
            const int pause_ms = detail::kugelaudio_pause_duration_ms_for_boundary(plan.new_texts[i - 1],
                                                                                   plan.new_texts[i],
                                                                                   p.pause_mode);
            const int effective_programmed_gap_ms = std::max(0, pause_ms - p.crossfade_ms);
            const size_t left_tail_quiet = trailing_quiet_samples(chunk_outputs[i - 1], sr, quiet_threshold);
            const size_t right_head_quiet = leading_quiet_samples(chunk_outputs[i], sr, quiet_threshold);
            const double left_tail_ms = sr > 0 ? 1000.0 * static_cast<double>(left_tail_quiet) / static_cast<double>(sr) : 0.0;
            const double right_head_ms = sr > 0 ? 1000.0 * static_cast<double>(right_head_quiet) / static_cast<double>(sr) : 0.0;
            const double estimated_gap_ms = std::max(0.0, left_tail_ms + static_cast<double>(pause_ms) + right_head_ms - static_cast<double>(p.crossfade_ms));
            std::fprintf(stderr,
                         "[chunking] boundary %zu->%zu pause_ms=%d crossfade_ms=%d effective_zero_gap_ms=%d estimated_audible_gap_ms=%.1f left_tail_quiet_ms=%.1f right_head_quiet_ms=%.1f left_tail50_rms=%.5f right_head50_rms=%.5f\n",
                         i,
                         i + 1,
                         pause_ms,
                         p.crossfade_ms,
                         effective_programmed_gap_ms,
                         estimated_gap_ms,
                         left_tail_ms,
                         right_head_ms,
                         chunk_outputs[i - 1].empty() ? 0.0 : rms_range(chunk_outputs[i - 1],
                                                                         chunk_outputs[i - 1].size() > static_cast<size_t>(std::max(1, sr / 20)) ? chunk_outputs[i - 1].size() - static_cast<size_t>(std::max(1, sr / 20)) : 0,
                                                                         chunk_outputs[i - 1].size()),
                         rms_range(chunk_outputs[i], 0, sr > 0 ? static_cast<size_t>(sr) / 20 : 0));
        }
    }

    std::string stitch_error;
    if (!detail::stitch_kugelaudio_audio_chunks(chunk_outputs,
                                                plan.new_texts,
                                                p.pause_mode,
                                                p.crossfade_ms,
                                                model->cfg.sample_rate,
                                                samples,
                                                &stitch_error)) {
        VV_LOG_ERROR("tts_15b: chunk stitching failed: %s", stitch_error.c_str());
        return -25;
    }
    VV_LOG_INFO("tts_15b: chunking stitched %zu chunks into %zu samples",
                plan.chunks.size(), samples->size());
    return 0;
}
}

int vibevoice_tts_generate(VibeVoiceModel*           model,
                           const std::string&        text,
                           const VibeVoiceTTSParams& p,
                           std::vector<float>*       samples) {
    if (!model || !samples) return -1;

    // Variant dispatch — the gguf already knows which path it wants;
    // callers don't need a separate entry point per model family.
    if (model->variant == "1.5b") {
        if (p.ref_audio_paths.empty()) {
            VV_LOG_ERROR("vibevoice_tts_generate: 1.5b model requires "
                         "at least one VibeVoiceTTSParams::ref_audio_paths "
                         "entry (single-speaker: pass a one-element vector)");
            return -1;
        }
        return tts_15b_generate(model, text, p, samples);
    }

    const auto& cfg = model->cfg;
    const auto& w   = model->w;

    // ---- 1. tokenize ----
    if (!model->tokenizer.vocab_size()) {
        VV_LOG_ERROR("vibevoice_tts_generate: tokenizer not loaded");
        return -2;
    }
    // mlx-audio convention: append "\n" to terminate the user turn so the
    // model knows the input is complete and starts the speech reply. The
    // upstream voice prefix ends mid-conversation; without a separator, the
    // model treats the new text as a continuation of the prior turn.
    std::string text_with_sep = text;
    while (!text_with_sep.empty() &&
           (text_with_sep.back() == ' ' || text_with_sep.back() == '\t' ||
            text_with_sep.back() == '\r' || text_with_sep.back() == '\n')) {
        text_with_sep.pop_back();
    }
    text_with_sep += "\n";
    const auto text_ids = model->tokenizer.encode(text_with_sep);
    if (text_ids.empty()) {
        VV_LOG_ERROR("tokenizer produced no tokens");
        return -3;
    }
    if (p.verbose) {
        std::fprintf(stderr, "[tts] %zu input text tokens\n", text_ids.size());
    }

    // ---- 2. embed text via lm.tok_embd ----
    // tok_embd is [hidden, vocab_size] in ggml after gguf load. May be fp32
    // or fp16 depending on the converter --dtype. Lives on the active
    // backend's buffer, so each row is fetched via ggml_backend_tensor_get
    // (a memcpy on CPU; a DtoH transfer on CUDA / Metal / Vulkan).
    const int hidden = cfg.hidden;
    const int n_text = static_cast<int>(text_ids.size());
    std::vector<float> text_embeds(static_cast<size_t>(hidden) * n_text);

    if (w.lm_tok_embd->type == GGML_TYPE_F32) {
        const size_t row = sizeof(float) * hidden;
        for (int t = 0; t < n_text; ++t) {
            const int id = text_ids[t];
            if (id < 0 || id >= cfg.vocab_size) {
                VV_LOG_ERROR("token id out of range: %d", id);
                return -5;
            }
            ggml_backend_tensor_get(w.lm_tok_embd, &text_embeds[hidden * t],
                                    row * static_cast<size_t>(id), row);
        }
    } else if (w.lm_tok_embd->type == GGML_TYPE_F16) {
        const size_t row = sizeof(ggml_fp16_t) * hidden;
        std::vector<ggml_fp16_t> staged(hidden);
        for (int t = 0; t < n_text; ++t) {
            const int id = text_ids[t];
            if (id < 0 || id >= cfg.vocab_size) {
                VV_LOG_ERROR("token id out of range: %d", id);
                return -5;
            }
            ggml_backend_tensor_get(w.lm_tok_embd, staged.data(),
                                    row * static_cast<size_t>(id), row);
            for (int i = 0; i < hidden; ++i) {
                text_embeds[hidden * t + i] = ggml_fp16_to_fp32(staged[i]);
            }
        }
    } else {
        VV_LOG_ERROR("vibevoice_tts_generate: lm.tok_embd unsupported dtype %d",
                     static_cast<int>(w.lm_tok_embd->type));
        return -4;
    }

    // CFG parallel state probe (reads from voice prompt).
    const bool use_cfg = p.voice && p.voice->has_neg && p.cfg_scale > 1.0f;

    // Resident K/V caches. lm and tlm grow with text + (tlm only) speech
    // frames. neg_tlm starts at the negative voice's seq_neg_tlm and
    // grows by the same speech-frame count as tlm. We size for the
    // pessimistic upper bound: voice prefix + n_text + max_speech_frames.
    const int hd   = cfg.head_dim;
    const int n_kv = cfg.n_kv_heads;
    const int max_lm_seq      = (p.voice ? p.voice->seq_lm     : 0) + n_text                       + 32;
    const int max_tlm_seq     = (p.voice ? p.voice->seq_tlm    : 0) + n_text + p.max_speech_frames + 32;
    const int max_neg_tlm_seq = (use_cfg ? p.voice->seq_neg_tlm : 0)         + p.max_speech_frames + 32;

    ResidentKV kv_lm, kv_tlm, kv_neg_tlm;
    if (!kv_lm.init (cfg.n_layers_lm,  hd, n_kv, max_lm_seq))  return -5;
    if (!kv_tlm.init(cfg.n_layers_tlm, hd, n_kv, max_tlm_seq)) return -5;
    if (use_cfg && !kv_neg_tlm.init(cfg.n_layers_tlm, hd, n_kv, max_neg_tlm_seq)) return -5;

    // Upload voice-prompt KV into the resident buffers. After this the
    // host-side voice->kv_* vectors are no longer touched - the resident
    // tensors are the source of truth for the rest of the call.
    auto upload_kv = [hd, n_kv](ResidentKV& dst,
                                const std::vector<LayerKV>& src,
                                int seq_len) {
        for (size_t li = 0; li < src.size(); ++li) {
            const size_t per = static_cast<size_t>(hd) * n_kv * seq_len;
            vv::backend_tensor_set(dst.k[li], src[li].k.data(), 0, sizeof(float) * per);
            vv::backend_tensor_set(dst.v[li], src[li].v.data(), 0, sizeof(float) * per);
        }
        dst.past_len = seq_len;
    };
    if (p.voice) {
        upload_kv(kv_lm,  p.voice->kv_lm,  p.voice->seq_lm);
        upload_kv(kv_tlm, p.voice->kv_tlm, p.voice->seq_tlm);
        if (use_cfg) upload_kv(kv_neg_tlm, p.voice->kv_neg_tlm, p.voice->seq_neg_tlm);
    }

    int                  lm_pos  = p.voice ? p.voice->seq_lm  : 0;
    int                  tlm_pos = p.voice ? p.voice->seq_tlm : 0;
    std::vector<float>   tlm_hidden_last = p.voice ? p.voice->tlm_last_hidden
                                                   : std::vector<float>(hidden, 0.0f);
    int                  neg_tlm_pos = use_cfg ? p.voice->seq_neg_tlm : 0;
    std::vector<float>   neg_tlm_hidden_last = use_cfg ? p.voice->neg_tlm_last_hidden
                                                       : std::vector<float>{};
    if (p.verbose) std::fprintf(stderr, "[tts] cfg=%s (scale=%.2f)\n",
                                use_cfg ? "on" : "off",
                                static_cast<double>(p.cfg_scale));

    // mlx-audio + upstream alternate text windows (5 tokens) with speech
    // windows (6 frames). The model is trained on this pattern; processing
    // all text up front then generating all speech misleads it.
    constexpr int kTextWindow   = 5;
    constexpr int kSpeechWindow = 6;

    // ---- diffusion + RNG setup ----
    DPMSolverConfig solver_cfg;
    solver_cfg.num_train_timesteps = 1000;
    solver_cfg.num_inference_steps = p.n_diffusion_steps;
    solver_cfg.solver_order        = 2;
    solver_cfg.lower_order_final   = true;
    DPMSolverState solver_state;
    dpm_solver_init(solver_cfg, &solver_state);

    DiffusionHeadConfig dh_cfg;
    dh_cfg.hidden      = hidden;
    dh_cfg.latent      = cfg.latent;
    dh_cfg.head_layers = cfg.head_layers;
    dh_cfg.ffn_ratio   = cfg.ffn_ratio;
    dh_cfg.eps         = cfg.rms_norm_eps;
    dh_cfg.freq_size   = 256;

    std::mt19937 rng(p.seed ? p.seed : std::random_device{}());
    std::normal_distribution<float> norm(0.0f, 1.0f);

    samples->clear();
    std::vector<float> all_latents;
    all_latents.reserve(static_cast<size_t>(p.max_speech_frames) * cfg.latent);

    int  text_pos = 0;
    bool finished = false;
    int  total_frames = 0;

    while (!finished && total_frames < p.max_speech_frames) {
        // ---- text window (up to 5 tokens) ----
        if (text_pos < n_text) {
            const int win = std::min(kTextWindow, n_text - text_pos);
            // Slice the text embeddings for this window.
            std::vector<float> emb_win(static_cast<size_t>(hidden) * win);
            std::memcpy(emb_win.data(),
                        text_embeds.data() + static_cast<size_t>(hidden) * text_pos,
                        sizeof(float) * hidden * win);

            // LM forward
            std::vector<float> lm_hidden;
            if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, /*output_norm=*/nullptr,
                                 lm_pos, win, emb_win.data(), &kv_lm,
                                 &lm_hidden, nullptr)) return -6;
            lm_pos += win;

            // TTS-LM input = LM hidden + text-type embedding
            std::vector<float> tlm_in = lm_hidden;
            add_input_type_embedding(cfg, w, win, /*type=*/1, tlm_in.data());

            if (!run_qwen2_stack(nullptr, cfg, w.tlm_layers,
                                 /*output_norm=*/w.tlm_output_norm,
                                 tlm_pos, win, tlm_in.data(), &kv_tlm,
                                 nullptr, &tlm_hidden_last)) return -8;
            tlm_pos += win;

            text_pos += win;
            if (p.verbose) std::fprintf(stderr,
                "[tts] text window: %d/%d tokens consumed\n", text_pos, n_text);
        }

        // ---- speech window (up to 6 frames or until EOS) ----
        const int sp_budget = std::min(kSpeechWindow, p.max_speech_frames - total_frames);
        for (int sp = 0; sp < sp_budget; ++sp) {
            const int frame = total_frames;
        // 5a. Sample one speech latent from the diffusion head.
        std::vector<float> z(cfg.latent);
        for (auto& v : z) v = norm(rng);

        std::vector<float> cond(static_cast<size_t>(hidden));
        std::memcpy(cond.data(), tlm_hidden_last.data(), sizeof(float) * hidden);

        // dpm_solver_sample expects shape [latent * frames * batch] with
        // frames=1, B=1 here. cond shape: [hidden * 1 * 1].
        std::vector<float> cond_neg;
        if (use_cfg) {
            cond_neg.assign(neg_tlm_hidden_last.begin(), neg_tlm_hidden_last.end());
        }
        if (dpm_solver_sample(z, cfg.latent, /*frames=*/1, /*batch=*/1,
                              cond, hidden,
                              w.dh, dh_cfg, solver_cfg, solver_state,
                              cond_neg, p.cfg_scale) != 0) {
            VV_LOG_ERROR("dpm_solver_sample failed at frame %d", frame);
            return -9;
        }

        // 5b. Buffer this latent (decoder runs on the full sequence later).
        all_latents.insert(all_latents.end(), z.begin(), z.end());

        // 5c. Project latent through connector → next-step embedding.
        auto ac_embed = run_speech_connector(cfg, w, z.data(), /*batch=*/1);
        // Add tts_input_types[0] (speech type)
        add_input_type_embedding(cfg, w, /*n_tokens=*/1, /*type=*/0, ac_embed.data());

        // 5d. Step TTS-LM by one position (positive branch).
            if (!run_qwen2_stack(nullptr, cfg, w.tlm_layers, /*output_norm=*/w.tlm_output_norm,
                                 tlm_pos, /*n_new=*/1, ac_embed.data(), &kv_tlm,
                                 nullptr, &tlm_hidden_last)) {
                VV_LOG_ERROR("TTS-LM speech step failed at frame %d", frame);
                return -10;
            }
            tlm_pos += 1;

            if (use_cfg) {
                std::vector<float> ac_embed_neg(ac_embed);
                if (!run_qwen2_stack(nullptr, cfg, w.tlm_layers,
                                     /*output_norm=*/w.tlm_output_norm,
                                     neg_tlm_pos, /*n_new=*/1, ac_embed_neg.data(),
                                     &kv_neg_tlm, nullptr, &neg_tlm_hidden_last)) {
                    VV_LOG_ERROR("neg TTS-LM speech step failed at frame %d", frame);
                    return -10;
                }
                neg_tlm_pos += 1;
            }

            const float eos = run_eos_classifier(cfg, w, tlm_hidden_last.data());
            if (p.verbose && (total_frames % 4 == 0 || eos > 0.5f)) {
                std::fprintf(stderr, "[tts] frame %d: eos=%.3f, %d latents\n",
                             total_frames, eos,
                             static_cast<int>(all_latents.size() / cfg.latent));
            }
            ++total_frames;
            if (eos > 0.5f) {
                if (p.verbose) std::fprintf(stderr, "[tts] EOS at frame %d\n", frame);
                finished = true;
                break;
            }
        }

        // If we've consumed all text AND not finished, the outer loop will
        // continue with empty-text iterations (just speech) until EOS or cap.
        if (text_pos >= n_text && !finished) {
            // No more text — keep generating speech until EOS or budget.
        }
    }

    // ---- 6. decode the full latent sequence in one pass ----
    const int n_frames = static_cast<int>(all_latents.size() / cfg.latent);
    if (n_frames > 0) {
        // scaled = latent / speech_scaling - speech_bias  (mlx-audio order)
        std::vector<float> scaled(all_latents.size());
        for (size_t i = 0; i < all_latents.size(); ++i) {
            scaled[i] = all_latents[i] / cfg.speech_scaling - cfg.speech_bias;
        }
        // The decoder expects the latent dim as the contiguous (innermost)
        // axis in ggml. Our `all_latents` is stored as N consecutive
        // [latent]-sized chunks in time order, which is exactly that layout
        // when reshaped to [n_frames, vae_dim] (numpy) -> ggml [vae_dim,
        // n_frames] -> oh wait we need to transpose. Re-pack as
        // [latent fastest, frames slower] (ggml ne[0]=n_frames, ne[1]=vae_dim
        // is wrong — should be ne[0]=vae_dim).
        //
        // Actually our existing decoder fixture / forward expects ne[0] =
        // T_compressed (frames). Let's check by tracing:
        //   encoder_forward output has ne = [T_compr, vae_dim, B]
        //   decoder_forward input  has ne = [T_compr, vae_dim, B]   (same)
        // So ne[0] is frames, ne[1] is vae_dim. Memory: frame fastest.
        // We append [latent] vectors per frame so memory is "latent fastest"
        // = mismatched. Need to transpose.
        std::vector<float> packed(scaled.size());
        for (int t = 0; t < n_frames; ++t) {
            for (int d = 0; d < cfg.latent; ++d) {
                packed[d * n_frames + t] = scaled[t * cfg.latent + d];
            }
        }
        auto audio = decode_latent_sequence(cfg, w, packed.data(), n_frames);
        *samples = std::move(audio);
    }

    if (p.verbose) std::fprintf(stderr, "[tts] decoded %zu samples from %d latents\n",
                                samples->size(), n_frames);
    return 0;
}

// ============================================================================
// 1.5B path
// ============================================================================
//
// The 1.5B model is structurally simpler than realtime: a single Qwen2.5-1.5B
// LM stack feeds the diffusion head directly (no `tts_lm` split, no `eos`
// classifier, no `tts.input_types` type embedding). Voice cloning is built
// in: the prompt has placeholder `<|vision_pad|>` tokens whose embeddings are
// replaced inline with features from the reference audio's at_enc + st_enc
// + connectors (summed). The LM emits `<|vision_end|>` when it has finished
// generating speech.
//
// Token IDs (from Qwen/Qwen2.5-1.5B vocab; the `VibeVoiceTextTokenizer`
// repurposes the unused `vision_*` tokens for speech):
//
//   speech_start     = <|vision_start|>     = 151652
//   speech_end       = <|vision_end|>       = 151653
//   speech_diffusion = <|vision_pad|>       = 151654

namespace {

constexpr int kSpeech15bStartId = 151652;   // <|vision_start|>
constexpr int kSpeech15bEndId   = 151653;   // <|vision_end|>
constexpr int kSpeech15bDiffId  = 151654;   // <|vision_pad|>
constexpr int kSpeech15bEosId   = 151643;   // Qwen EOS used by canonical speech loop
constexpr float kKugelAudioSpeechEndPenalty = 1.5f;
constexpr int kSpeech15bImgPadId = 151655;  // <|image_pad|> — negative branch
constexpr int kSpeech15bCompressRatio = 3200;

double generated_duration_seconds_for_frames(int frames, int sample_rate) {
    if (frames <= 0 || sample_rate <= 0) return 0.0;
    return (static_cast<double>(frames) * kSpeech15bCompressRatio) / sample_rate;
}

void log_tts_frame_progress(int frames, int max_frames, int sample_rate) {
    const int capped_max = std::max(1, max_frames);
    const double seconds = generated_duration_seconds_for_frames(frames, sample_rate);
    const double pct = 100.0 * static_cast<double>(frames) / capped_max;
    std::fprintf(stderr,
                 "[tts_15b] progress: %.2fs generated, frame %d/%d (%.1f%%)\n",
                 seconds, frames, capped_max, pct);
}

// Returns true if `text` contains any explicit "Speaker N:" marker.
bool text_has_speaker_prefix(const std::string& text) {
    static const std::regex re(R"(\bSpeaker\s+\d+\s*:)",
                                std::regex::ECMAScript);
    return std::regex_search(text, re);
}

}  // namespace

namespace detail {

std::vector<int32_t> kugelaudio_valid_speech_token_ids() {
    return {kSpeech15bStartId, kSpeech15bEndId, kSpeech15bDiffId, kSpeech15bEosId};
}

void apply_kugelaudio_speech_end_penalty(std::vector<float>* logits) {
    if (!logits) return;
    if (static_cast<size_t>(kSpeech15bEndId) < logits->size()) {
        (*logits)[static_cast<size_t>(kSpeech15bEndId)] -= kKugelAudioSpeechEndPenalty;
    }
}

int select_kugelaudio_speech_token_from_logits(const std::vector<float>& logits) {
    const auto valid_ids = kugelaudio_valid_speech_token_ids();
    int   best_id = -1;
    float best_v  = -std::numeric_limits<float>::infinity();
    for (const int32_t id : valid_ids) {
        if (id < 0 || static_cast<size_t>(id) >= logits.size()) continue;
        if (best_id < 0 || logits[static_cast<size_t>(id)] > best_v) {
            best_v = logits[static_cast<size_t>(id)];
            best_id = id;
        }
    }
    return best_id;
}

bool kugelaudio_token_requires_cfg_reset(int32_t token_id) {
    return token_id == kSpeech15bStartId;
}

bool kugelaudio_token_stops_generation(int32_t token_id) {
    return token_id == kSpeech15bEndId || token_id == kSpeech15bEosId;
}

std::vector<int32_t> kugelaudio_valid_speech_token_ids_for_test() {
    return kugelaudio_valid_speech_token_ids();
}

float kugelaudio_speech_end_penalty_for_test() {
    return kKugelAudioSpeechEndPenalty;
}

void apply_kugelaudio_speech_end_penalty_for_test(std::vector<float>* logits) {
    apply_kugelaudio_speech_end_penalty(logits);
}

std::vector<float> run_speech_connector_for_test(const VibeVoiceConfig& cfg,
                                                 const VibeVoiceWeights& w,
                                                 const float* x,
                                                 int batch) {
    return run_speech_connector(cfg, w, x, batch);
}

DPMSolverConfig kugelaudio_solver_config_for_test(int requested_steps) {
    DPMSolverConfig solver_cfg;
    solver_cfg.num_train_timesteps = 1000;
    solver_cfg.num_inference_steps = requested_steps > 0 ? requested_steps : 20;
    solver_cfg.solver_order        = 2;
    solver_cfg.lower_order_final   = true;
    solver_cfg.sde_dpmsolver_plus_plus = true;
    solver_cfg.cast_sample_to_f16  = true;
    return solver_cfg;
}

int select_kugelaudio_speech_token_from_logits_for_test(const std::vector<float>& logits) {
    return select_kugelaudio_speech_token_from_logits(logits);
}

bool kugelaudio_token_requires_cfg_reset_for_test(int32_t token_id) {
    return kugelaudio_token_requires_cfg_reset(token_id);
}

bool kugelaudio_token_stops_generation_for_test(int32_t token_id) {
    return kugelaudio_token_stops_generation(token_id);
}

const KugelAudioRequestPolicy& kugelaudio_v1_request_policy() {
    static const KugelAudioRequestPolicy kPolicy = {
        /*allow_pre_baked_voice=*/false,
        /*min_ref_audio_inputs=*/1,
        /*max_ref_audio_inputs=*/1,
        /*allow_speaker_tagged_dialog=*/false,
        /*supported_shape=*/"KugelAudio v1 supports only single-speaker TTS with exactly one raw reference audio input and plain untagged text",
    };
    return kPolicy;
}

bool validate_kugelaudio_request(const std::string& text,
                                 const VibeVoiceTTSParams& p,
                                 const KugelAudioRequestPolicy& policy,
                                 std::string* error) {
    const char* supported_shape = policy.supported_shape ? policy.supported_shape : "unsupported KugelAudio request shape";
    if (p.voice && !policy.allow_pre_baked_voice) {
        if (error) *error = std::string("unsupported KugelAudio runtime feature: pre-baked voice gguf conditioning; ") + supported_shape;
        return false;
    }
    if (p.ref_audio_paths.size() < policy.min_ref_audio_inputs ||
        p.ref_audio_paths.size() > policy.max_ref_audio_inputs) {
        if (error) *error = std::string("unsupported KugelAudio runtime feature: got ")
            + std::to_string(p.ref_audio_paths.size())
            + " raw reference audio input(s); " + supported_shape;
        return false;
    }
    if (text_has_speaker_prefix(text) && !policy.allow_speaker_tagged_dialog) {
        if (error) *error = std::string("unsupported KugelAudio runtime feature: Speaker-tagged dialog input; ") + supported_shape;
        return false;
    }
    return true;
}

bool validate_kugelaudio_single_speaker_request(const std::string& text,
                                                const VibeVoiceTTSParams& p,
                                                std::string* error) {
    return validate_kugelaudio_request(text, p, kugelaudio_v1_request_policy(), error);
}

}  // namespace detail

namespace {

std::string format_kugelaudio_single_speaker_text(const std::string& text) {
    std::string formatted_text = text;
    while (!formatted_text.empty() &&
           (formatted_text.back() == ' ' || formatted_text.back() == '\t' ||
            formatted_text.back() == '\r' || formatted_text.back() == '\n')) {
        formatted_text.pop_back();
    }
    if (formatted_text.rfind("Speaker", 0) != 0) {
        formatted_text = "Speaker 0: " + formatted_text;
    }
    return formatted_text;
}

struct KugelAudioPromptSections {
    std::string system_prompt;
    std::string voice_input_header;
    std::string voice_speaker_prefix;
    std::string text_input_header;
    std::string speaker_text;
    std::string speech_output_header;
};

KugelAudioPromptSections build_kugelaudio_prompt_sections(int /*vae_tok_len*/,
                                                          const std::string& text,
                                                          const std::string& continuity_instruction = {}) {
    KugelAudioPromptSections s;
    s.system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";
    if (!continuity_instruction.empty()) {
        s.system_prompt += " " + continuity_instruction + "\n";
    }
    s.voice_input_header = " Voice input:\n";
    s.voice_speaker_prefix = " Speaker 0:";
    s.text_input_header = " Text input:\n";
    s.speaker_text = " " + format_kugelaudio_single_speaker_text(text) + "\n";
    s.speech_output_header = " Speech output:\n";
    return s;
}

void build_kugelaudio_prompt_input_ids(const Tokenizer& tokenizer,
                                       int vae_tok_len,
                                       const std::string& text,
                                       std::vector<int32_t>* input_ids,
                                       std::vector<int>* pad_positions,
                                       const std::string& continuity_instruction = {}) {
    const auto s = build_kugelaudio_prompt_sections(vae_tok_len, text, continuity_instruction);
    input_ids->clear();
    pad_positions->clear();

    auto append = [&](const std::string& chunk) {
        auto ids = tokenizer.encode(chunk);
        input_ids->insert(input_ids->end(), ids.begin(), ids.end());
    };

    append(s.system_prompt);
    append(s.voice_input_header);
    append(s.voice_speaker_prefix);
    for (int j = 0; j < vae_tok_len; ++j) {
        pad_positions->push_back(static_cast<int>(input_ids->size()));
        input_ids->push_back(kSpeech15bDiffId);
    }
    append("\n");
    append(s.text_input_header);
    append(s.speaker_text);
    append(s.speech_output_header);
    input_ids->push_back(kSpeech15bStartId);
}

// Legacy VibeVoice 1.5B prompt builder. Kept for non-KugelAudio paths.
std::string build_prompt_15b_legacy(const std::vector<int>& vae_tok_lens,
                                    const std::string& text) {
    int total_pads = 0;
    for (int t : vae_tok_lens) total_pads += t;

    std::string out;
    out.reserve(2048 + total_pads * 16 + text.size());
    out += " Transform the text provided by various speakers into speech "
           "output, utilizing the distinct voice of each respective speaker.\n";
    out += " Voice input:\n";
    for (size_t i = 0; i < vae_tok_lens.size(); ++i) {
        out += " Speaker " + std::to_string(i) + ":";
        out += "<|vision_start|>";
        for (int j = 0; j < vae_tok_lens[i]; ++j) out += "<|vision_pad|>";
        out += "<|vision_end|>\n";
    }
    out += " Text input:\n";
    if (text_has_speaker_prefix(text)) {
        // User already supplied speaker-tagged dialog; pass through.
        // Each line should begin with " Speaker N:" — we add the
        // leading space if the user forgot it on the first line.
        if (!text.empty() && text[0] != ' ') out += ' ';
        out += text;
        if (out.empty() || out.back() != '\n') out += "\n";
    } else {
        out += " Speaker 0:";
        out += text;
        out += "\n";
    }
    out += " Speech output:\n";
    out += "<|vision_start|>";
    return out;
}

// Canonical KugelAudio v1 single-speaker prompt builder, matching
// kugelaudio_open.processors.kugelaudio_processor.KugelAudioProcessor.
std::string build_kugelaudio_prompt_single_speaker(int vae_tok_len,
                                                   const std::string& text) {
    const auto s = build_kugelaudio_prompt_sections(vae_tok_len, text);
    std::string out;
    out.reserve(1024 + static_cast<size_t>(vae_tok_len) * 14 + s.speaker_text.size());
    out += s.system_prompt;
    out += s.voice_input_header;
    out += s.voice_speaker_prefix;
    for (int j = 0; j < vae_tok_len; ++j) out += "<|vision_pad|>";
    out += "\n";
    out += s.text_input_header;
    out += s.speaker_text;
    out += s.speech_output_header;
    out += "<|vision_start|>";
    return out;
}

// Fetch a single row from `tok_embd` (handles fp32 / fp16). Caller passes
// a destination buffer of `hidden` floats.
void embed_row(struct ggml_tensor* tok_embd, int id, int hidden, float* dst) {
    if (tok_embd->type == GGML_TYPE_F32) {
        const size_t row = sizeof(float) * hidden;
        ggml_backend_tensor_get(tok_embd, dst,
                                row * static_cast<size_t>(id), row);
    } else {
        // fp16
        const size_t row = sizeof(ggml_fp16_t) * hidden;
        std::vector<ggml_fp16_t> staged(hidden);
        ggml_backend_tensor_get(tok_embd, staged.data(),
                                row * static_cast<size_t>(id), row);
        for (int i = 0; i < hidden; ++i) {
            dst[i] = ggml_fp16_to_fp32(staged[i]);
        }
    }
}

}  // namespace

namespace detail {
std::string build_kugelaudio_prompt_single_speaker_for_test(int vae_tok_len,
                                                            const std::string& text) {
    return build_kugelaudio_prompt_single_speaker(vae_tok_len, text);
}

std::vector<int32_t> build_kugelaudio_inserted_speech_tokens_for_test(int vae_tok_len) {
    std::vector<int32_t> ids;
    ids.reserve(static_cast<size_t>(vae_tok_len) + 1);
    for (int i = 0; i < vae_tok_len; ++i) ids.push_back(kSpeech15bDiffId);
    ids.push_back(kSpeech15bStartId);
    return ids;
}

std::vector<int32_t> build_kugelaudio_negative_seed_tokens_for_test() {
    return {kSpeech15bStartId};
}

void build_kugelaudio_prompt_input_ids_for_test(const Tokenizer& tokenizer,
                                                int vae_tok_len,
                                                const std::string& text,
                                                std::vector<int32_t>* input_ids,
                                                std::vector<int>* pad_positions) {
    build_kugelaudio_prompt_input_ids(tokenizer, vae_tok_len, text, input_ids, pad_positions);
}

int kugelaudio_speech_start_id_for_test() { return kSpeech15bStartId; }
int kugelaudio_speech_end_id_for_test() { return kSpeech15bEndId; }
int kugelaudio_speech_diffusion_id_for_test() { return kSpeech15bDiffId; }
int kugelaudio_eos_id_for_test() { return kSpeech15bEosId; }
int kugelaudio_image_pad_id_for_test() { return kSpeech15bImgPadId; }
int kugelaudio_pause_duration_ms_for_test(const std::string& left_text,
                                          const std::string& right_text,
                                          ChunkPauseMode pause_mode) {
    return detail::kugelaudio_pause_duration_ms_for_boundary(left_text, right_text, pause_mode);
}
}  // namespace detail

namespace {
bool should_short_circuit_after_conditioning_for_test() {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_TEST_STOP_AFTER_CONDITIONING");
    return v && v[0] && std::strcmp(v, "0") != 0;
}

bool should_short_circuit_after_connectors_for_test() {
    const char* v = kugelaudio_getenv("VIBEVOICE_KUGELAUDIO_TEST_STOP_AFTER_CONNECTORS");
    return v && v[0] && std::strcmp(v, "0") != 0;
}

int tts_15b_generate_single_pass(VibeVoiceModel*            model,
                                 const std::string&         text,
                                 const VibeVoiceTTSParams&  p,
                                 std::vector<float>*        samples,
                                 const PreparedSpeechConditioning* prepared_conditioning,
                                 const ChunkLatentContinuityState* latent_prefix,
                                 ChunkLatentContinuityState* latent_tail_out) {
    if (!model || !samples) return -1;
    if (model->variant != "1.5b") {
        VV_LOG_ERROR("tts_15b: model is variant=%s, expected 1.5b",
                     model->variant.c_str());
        return -1;
    }

    const bool is_kugelaudio = model->loader.has_key("kugelaudio.architecture");
    if (is_kugelaudio) {
        std::string gate_error;
        if (!detail::validate_kugelaudio_single_speaker_request(text, p, &gate_error)) {
            VV_LOG_ERROR("tts_15b: %s", gate_error.c_str());
            if (p.voice) return -20;
            if (p.ref_audio_paths.size() != 1) return -21;
            return -22;
        }
    }

    if (!model->tokenizer.vocab_size()) {
        VV_LOG_ERROR("tts_15b: tokenizer not loaded");
        return -2;
    }
    if (p.ref_audio_paths.empty()) {
        VV_LOG_ERROR("tts_15b: ref_audio_paths is empty");
        return -3;
    }

    const auto& cfg = model->cfg;
    const auto& w   = model->w;
    const int hidden = cfg.hidden;

    samples->clear();

    // ---- 1. prepare raw-reference speech conditioning ----
    PreparedSpeechConditioning local_conditioning;
    const PreparedSpeechConditioning* conditioning = prepared_conditioning;
    std::mt19937 rng(p.seed ? p.seed : std::random_device{}());
    if (!conditioning) {
        PreparedSpeechConditioningResult prep = prepare_speech_conditioning(model, p, is_kugelaudio, rng, &local_conditioning);
        if (prep.rc != 0) return prep.rc;
        if (prep.stopped_after_preprocessing || prep.stopped_after_connectors) {
            samples->assign(240, 0.0f);
            return 0;
        }
        conditioning = &local_conditioning;
    } else if (is_kugelaudio) {
        // Cached conditioning must not shift the diffusion RNG stream relative
        // to the uncached single-pass path. Consume the same acoustic sampling
        // draws that produced the cached features, then continue with diffusion.
        for (size_t count : conditioning->acoustic_feature_counts) {
            std::vector<float> dummy(count, 0.0f);
            (void) kugelaudio_sample_acoustic_features(dummy,
                                                       cfg.acoustic_fix_std,
                                                       cfg.acoustic_std_dist_type,
                                                       rng);
        }
    }

    const std::vector<int>& per_speaker_Tc = conditioning->per_speaker_Tc;
    const std::vector<float>& speech_features = conditioning->speech_features;
    const int total_Tc = conditioning->total_Tc;
    if (p.verbose) {
        for (size_t spk = 0; spk < per_speaker_Tc.size(); ++spk) {
            const size_t ref_samples = spk < conditioning->ref_sample_counts.size()
                ? conditioning->ref_sample_counts[spk]
                : 0;
            std::fprintf(stderr,
                "[tts_15b] speaker %zu: ref %zu samples -> %d compressed frames\n",
                spk, ref_samples, per_speaker_Tc[spk]);
        }
    }

    // ---- 2. build prompt + tokenize ----
    std::vector<int32_t> input_ids;
    std::vector<int> pad_positions;
    if (is_kugelaudio) {
        build_kugelaudio_prompt_input_ids(model->tokenizer, per_speaker_Tc.front(), text,
                                          &input_ids, &pad_positions,
                                          p.prompt_continuity_instruction);
    } else {
        const std::string prompt = build_prompt_15b_legacy(per_speaker_Tc, text);
        input_ids = model->tokenizer.encode(prompt);
        pad_positions.reserve(total_Tc);
        for (int i = 0; i < static_cast<int>(input_ids.size()); ++i) {
            if (input_ids[i] == kSpeech15bDiffId) pad_positions.push_back(i);
        }
    }
    if (input_ids.empty()) {
        VV_LOG_ERROR("tts_15b: tokenizer returned no tokens");
        return -7;
    }
    if (is_kugelaudio) {
        kugelaudio_dump_i32_vector("08_prompt_input_ids", input_ids,
                                   "tokenized KugelAudio prompt input ids");
        kugelaudio_dump_int_vector("09_pad_positions", pad_positions,
                                   "positions replaced by speech conditioning embeddings");
    }
    const int N = static_cast<int>(input_ids.size());

    if (static_cast<int>(pad_positions.size()) != total_Tc) {
        VV_LOG_ERROR("tts_15b: prompt has %zu vision_pad tokens, want %d "
                     "(sum of per-speaker frames; tokenizer mis-decoding?)",
                     pad_positions.size(), total_Tc);
        return -8;
    }
    if (p.verbose) std::fprintf(stderr,
        "[tts_15b] prompt %d tokens, %d are speech-pad across %zu speaker(s)\n",
        N, total_Tc, p.ref_audio_paths.size());

    // ---- 3. embed prompt + splice speech features (in speaker order) ----
    std::vector<float> embeds(static_cast<size_t>(hidden) * N);
    for (int t = 0; t < N; ++t) {
        const int id = input_ids[t];
        if (id < 0 || id >= cfg.vocab_size) {
            VV_LOG_ERROR("tts_15b: token id out of range: %d", id);
            return -9;
        }
        embed_row(w.lm_tok_embd, id, hidden, &embeds[hidden * t]);
    }
    if (is_kugelaudio) {
        kugelaudio_dump_f32_matrix("10_prompt_embeds_pre_splice", embeds,
                                   static_cast<size_t>(N), static_cast<size_t>(hidden),
                                   "prompt token embeddings before speech-feature splice");
    }
    // pad_positions is in document order, which is also speaker order
    // (build_prompt_15b emits all of speaker 0's vision_pads first,
    // then speaker 1's, ...). So the k-th pad gets the k-th feature
    // row from the concatenated speech_features buffer.
    for (int k = 0; k < total_Tc; ++k) {
        const int pos = pad_positions[k];
        std::memcpy(&embeds[static_cast<size_t>(hidden) * pos],
                    &speech_features[static_cast<size_t>(hidden) * k],
                    sizeof(float) * hidden);
    }
    if (is_kugelaudio) {
        kugelaudio_dump_f32_matrix("11_prompt_embeds_post_splice", embeds,
                                   static_cast<size_t>(N), static_cast<size_t>(hidden),
                                   "prompt embeddings after inserting speech conditioning features");
    }

    // ---- 6. resident KV cache + LM prefill ----
    const int max_seq = N + p.max_speech_frames + 32;
    ResidentKV kv_lm;
    if (!kv_lm.init(cfg.n_layers_lm, cfg.head_dim, cfg.n_kv_heads, max_seq)) {
        VV_LOG_ERROR("tts_15b: resident KV init failed");
        return -10;
    }

    std::vector<float> hidden_last;
    if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                         /*past_len=*/0, /*n_new=*/N, embeds.data(),
                         &kv_lm, /*all_hidden_out=*/nullptr, &hidden_last)) {
        VV_LOG_ERROR("tts_15b: prefill LM forward failed");
        return -11;
    }
    if (is_kugelaudio) {
        kugelaudio_dump_f32_vector("12_prefill_hidden_last_pos", hidden_last,
                                   "last hidden state after positive prefill");
    }
    int lm_pos = N;

    // ---- 6b. negative branch (CFG) ----
    // Build a parallel LM forward over the SAME prompt structure as
    // positive, but with each <|vision_pad|> position embedded as
    // <|image_pad|> instead of as the reference-audio feature. That
    // keeps the negative LM's hidden-state distribution close to
    // positive's (same prompt length, same KV depth at each step) so
    // the diffusion head sees an in-distribution `cond_neg` — just
    // one without the specific voice/text conditioning.
    //
    // (A simpler "single image_pad token" negative — the literal
    // analog of the streaming model's pre-baked neg_tts_lm state —
    // collapses for the 1.5B model: its diffusion head was trained
    // against full-length conditioning hidden states, not the 1-token
    // state the streaming TLM uses. Mixing v predictions from such a
    // low-rank baseline produces noise even at cfg_scale=1.05.)
    //
    // Empirical sweet spot for cfg_scale on the closed-loop word-recall
    // benchmark (1.5B Q8_0, samples/2p_argument 5s ref):
    //   cfg=1.0  -> 100%   (CFG off)
    //   cfg=1.3  -> 88.9%
    //   cfg=2.0  -> 100%
    //   cfg=3.0  -> 100%
    //   cfg>=5   -> noise / [Music] artifact
    const bool use_cfg = (p.cfg_scale > 1.0f);
    const int rolling_kv_frames = is_kugelaudio ? kugelaudio_rolling_kv_frames() : -1;
    const bool use_rolling_kv = rolling_kv_frames >= 0;
    if (use_rolling_kv && use_cfg) {
        VV_LOG_ERROR("tts_15b: VIBEVOICE_KUGELAUDIO_ROLLING_KV_FRAMES currently requires cfg=1.0");
        return -17;
    }
    if (use_rolling_kv) {
        VV_LOG_INFO("tts_15b: using diagnostic rolling KV history frames=%d", rolling_kv_frames);
    }
    std::vector<float> rolling_step_history;
    std::vector<float> previous_fed_step_embed;
    std::vector<float> previous_fed_diffusion_cond;
    const bool cast_step_embed_input_f16 = is_kugelaudio && kugelaudio_cast_step_embed_f16(model, p);
    if (cast_step_embed_input_f16) {
        VV_LOG_INFO("tts_15b: rounding generated step embeddings to f16 before LM feedback");
    }
    float temporal_step_alpha = 0.0f;
    const bool use_temporal_step_stabilizer = is_kugelaudio && kugelaudio_temporal_step_embed_alpha(&temporal_step_alpha);
    if (use_temporal_step_stabilizer) {
        VV_LOG_INFO("tts_15b: using diagnostic temporal step-embedding stabilizer alpha=%.3f", static_cast<double>(temporal_step_alpha));
    }
    float temporal_cond_alpha = 0.0f;
    const bool use_temporal_cond_stabilizer = is_kugelaudio && kugelaudio_temporal_diffusion_cond_alpha(&temporal_cond_alpha);
    if (use_temporal_cond_stabilizer) {
        VV_LOG_INFO("tts_15b: using diagnostic temporal diffusion-cond stabilizer alpha=%.3f", static_cast<double>(temporal_cond_alpha));
    }
    const float latent_refine_strength = std::clamp(p.latent_refine_strength, 0.0f, 1.0f);
    const bool use_latent_refine = is_kugelaudio && latent_refine_strength > 0.0f;
    if (use_latent_refine) {
        VV_LOG_INFO("tts_15b: using diagnostic latent refinement strength=%.3f steps=%d",
                    static_cast<double>(latent_refine_strength), p.latent_refine_steps);
    }
    ResidentKV         kv_neg;
    std::vector<float> neg_hidden_last;
    int                neg_pos = 0;
    if (use_cfg) {
        if (!kv_neg.init(cfg.n_layers_lm, cfg.head_dim, cfg.n_kv_heads, max_seq)) {
            VV_LOG_ERROR("tts_15b: neg KV init failed");
            return -10;
        }
        std::vector<float> neg_embeds;
        int neg_prefill_tokens = 0;
        if (is_kugelaudio) {
            neg_embeds.resize(static_cast<size_t>(hidden));
            embed_row(w.lm_tok_embd, kSpeech15bStartId, hidden, neg_embeds.data());
            neg_prefill_tokens = 1;
        } else {
            // Legacy VibeVoice path: copy positive embeds, then overwrite every
            // vision_pad position (across all speakers) with the image_pad embedding.
            neg_embeds = embeds;
            std::vector<float> img_pad_embed(hidden);
            embed_row(w.lm_tok_embd, kSpeech15bImgPadId, hidden, img_pad_embed.data());
            for (int k = 0; k < total_Tc; ++k) {
                const int pos = pad_positions[k];
                std::memcpy(&neg_embeds[static_cast<size_t>(hidden) * pos],
                            img_pad_embed.data(),
                            sizeof(float) * hidden);
            }
            neg_prefill_tokens = N;
        }
        if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                             /*past_len=*/0, /*n_new=*/neg_prefill_tokens, neg_embeds.data(),
                             &kv_neg, /*all_hidden_out=*/nullptr,
                             &neg_hidden_last)) {
            VV_LOG_ERROR("tts_15b: neg prefill failed");
            return -11;
        }
        neg_pos = neg_prefill_tokens;
        if (is_kugelaudio) {
            kugelaudio_dump_f32_vector("13_prefill_hidden_last_neg", neg_hidden_last,
                                       "last hidden state after negative/CFG prefill");
        }
        if (p.verbose) std::fprintf(stderr,
            "[tts_15b] CFG on, scale=%.2f (neg branch prefilled %d tokens)\n",
            static_cast<double>(p.cfg_scale), neg_prefill_tokens);
    } else if (p.verbose) {
        std::fprintf(stderr, "[tts_15b] CFG off (cfg_scale=%.2f)\n",
                     static_cast<double>(p.cfg_scale));
    }

    auto reset_kugelaudio_neg_branch = [&]() -> bool {
        kv_neg.past_len = 0;
        std::vector<float> neg_seed_embed(static_cast<size_t>(hidden));
        embed_row(w.lm_tok_embd, kSpeech15bStartId, hidden, neg_seed_embed.data());
        if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                             /*past_len=*/0, /*n_new=*/1, neg_seed_embed.data(),
                             &kv_neg, /*all_hidden_out=*/nullptr,
                             &neg_hidden_last)) {
            VV_LOG_ERROR("tts_15b: neg branch reset failed");
            return false;
        }
        neg_pos = 1;
        return true;
    };

    // ---- 7. speech-generation loop ----
    // KugelAudio parity note: reuse the existing ggml DPM-Solver++ path with
    // the same canonical scheduler shape (1000 train steps, order-2,
    // lower_order_final) as kugelaudio_open.models.kugelaudio_inference.
    DPMSolverConfig solver_cfg = detail::kugelaudio_solver_config_for_test(p.n_diffusion_steps);
    DPMSolverState solver_state;
    dpm_solver_init(solver_cfg, &solver_state);

    DiffusionHeadConfig dh_cfg;
    dh_cfg.hidden      = hidden;
    dh_cfg.latent      = cfg.latent;
    dh_cfg.head_layers = cfg.head_layers;
    dh_cfg.ffn_ratio   = cfg.ffn_ratio;
    dh_cfg.eps         = cfg.rms_norm_eps;
    dh_cfg.freq_size   = 256;

    if (latent_prefix && latent_prefix->frames > 0 &&
        latent_prefix->latents.size() >= static_cast<size_t>(latent_prefix->frames) * static_cast<size_t>(cfg.latent)) {
        for (int t = 0; t < latent_prefix->frames; ++t) {
            const float* z_prev = latent_prefix->latents.data() + static_cast<size_t>(t) * static_cast<size_t>(cfg.latent);
            auto step_embed = run_speech_connector(cfg, w, z_prev, /*batch=*/1);
            if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                                 lm_pos, /*n_new=*/1, step_embed.data(),
                                 &kv_lm, nullptr, &hidden_last)) {
                VV_LOG_ERROR("tts_15b: latent-prefix continuity LM step failed at prefix frame %d", t);
                return -13;
            }
            ++lm_pos;
            if (use_cfg) {
                if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                                     neg_pos, /*n_new=*/1, step_embed.data(),
                                     &kv_neg, nullptr, &neg_hidden_last)) {
                    VV_LOG_ERROR("tts_15b: latent-prefix continuity neg LM step failed at prefix frame %d", t);
                    return -13;
                }
                ++neg_pos;
            }
        }
        VV_LOG_INFO("tts_15b: applied latent-prefix continuity frames=%d", latent_prefix->frames);
    }

    std::normal_distribution<float> norm(0.0f, 1.0f);

    std::vector<float> all_latents;
    all_latents.reserve(static_cast<size_t>(p.max_speech_frames) * cfg.latent);
    std::vector<float> latent_refine_cond_history;
    std::vector<float> latent_refine_cond_neg_history;
    if (use_latent_refine) {
        latent_refine_cond_history.reserve(static_cast<size_t>(p.max_speech_frames) * static_cast<size_t>(hidden));
        if (use_cfg) latent_refine_cond_neg_history.reserve(static_cast<size_t>(p.max_speech_frames) * static_cast<size_t>(hidden));
    }

    int  total_frames = 0;
    int  control_steps_without_audio = 0;
    bool finished     = false;
    bool dumped_first_logits = false;
    bool dumped_first_diffusion = false;
    int suppressed_stop_tokens = 0;
    int eos_guard_suppressed_stop_tokens = 0;

    while (!finished && total_frames < p.max_speech_frames) {
        if (is_kugelaudio) {
            auto logits = detail::lm_head_logits_last(model->lm_head, hidden_last,
                                                      hidden, cfg.vocab_size);
            if (static_cast<int>(logits.size()) != cfg.vocab_size) {
                VV_LOG_ERROR("tts_15b: failed to read KugelAudio speech-path logits");
                return -14;
            }
            detail::apply_kugelaudio_speech_end_penalty(&logits);
            int next_token = detail::select_kugelaudio_speech_token_from_logits(logits);
            if (p.min_speech_frames > 0 && total_frames < p.min_speech_frames &&
                detail::kugelaudio_token_stops_generation(next_token)) {
                if (static_cast<size_t>(kSpeech15bEndId) < logits.size()) {
                    logits[static_cast<size_t>(kSpeech15bEndId)] = -std::numeric_limits<float>::infinity();
                }
                if (static_cast<size_t>(kSpeech15bEosId) < logits.size()) {
                    logits[static_cast<size_t>(kSpeech15bEosId)] = -std::numeric_limits<float>::infinity();
                }
                next_token = detail::select_kugelaudio_speech_token_from_logits(logits);
                ++suppressed_stop_tokens;
                if (p.verbose && (suppressed_stop_tokens <= 5 || suppressed_stop_tokens % 50 == 0)) {
                    std::fprintf(stderr,
                        "[tts_15b] suppressed speech_end/eos at frame %d (min_frames=%d, next=%d)\n",
                        total_frames, p.min_speech_frames, next_token);
                }
            }
            if (is_kugelaudio && kugelaudio_should_dump_frame(total_frames)) {
                kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "logits"), logits,
                                           "per-frame constrained/penalized logits before speech-token selection");
                kugelaudio_dump_i32_scalar(kugelaudio_frame_stage(total_frames, "selected_token"), next_token,
                                           "per-frame selected speech-path token");
            }
            if (is_kugelaudio && !dumped_first_logits) {
                kugelaudio_dump_f32_vector("14_first_logits", logits,
                                           "first constrained/penalized logits before speech-token selection");
                kugelaudio_dump_i32_scalar("15_first_selected_token", next_token,
                                           "selected first speech-path control token");
                dumped_first_logits = true;
            }
            if (detail::kugelaudio_token_stops_generation(next_token) &&
                p.chunk_eos_guard_frames > 0 &&
                total_frames > 0 &&
                eos_guard_suppressed_stop_tokens < p.chunk_eos_guard_frames) {
                if (static_cast<size_t>(kSpeech15bEndId) < logits.size()) {
                    logits[static_cast<size_t>(kSpeech15bEndId)] = -std::numeric_limits<float>::infinity();
                }
                if (static_cast<size_t>(kSpeech15bEosId) < logits.size()) {
                    logits[static_cast<size_t>(kSpeech15bEosId)] = -std::numeric_limits<float>::infinity();
                }
                next_token = detail::select_kugelaudio_speech_token_from_logits(logits);
                ++eos_guard_suppressed_stop_tokens;
                if (p.verbose) {
                    std::fprintf(stderr,
                        "[tts_15b] eos guard suppressed speech_end/eos at frame %d (guard=%d/%d, next=%d)\n",
                        total_frames, eos_guard_suppressed_stop_tokens, p.chunk_eos_guard_frames, next_token);
                }
            }
            if (detail::kugelaudio_token_stops_generation(next_token)) {
                if (p.verbose) std::fprintf(stderr,
                    "[tts_15b] speech_end/eos before diffusion at frame %d\n", total_frames);
                finished = true;
                break;
            }
            if (detail::kugelaudio_token_requires_cfg_reset(next_token)) {
                std::vector<float> start_embed(static_cast<size_t>(hidden));
                embed_row(w.lm_tok_embd, kSpeech15bStartId, hidden, start_embed.data());
                if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                                     lm_pos, /*n_new=*/1, start_embed.data(),
                                     &kv_lm, nullptr, &hidden_last)) {
                    VV_LOG_ERROR("tts_15b: LM speech_start step failed");
                    return -13;
                }
                ++lm_pos;
                if (use_cfg && !reset_kugelaudio_neg_branch()) {
                    return -11;
                }
                if (++control_steps_without_audio > 32) {
                    VV_LOG_ERROR("tts_15b: exceeded KugelAudio control-token guard without generating speech");
                    return -15;
                }
                continue;
            }
            control_steps_without_audio = 0;
        }

        // 7a. sample one speech latent via DPM-Solver, optionally with CFG.
        std::vector<float> z(cfg.latent);
        for (auto& v : z) v = norm(rng);
        if (is_kugelaudio && kugelaudio_noise_load_enabled()) {
            std::vector<float> loaded_noise;
            const std::filesystem::path noise_dir(kugelaudio_noise_dir_env());
            const std::string noise_stage = kugelaudio_frame_stage(total_frames, "diffusion_noise");
            if (kugelaudio_load_f32_vector_from_dump(noise_dir, noise_stage, static_cast<size_t>(cfg.latent), &loaded_noise)) {
                z.swap(loaded_noise);
            } else {
                VV_LOG_ERROR("tts_15b: requested canonical noise but missing/incompatible %s in %s",
                             noise_stage.c_str(), noise_dir.string().c_str());
                return -16;
            }
        }
        if (is_kugelaudio && kugelaudio_should_dump_frame(total_frames)) {
            kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "diffusion_noise"), z,
                                       "per-frame initial Gaussian diffusion noise before DPM sampling");
        }

        std::vector<float> cond(static_cast<size_t>(hidden));
        std::memcpy(cond.data(), hidden_last.data(), sizeof(float) * hidden);
        float cond_blend_alpha = 0.0f;
        const bool use_cond_blend = is_kugelaudio && kugelaudio_diffusion_cond_blend_alpha(&cond_blend_alpha);
        if (is_kugelaudio && (kugelaudio_teacher_force_diffusion_cond() || use_cond_blend)) {
            if (kugelaudio_should_dump_frame(total_frames)) {
                kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "diffusion_cond_pos_model"), cond,
                                           "per-frame C++ positive diffusion conditioning before teacher/blend forcing");
            }
            std::vector<float> canonical_cond;
            const std::filesystem::path noise_dir(kugelaudio_noise_dir_env() ? kugelaudio_noise_dir_env() : "");
            const std::string stage = kugelaudio_frame_stage(total_frames, "diffusion_cond_pos");
            if (!kugelaudio_noise_load_enabled() ||
                !kugelaudio_load_f32_vector_from_dump(noise_dir, stage, static_cast<size_t>(hidden), &canonical_cond)) {
                VV_LOG_ERROR("tts_15b: requested canonical diffusion cond but missing/incompatible %s in %s",
                             stage.c_str(), noise_dir.string().c_str());
                return -16;
            }
            if (use_cond_blend) {
                for (size_t i = 0; i < cond.size(); ++i) {
                    cond[i] = canonical_cond[i] * (1.0f - cond_blend_alpha) + cond[i] * cond_blend_alpha;
                }
            } else {
                cond.swap(canonical_cond);
            }
        }

        std::vector<float> cond_neg;
        if (use_cfg) cond_neg.assign(neg_hidden_last.begin(), neg_hidden_last.end());
        if (use_temporal_cond_stabilizer && !previous_fed_diffusion_cond.empty() &&
            previous_fed_diffusion_cond.size() == cond.size()) {
            if (is_kugelaudio && kugelaudio_should_dump_frame(total_frames)) {
                kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "diffusion_cond_pos_pre_temporal"), cond,
                                           "per-frame positive diffusion conditioning before temporal stabilizer");
            }
            for (size_t i = 0; i < cond.size(); ++i) {
                cond[i] = temporal_cond_alpha * cond[i] +
                          (1.0f - temporal_cond_alpha) * previous_fed_diffusion_cond[i];
            }
        }
        previous_fed_diffusion_cond = cond;
        if (is_kugelaudio && kugelaudio_cast_diffusion_cond_f16()) {
            if (kugelaudio_should_dump_frame(total_frames)) {
                kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "diffusion_cond_pos_pre_cast_f16"), cond,
                                           "per-frame positive diffusion conditioning before f16 cast");
            }
            for (float& v : cond) {
                v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
            }
        }
        if (is_kugelaudio && kugelaudio_should_dump_frame(total_frames)) {
            kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "diffusion_cond_pos"), cond,
                                       "per-frame positive conditioning vector for diffusion sample");
            if (use_cfg) {
                kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "diffusion_cond_neg"), cond_neg,
                                           "per-frame negative conditioning vector for diffusion sample");
            }
        }
        if (is_kugelaudio && !dumped_first_diffusion) {
            kugelaudio_dump_f32_vector("16_first_diffusion_cond_pos", cond,
                                       "positive conditioning vector for first diffusion sample");
            if (use_cfg) {
                kugelaudio_dump_f32_vector("17_first_diffusion_cond_neg", cond_neg,
                                           "negative conditioning vector for first diffusion sample");
            }
        }
        if (use_latent_refine) {
            latent_refine_cond_history.insert(latent_refine_cond_history.end(), cond.begin(), cond.end());
            if (use_cfg) {
                latent_refine_cond_neg_history.insert(latent_refine_cond_neg_history.end(), cond_neg.begin(), cond_neg.end());
            }
        }

        std::vector<float> dpm_variance_noise;
        if (solver_cfg.sde_dpmsolver_plus_plus) {
            const size_t step_noise_count = static_cast<size_t>(solver_cfg.num_inference_steps) * static_cast<size_t>(cfg.latent);
            dpm_variance_noise.resize(step_noise_count);
            for (int si = 0; si < solver_cfg.num_inference_steps; ++si) {
                std::vector<float> step_noise(static_cast<size_t>(cfg.latent));
                if (is_kugelaudio && kugelaudio_noise_load_enabled()) {
                    const std::filesystem::path noise_dir(kugelaudio_noise_dir_env());
                    char suffix[64];
                    std::snprintf(suffix, sizeof(suffix), "dpm_step_%02d_variance_noise", si);
                    const std::string noise_stage = kugelaudio_frame_stage(total_frames, suffix);
                    if (!kugelaudio_load_f32_vector_from_dump(noise_dir, noise_stage, static_cast<size_t>(cfg.latent), &step_noise)) {
                        VV_LOG_ERROR("tts_15b: requested canonical DPM variance noise but missing/incompatible %s in %s",
                                     noise_stage.c_str(), noise_dir.string().c_str());
                        return -16;
                    }
                } else {
                    for (auto& v : step_noise) v = norm(rng);
                }
                std::memcpy(dpm_variance_noise.data() + static_cast<size_t>(si) * static_cast<size_t>(cfg.latent),
                            step_noise.data(), sizeof(float) * static_cast<size_t>(cfg.latent));
                if (is_kugelaudio && kugelaudio_should_dump_frame(total_frames)) {
                    char suffix[64];
                    std::snprintf(suffix, sizeof(suffix), "dpm_step_%02d_variance_noise", si);
                    kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, suffix), step_noise,
                                               "per-frame per-DPM-step SDE variance noise");
                }
            }
        }

        std::function<void(int, const std::vector<float>&)> dpm_model_trace;
        std::function<void(int, const std::vector<float>&)> dpm_trace;
        if (is_kugelaudio && kugelaudio_should_dump_frame(total_frames)) {
            dpm_model_trace = [&](int step_index, const std::vector<float>& model_output) {
                char suffix[64];
                std::snprintf(suffix, sizeof(suffix), "dpm_step_%02d_model_output", step_index);
                kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, suffix), model_output,
                                           "per-frame raw diffusion-head model output before scheduler step");
            };
            dpm_trace = [&](int step_index, const std::vector<float>& sample) {
                char suffix[64];
                std::snprintf(suffix, sizeof(suffix), "dpm_step_%02d_sample", step_index);
                kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, suffix), sample,
                                           "per-frame DPM sample after scheduler step");
            };
        }
        if (dpm_solver_sample(z, cfg.latent, /*frames=*/1, /*batch=*/1,
                              cond, hidden,
                              w.dh, dh_cfg, solver_cfg, solver_state,
                              cond_neg, p.cfg_scale,
                              solver_cfg.sde_dpmsolver_plus_plus ? &dpm_variance_noise : nullptr,
                              dpm_model_trace ? &dpm_model_trace : nullptr,
                              dpm_trace ? &dpm_trace : nullptr) != 0) {
            VV_LOG_ERROR("tts_15b: dpm_solver_sample failed at frame %d", total_frames);
            return -12;
        }
        all_latents.insert(all_latents.end(), z.begin(), z.end());
        if (is_kugelaudio && kugelaudio_should_dump_frame(total_frames)) {
            kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "diffusion_latent"), z,
                                       "per-frame diffusion-sampled latent before acoustic decode unscale");
        }
        if (is_kugelaudio && !dumped_first_diffusion) {
            kugelaudio_dump_f32_vector("18_first_diffusion_latent", z,
                                       "first diffusion-sampled latent before acoustic decode unscale");
        }

        // 7b. project latent -> next-step LM input embedding.
        auto step_embed = run_speech_connector(cfg, w, z.data(), /*batch=*/1);
        float blend_alpha = 0.0f;
        const bool use_blend_step_embeds = is_kugelaudio && kugelaudio_step_embed_blend_alpha(&blend_alpha);
        if (is_kugelaudio && (kugelaudio_teacher_force_step_embeds() || use_blend_step_embeds)) {
            if (kugelaudio_should_dump_frame(total_frames)) {
                kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "step_embed_model"), step_embed,
                                           "per-frame C++ generated step embedding before teacher/blend forcing");
            }
            std::vector<float> canonical_step_embed;
            const std::filesystem::path noise_dir(kugelaudio_noise_dir_env() ? kugelaudio_noise_dir_env() : "");
            const std::string stage = kugelaudio_frame_stage(total_frames, "step_embed");
            if (!kugelaudio_noise_load_enabled() ||
                !kugelaudio_load_f32_vector_from_dump(noise_dir, stage, static_cast<size_t>(hidden), &canonical_step_embed)) {
                VV_LOG_ERROR("tts_15b: requested canonical step embeddings but missing/incompatible %s in %s",
                             stage.c_str(), noise_dir.string().c_str());
                return -16;
            }
            if (use_blend_step_embeds) {
                for (size_t i = 0; i < step_embed.size(); ++i) {
                    step_embed[i] = canonical_step_embed[i] * (1.0f - blend_alpha) + step_embed[i] * blend_alpha;
                }
            } else {
                step_embed.swap(canonical_step_embed);
            }
        }
        if (cast_step_embed_input_f16) {
            if (kugelaudio_should_dump_frame(total_frames)) {
                kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "step_embed_pre_cast_f16"), step_embed,
                                           "per-frame step embedding before f16 input cast");
            }
            for (float& v : step_embed) {
                v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
            }
        }
        if (use_temporal_step_stabilizer && !previous_fed_step_embed.empty() &&
            previous_fed_step_embed.size() == step_embed.size()) {
            if (is_kugelaudio && kugelaudio_should_dump_frame(total_frames)) {
                kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "step_embed_pre_temporal"), step_embed,
                                           "per-frame step embedding before temporal stabilizer");
            }
            for (size_t i = 0; i < step_embed.size(); ++i) {
                step_embed[i] = temporal_step_alpha * step_embed[i] +
                                (1.0f - temporal_step_alpha) * previous_fed_step_embed[i];
            }
        }
        previous_fed_step_embed = step_embed;
        if (is_kugelaudio && kugelaudio_should_dump_frame(total_frames)) {
            kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "step_embed"), step_embed,
                                       "per-frame next-step embedding produced from diffusion latent via acoustic connector");
        }
        if (is_kugelaudio && !dumped_first_diffusion) {
            kugelaudio_dump_f32_vector("19_first_step_embed", step_embed,
                                       "next-step embedding produced from first diffusion latent via acoustic connector");
            dumped_first_diffusion = true;
        }

        // 7c. step LM by 1 position with the speech embedding (positive).
        if (use_rolling_kv) {
            rolling_step_history.insert(rolling_step_history.end(), step_embed.begin(), step_embed.end());
            const int history_frames = static_cast<int>(rolling_step_history.size() / static_cast<size_t>(hidden));
            const int keep_frames = std::min(rolling_kv_frames, history_frames);
            const int rolling_max_seq = N + keep_frames + 32;
            if (!kv_lm.init(cfg.n_layers_lm, cfg.head_dim, cfg.n_kv_heads, rolling_max_seq)) {
                VV_LOG_ERROR("tts_15b: rolling KV init failed at frame %d", total_frames);
                return -10;
            }
            if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                                 /*past_len=*/0, /*n_new=*/N, embeds.data(),
                                 &kv_lm, nullptr, &hidden_last)) {
                VV_LOG_ERROR("tts_15b: rolling KV prompt rebuild failed at frame %d", total_frames);
                return -13;
            }
            if (keep_frames > 0) {
                const size_t begin_frame = static_cast<size_t>(history_frames - keep_frames);
                const float* hist_ptr = rolling_step_history.data() + begin_frame * static_cast<size_t>(hidden);
                const int history_pos_start = N + history_frames - keep_frames;
                if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                                     history_pos_start, /*n_new=*/keep_frames, hist_ptr,
                                     &kv_lm, nullptr, &hidden_last)) {
                    VV_LOG_ERROR("tts_15b: rolling KV history rebuild failed at frame %d", total_frames);
                    return -13;
                }
            }
            lm_pos = N + history_frames;
        } else {
            if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                                 lm_pos, /*n_new=*/1, step_embed.data(),
                                 &kv_lm, nullptr, &hidden_last)) {
                VV_LOG_ERROR("tts_15b: LM step failed at frame %d", total_frames);
                return -13;
            }
            ++lm_pos;
        }
        if (is_kugelaudio && kugelaudio_should_dump_frame(total_frames)) {
            kugelaudio_dump_f32_vector(kugelaudio_frame_stage(total_frames, "hidden_after_step"), hidden_last,
                                       "per-frame LM hidden state after feeding generated speech embedding");
        }

        // 7c'. negative branch sees the same speech embed.
        if (use_cfg) {
            if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                                 neg_pos, /*n_new=*/1, step_embed.data(),
                                 &kv_neg, nullptr, &neg_hidden_last)) {
                VV_LOG_ERROR("tts_15b: neg LM step failed at frame %d", total_frames);
                return -13;
            }
            ++neg_pos;
        }
        ++total_frames;
        if (p.verbose) {
            log_tts_frame_progress(total_frames, p.max_speech_frames, cfg.sample_rate);
        }

        // 7d. legacy non-KugelAudio speech-end detection from LM logits.
        if (!is_kugelaudio) {
            auto logits = detail::lm_head_logits_last(model->lm_head, hidden_last,
                                                      hidden, cfg.vocab_size);
            if (static_cast<int>(logits.size()) == cfg.vocab_size) {
                int   best_id = 0;
                float best_v  = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < cfg.vocab_size; ++i) {
                    if (logits[i] > best_v) { best_v = logits[i]; best_id = i; }
                }
                if (best_id == kSpeech15bEndId) {
                    if (p.verbose) std::fprintf(stderr,
                        "[tts_15b] speech_end at frame %d\n", total_frames);
                    finished = true;
                }
            }
        }
    }

    // Optional img2img-style refinement pass over generated latents. This is
    // deliberately decode-only: LM feedback has already consumed first-pass
    // latents, so refinement cannot introduce generated waveform/state carry.
    const int n_frames = static_cast<int>(all_latents.size() / cfg.latent);
    if (use_latent_refine && n_frames > 0) {
        int refine_steps = p.latent_refine_steps > 0
            ? std::min(p.latent_refine_steps, solver_cfg.num_inference_steps)
            : static_cast<int>(std::ceil(latent_refine_strength * static_cast<float>(solver_cfg.num_inference_steps)));
        refine_steps = std::clamp(refine_steps, 1, solver_cfg.num_inference_steps);
        const int start_step = solver_cfg.num_inference_steps - refine_steps;
        const int t_refine = solver_state.timesteps[start_step];
        const float a_refine = solver_state.alpha_t[t_refine];
        const float sg_refine = solver_state.sigma_t[t_refine];
        VV_LOG_INFO("tts_15b: latent refinement frames=%d strength=%.3f start_step=%d reverse_steps=%d timestep=%d",
                    n_frames, static_cast<double>(latent_refine_strength), start_step, refine_steps, t_refine);
        for (int f = 0; f < n_frames; ++f) {
            std::vector<float> z(cfg.latent);
            std::vector<float> initial_noise(cfg.latent);
            for (int k = 0; k < cfg.latent; ++k) {
                initial_noise[static_cast<size_t>(k)] = norm(rng);
                const float x0 = all_latents[static_cast<size_t>(f) * static_cast<size_t>(cfg.latent) + static_cast<size_t>(k)];
                z[static_cast<size_t>(k)] = a_refine * x0 + sg_refine * initial_noise[static_cast<size_t>(k)];
            }
            std::vector<float> cond(latent_refine_cond_history.begin() + static_cast<std::ptrdiff_t>(f) * hidden,
                                    latent_refine_cond_history.begin() + static_cast<std::ptrdiff_t>(f + 1) * hidden);
            std::vector<float> cond_neg;
            if (use_cfg && latent_refine_cond_neg_history.size() >= static_cast<size_t>(f + 1) * static_cast<size_t>(hidden)) {
                cond_neg.assign(latent_refine_cond_neg_history.begin() + static_cast<std::ptrdiff_t>(f) * hidden,
                                latent_refine_cond_neg_history.begin() + static_cast<std::ptrdiff_t>(f + 1) * hidden);
            }
            std::vector<float> refine_variance_noise;
            if (solver_cfg.sde_dpmsolver_plus_plus) {
                refine_variance_noise.assign(static_cast<size_t>(solver_cfg.num_inference_steps) * static_cast<size_t>(cfg.latent), 0.0f);
                for (int si = start_step; si < solver_cfg.num_inference_steps; ++si) {
                    for (int k = 0; k < cfg.latent; ++k) {
                        refine_variance_noise[static_cast<size_t>(si) * static_cast<size_t>(cfg.latent) + static_cast<size_t>(k)] = norm(rng);
                    }
                }
            }
            if (dpm_solver_sample_from_step(z, start_step, cfg.latent, 1, 1,
                                            cond, hidden, w.dh, dh_cfg,
                                            solver_cfg, solver_state,
                                            cond_neg, p.cfg_scale,
                                            solver_cfg.sde_dpmsolver_plus_plus ? &refine_variance_noise : nullptr,
                                            nullptr, nullptr) != 0) {
                VV_LOG_ERROR("tts_15b: latent refinement failed at frame %d", f);
                return -18;
            }
            std::memcpy(all_latents.data() + static_cast<size_t>(f) * static_cast<size_t>(cfg.latent),
                        z.data(), sizeof(float) * static_cast<size_t>(cfg.latent));
        }
    }

    // ---- 8. decode latents to waveform ----
    if (n_frames > 0) {
        std::vector<float> scaled(all_latents.size());
        for (size_t i = 0; i < all_latents.size(); ++i) {
            scaled[i] = all_latents[i] / cfg.speech_scaling - cfg.speech_bias;
        }
        // Repack from per-frame [latent] to ggml-order [vae_dim, n_frames].
        std::vector<float> packed(scaled.size());
        for (int t = 0; t < n_frames; ++t) {
            for (int d = 0; d < cfg.latent; ++d) {
                packed[d * n_frames + t] = scaled[t * cfg.latent + d];
            }
        }
        const bool active_backend_is_cpu = ggml_backend_is_cpu(vv::backend());
        const bool use_cpu_final_decoder =
            p.final_decoder_backend == FinalDecoderBackend::Cpu &&
            !active_backend_is_cpu;
        const bool use_auto_stream_final_decoder =
            p.final_decoder_backend == FinalDecoderBackend::Auto &&
            !active_backend_is_cpu;
        const bool use_stream_final_decoder =
            p.final_decoder_backend == FinalDecoderBackend::StreamActive ||
            use_auto_stream_final_decoder;
        if (use_cpu_final_decoder) {
            if (!ensure_cpu_decoder_shadow(model)) {
                VV_LOG_ERROR("tts_15b: failed to prepare CPU shadow decoder for hybrid final decode");
                return -14;
            }
            VV_LOG_INFO("tts_15b: using hybrid final decoder backend=CPU for %d latent frames", n_frames);
            *samples = decode_latent_sequence_cpu(cfg, model->cpu_decoder_shadow, packed.data(), n_frames);
        } else if (use_stream_final_decoder) {
            VV_LOG_INFO("tts_15b: using streamed final decoder on backend=%s for %d latent frames (chunk=%d%s)",
                        vv::backend_name(), n_frames, streamed_decoder_chunk_frames(),
                        use_auto_stream_final_decoder ? ", auto" : "");
            *samples = decode_latent_sequence_streaming(cfg, w, packed.data(), n_frames);
        } else {
            *samples = decode_latent_sequence(cfg, w, packed.data(), n_frames);
        }
        if (samples->empty()) {
            VV_LOG_ERROR("tts_15b: final decode produced no samples (backend=%s, final_decoder=%s)",
                         vv::backend_name(),
                         use_cpu_final_decoder ? "cpu" : (use_stream_final_decoder ? "stream_active" : "active"));
            return -15;
        }
    }
    if (latent_tail_out) {
        latent_tail_out->latents.clear();
        latent_tail_out->frames = 0;
        if (n_frames > 0 && cfg.latent > 0) {
            const int frame_samples = cfg.sample_rate > 0 ? kSpeech15bCompressRatio : kSpeech15bCompressRatio;
            int keep_frames = p.continuity_tail_ms > 0 && cfg.sample_rate > 0
                ? static_cast<int>((static_cast<int64_t>(p.continuity_tail_ms) * cfg.sample_rate + frame_samples * 1000 - 1) /
                                   (frame_samples * 1000))
                : 0;
            keep_frames = std::clamp(keep_frames, 0, n_frames);
            if (keep_frames > 0) {
                const size_t begin = static_cast<size_t>(n_frames - keep_frames) * static_cast<size_t>(cfg.latent);
                latent_tail_out->latents.assign(all_latents.begin() + static_cast<std::ptrdiff_t>(begin), all_latents.end());
                latent_tail_out->frames = keep_frames;
                VV_LOG_INFO("tts_15b: captured latent-prefix continuity tail frames=%d", keep_frames);
            }
        }
    }
    if (suppressed_stop_tokens > 0) {
        VV_LOG_INFO("tts_15b: suppressed %d speech_end/eos token(s) before min_frames=%d",
                    suppressed_stop_tokens, p.min_speech_frames);
    }
    if (eos_guard_suppressed_stop_tokens > 0) {
        VV_LOG_INFO("tts_15b: eos guard suppressed %d speech_end/eos token(s) after audio frames",
                    eos_guard_suppressed_stop_tokens);
    }
    if (p.verbose) std::fprintf(stderr,
        "[tts_15b] %d frames -> %zu samples\n", n_frames, samples->size());
    return 0;
}

int tts_15b_generate_segmented_state(VibeVoiceModel*                         model,
                                     const detail::KugelAudioChunkPlan&      plan,
                                     const VibeVoiceTTSParams&               p,
                                     std::vector<float>*                     samples) {
    if (!model || !samples) return -1;
    if (model->variant != "1.5b" || !model->loader.has_key("kugelaudio.architecture")) {
        VV_LOG_ERROR("tts_15b: segmented-state is only supported for KugelAudio 1.5B-style TTS");
        return -1;
    }
    if (p.cfg_scale > 1.0f) {
        VV_LOG_ERROR("tts_15b: segmented-state currently requires --cfg 1.0 (got %.2f)", static_cast<double>(p.cfg_scale));
        return -32;
    }
    if (plan.chunks.empty()) return -24;

    const auto& cfg = model->cfg;
    const auto& w = model->w;
    const int hidden = cfg.hidden;
    const bool cast_step_embed_input_f16 = kugelaudio_cast_step_embed_f16(model, p);
    if (cast_step_embed_input_f16) {
        VV_LOG_INFO("tts_15b: rounding generated step embeddings to f16 before LM feedback");
    }
    samples->clear();

    PreparedSpeechConditioning conditioning;
    std::mt19937 rng(p.seed ? p.seed : std::random_device{}());
    PreparedSpeechConditioningResult prep = prepare_speech_conditioning(model, p, /*is_kugelaudio=*/true, rng, &conditioning);
    if (prep.rc != 0) return prep.rc;
    if (prep.stopped_after_preprocessing || prep.stopped_after_connectors) {
        samples->assign(240, 0.0f);
        return 0;
    }
    if (conditioning.per_speaker_Tc.empty()) return -6;

    auto build_transition_ids = [&](const std::string& chunk_text) {
        std::vector<int32_t> ids;
        auto append = [&](const std::string& s) {
            auto part = model->tokenizer.encode(s);
            ids.insert(ids.end(), part.begin(), part.end());
        };
        append("\n Text input:\n");
        append(" " + format_kugelaudio_single_speaker_text(chunk_text) + "\n");
        append(" Speech output:\n");
        ids.push_back(kSpeech15bStartId);
        return ids;
    };

    std::vector<int32_t> input_ids;
    std::vector<int> pad_positions;
    build_kugelaudio_prompt_input_ids(model->tokenizer,
                                      conditioning.per_speaker_Tc.front(),
                                      plan.chunks.front(),
                                      &input_ids,
                                      &pad_positions);
    if (input_ids.empty() || static_cast<int>(pad_positions.size()) != conditioning.total_Tc) {
        VV_LOG_ERROR("tts_15b: segmented-state prompt setup failed");
        return -8;
    }

    std::vector<std::vector<int32_t>> transitions(plan.chunks.size());
    size_t transition_token_count = 0;
    for (size_t i = 1; i < plan.chunks.size(); ++i) {
        transitions[i] = build_transition_ids(plan.chunks[i]);
        transition_token_count += transitions[i].size();
    }

    auto embed_ids = [&](const std::vector<int32_t>& ids, std::vector<float>* embeds) -> bool {
        embeds->assign(static_cast<size_t>(hidden) * ids.size(), 0.0f);
        for (size_t t = 0; t < ids.size(); ++t) {
            const int id = ids[t];
            if (id < 0 || id >= cfg.vocab_size) {
                VV_LOG_ERROR("tts_15b: token id out of range in segmented-state: %d", id);
                return false;
            }
            embed_row(w.lm_tok_embd, id, hidden, embeds->data() + static_cast<size_t>(hidden) * t);
        }
        return true;
    };

    std::vector<float> embeds;
    if (!embed_ids(input_ids, &embeds)) return -9;
    for (int k = 0; k < conditioning.total_Tc; ++k) {
        const int pos = pad_positions[k];
        std::memcpy(&embeds[static_cast<size_t>(hidden) * pos],
                    &conditioning.speech_features[static_cast<size_t>(hidden) * k],
                    sizeof(float) * hidden);
    }

    const int per_segment_frames = std::max(1, p.max_speech_frames);
    const int total_frame_budget = per_segment_frames * static_cast<int>(plan.chunks.size());
    const int max_seq = static_cast<int>(input_ids.size() + transition_token_count) + total_frame_budget +
                        static_cast<int>(plan.chunks.size()) * 4 + 32;

    ResidentKV kv_lm;
    if (!kv_lm.init(cfg.n_layers_lm, cfg.head_dim, cfg.n_kv_heads, max_seq)) {
        VV_LOG_ERROR("tts_15b: segmented-state resident KV init failed");
        return -10;
    }

    std::vector<float> hidden_last;
    if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                         /*pos_start=*/0, static_cast<int>(input_ids.size()), embeds.data(),
                         &kv_lm, nullptr, &hidden_last)) {
        VV_LOG_ERROR("tts_15b: segmented-state initial prefill failed");
        return -11;
    }
    int lm_pos = static_cast<int>(input_ids.size());

    DPMSolverConfig solver_cfg = detail::kugelaudio_solver_config_for_test(p.n_diffusion_steps);
    DPMSolverState solver_state;
    dpm_solver_init(solver_cfg, &solver_state);
    DiffusionHeadConfig dh_cfg;
    dh_cfg.hidden      = hidden;
    dh_cfg.latent      = cfg.latent;
    dh_cfg.head_layers = cfg.head_layers;
    dh_cfg.ffn_ratio   = cfg.ffn_ratio;
    dh_cfg.eps         = cfg.rms_norm_eps;
    dh_cfg.freq_size   = 256;
    std::normal_distribution<float> norm(0.0f, 1.0f);

    std::vector<float> all_latents;
    all_latents.reserve(static_cast<size_t>(total_frame_budget) * cfg.latent);
    int total_frames = 0;
    int suppressed_segments = 0;

    auto feed_token_id = [&](int32_t token_id) -> bool {
        std::vector<float> tok_embed(static_cast<size_t>(hidden));
        embed_row(w.lm_tok_embd, token_id, hidden, tok_embed.data());
        if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                             lm_pos, 1, tok_embed.data(), &kv_lm, nullptr, &hidden_last)) {
            return false;
        }
        ++lm_pos;
        return true;
    };

    auto feed_ids = [&](const std::vector<int32_t>& ids) -> bool {
        if (ids.empty()) return true;
        std::vector<float> e;
        if (!embed_ids(ids, &e)) return false;
        if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                             lm_pos, static_cast<int>(ids.size()), e.data(), &kv_lm, nullptr, &hidden_last)) {
            return false;
        }
        lm_pos += static_cast<int>(ids.size());
        return true;
    };

    for (size_t chunk_i = 0; chunk_i < plan.chunks.size(); ++chunk_i) {
        if (chunk_i > 0) {
            if (!feed_ids(transitions[chunk_i])) {
                VV_LOG_ERROR("tts_15b: segmented-state transition prefill failed at chunk %zu", chunk_i + 1);
                return -11;
            }
            VV_LOG_INFO("tts_15b: segmented-state inserted text transition chunk=%zu tokens=%zu pos=%d",
                        chunk_i + 1, transitions[chunk_i].size(), lm_pos);
        }

        int segment_frames = 0;
        int control_steps_without_audio = 0;
        bool segment_finished = false;
        while (!segment_finished && segment_frames < per_segment_frames && total_frames < total_frame_budget) {
            auto logits = detail::lm_head_logits_last(model->lm_head, hidden_last, hidden, cfg.vocab_size);
            if (static_cast<int>(logits.size()) != cfg.vocab_size) return -14;
            detail::apply_kugelaudio_speech_end_penalty(&logits);
            int next_token = detail::select_kugelaudio_speech_token_from_logits(logits);
            if (detail::kugelaudio_token_stops_generation(next_token)) {
                if (!feed_token_id(next_token)) {
                    VV_LOG_ERROR("tts_15b: segmented-state stop-token feed failed at chunk %zu", chunk_i + 1);
                    return -13;
                }
                segment_finished = true;
                break;
            }
            if (detail::kugelaudio_token_requires_cfg_reset(next_token)) {
                if (!feed_token_id(kSpeech15bStartId)) {
                    VV_LOG_ERROR("tts_15b: segmented-state speech_start feed failed at chunk %zu", chunk_i + 1);
                    return -13;
                }
                if (++control_steps_without_audio > 32) {
                    VV_LOG_ERROR("tts_15b: segmented-state exceeded control-token guard at chunk %zu", chunk_i + 1);
                    return -15;
                }
                continue;
            }
            control_steps_without_audio = 0;

            std::vector<float> z(cfg.latent);
            for (auto& v : z) v = norm(rng);
            std::vector<float> cond(hidden);
            std::memcpy(cond.data(), hidden_last.data(), sizeof(float) * hidden);
            std::vector<float> cond_neg;
            if (dpm_solver_sample(z, cfg.latent, 1, 1, cond, hidden,
                                  w.dh, dh_cfg, solver_cfg, solver_state,
                                  cond_neg, p.cfg_scale) != 0) {
                VV_LOG_ERROR("tts_15b: segmented-state dpm_solver_sample failed chunk=%zu frame=%d", chunk_i + 1, segment_frames);
                return -12;
            }
            all_latents.insert(all_latents.end(), z.begin(), z.end());
            auto step_embed = run_speech_connector(cfg, w, z.data(), 1);
            if (cast_step_embed_input_f16) {
                for (float& v : step_embed) v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
            }
            if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, w.tlm_output_norm,
                                 lm_pos, 1, step_embed.data(), &kv_lm, nullptr, &hidden_last)) {
                VV_LOG_ERROR("tts_15b: segmented-state LM speech step failed chunk=%zu frame=%d", chunk_i + 1, segment_frames);
                return -13;
            }
            ++lm_pos;
            ++segment_frames;
            ++total_frames;
            if (p.verbose) log_tts_frame_progress(total_frames, total_frame_budget, cfg.sample_rate);
        }
        if (!segment_finished && segment_frames >= per_segment_frames && chunk_i + 1 < plan.chunks.size()) {
            ++suppressed_segments;
        }
        VV_LOG_INFO("tts_15b: segmented-state chunk=%zu/%zu frames=%d total_frames=%d pos=%d finished=%s",
                    chunk_i + 1, plan.chunks.size(), segment_frames, total_frames, lm_pos,
                    segment_finished ? "yes" : "budget");
    }

    const int n_frames = static_cast<int>(all_latents.size() / cfg.latent);
    if (n_frames <= 0) return -15;
    std::vector<float> scaled(all_latents.size());
    for (size_t i = 0; i < all_latents.size(); ++i) scaled[i] = all_latents[i] / cfg.speech_scaling - cfg.speech_bias;
    std::vector<float> packed(scaled.size());
    for (int t = 0; t < n_frames; ++t) {
        for (int d = 0; d < cfg.latent; ++d) packed[d * n_frames + t] = scaled[t * cfg.latent + d];
    }
    const bool active_backend_is_cpu = ggml_backend_is_cpu(vv::backend());
    const bool use_cpu_final_decoder = p.final_decoder_backend == FinalDecoderBackend::Cpu && !active_backend_is_cpu;
    const bool use_auto_stream_final_decoder = p.final_decoder_backend == FinalDecoderBackend::Auto && !active_backend_is_cpu;
    const bool use_stream_final_decoder = p.final_decoder_backend == FinalDecoderBackend::StreamActive || use_auto_stream_final_decoder;
    if (use_cpu_final_decoder) {
        if (!ensure_cpu_decoder_shadow(model)) return -14;
        *samples = decode_latent_sequence_cpu(cfg, model->cpu_decoder_shadow, packed.data(), n_frames);
    } else if (use_stream_final_decoder) {
        VV_LOG_INFO("tts_15b: using streamed final decoder on backend=%s for %d latent frames (chunk=%d%s)",
                    vv::backend_name(), n_frames, streamed_decoder_chunk_frames(),
                    use_auto_stream_final_decoder ? ", auto" : "");
        *samples = decode_latent_sequence_streaming(cfg, w, packed.data(), n_frames);
    } else {
        *samples = decode_latent_sequence(cfg, w, packed.data(), n_frames);
    }
    if (samples->empty()) return -15;
    VV_LOG_INFO("tts_15b: segmented-state complete chunks=%zu frames=%d samples=%zu budget_segments=%d",
                plan.chunks.size(), n_frames, samples->size(), suppressed_segments);
    if (p.verbose) std::fprintf(stderr, "[tts_15b] %d segmented-state frames -> %zu samples\n", n_frames, samples->size());
    return 0;
}
}  // namespace

}  // namespace vv
