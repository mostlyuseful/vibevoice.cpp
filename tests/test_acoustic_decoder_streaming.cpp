#include "acoustic_tokenizer.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include "model_loader.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
bool file_ok(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}
}

int main() {
#ifndef VV_FIXTURES_DIR
#  define VV_FIXTURES_DIR "tests/fixtures"
#endif
    const std::string path = std::string(VV_FIXTURES_DIR) + "/acoustic.gguf";
    if (!file_ok(path)) {
        std::fprintf(stderr, "skip: missing %s\n  run: python tests/dump_acoustic_reference.py --out %s\n",
                     path.c_str(), path.c_str());
        return 77;
    }

    vv::ModelLoader loader;
    if (!loader.load(path)) return 1;

    vv::AcousticConfig cfg;
    cfg.channels  = loader.get_i32("acoustic.channels");
    cfg.vae_dim   = loader.get_i32("acoustic.vae_dim");
    cfg.n_filters = loader.get_i32("acoustic.n_filters");
    cfg.eps       = loader.get_f32("acoustic.eps", 1e-5f);
    auto ratios   = loader.get_i32_array("acoustic.ratios");
    auto depths   = loader.get_i32_array("acoustic.depths");
    cfg.ratios.assign(ratios.begin(), ratios.end());
    cfg.depths.assign(depths.begin(), depths.end());
    const int T_dec_in = loader.get_i32("acoustic.T_dec_in");

    vv::DecoderWeights dw;
    if (!vv::load_decoder(loader, "dec", cfg, &dw)) return 2;

    struct ggml_tensor* z_t       = loader.tensor("test.decoder_input");
    struct ggml_tensor* dec_exp_t = loader.tensor("test.decoder_output");
    if (!z_t || !dec_exp_t) return 3;

    vv::StreamingCache cache;
    cache.is_first_chunk = true;
    const int chunk_frames = 1;
    std::vector<float> all;
    const float* z_src = static_cast<const float*>(z_t->data);
    const int latent = static_cast<int>(z_t->ne[1]);

    for (int off = 0; off < T_dec_in; off += chunk_frames) {
        const int seg_T = std::min(chunk_frames, T_dec_in - off);
        cache.is_final_chunk = (off + seg_T == T_dec_in);

        struct ggml_init_params p{};
        p.mem_size = 256ull * 1024 * 1024;
        p.no_alloc = false;
        struct ggml_context* ctx = ggml_init(p);
        if (!ctx) return 4;

        struct ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seg_T, latent, 1);
        float* xdst = static_cast<float*>(x->data);
        for (int d = 0; d < latent; ++d) {
            for (int t = 0; t < seg_T; ++t) {
                xdst[d * seg_T + t] = z_src[d * T_dec_in + off + t];
            }
        }

        struct ggml_tensor* y = vv::decoder_forward_streaming(ctx, x, dw, cfg, cache);
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        for (auto& kv : cache) {
            if (kv.second.next_view) ggml_build_forward_expand(gf, kv.second.next_view);
        }

        for (auto& kv : cache) {
            auto& e = kv.second;
            if (!e.prefix || e.T == 0) continue;
            const size_t need = static_cast<size_t>(e.T) * e.C;
            float* pdst = static_cast<float*>(e.prefix->data);
            if (cache.is_first_chunk || e.data.size() != need) {
                std::fill(pdst, pdst + need, 0.0f);
            } else {
                std::memcpy(pdst, e.data.data(), need * sizeof(float));
            }
        }

        if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx);
            return 5;
        }

        const size_t n = static_cast<size_t>(y->ne[0]) * y->ne[1] * y->ne[2];
        const float* yd = static_cast<const float*>(y->data);
        all.insert(all.end(), yd, yd + n);

        for (auto& kv : cache) {
            auto& e = kv.second;
            if (!e.next_view || e.T == 0) continue;
            const size_t need = static_cast<size_t>(e.T) * e.C;
            e.data.assign(need, 0.0f);
            std::memcpy(e.data.data(), e.next_view->data, need * sizeof(float));
            e.next_view = nullptr;
            e.prefix = nullptr;
        }
        cache.is_first_chunk = false;
        ggml_free(ctx);
    }

    const size_t n_exp = static_cast<size_t>(dec_exp_t->ne[0]) * dec_exp_t->ne[1] * dec_exp_t->ne[2];
    if (all.size() != n_exp) {
        std::fprintf(stderr, "FAIL: streamed decoder size mismatch: got %zu expected %zu\n", all.size(), n_exp);
        return 6;
    }
    const float* exp = static_cast<const float*>(dec_exp_t->data);
    double max_abs = 0.0, sa = 0.0, sb = 0.0, sab = 0.0;
    for (size_t i = 0; i < n_exp; ++i) {
        const double aa = all[i];
        const double bb = exp[i];
        const double d = std::fabs(aa - bb);
        if (d > max_abs) max_abs = d;
        sa += aa * aa;
        sb += bb * bb;
        sab += aa * bb;
    }
    const double cos = sab / (std::sqrt(sa) * std::sqrt(sb) + 1e-12);
    std::printf("decoder_streaming: max_abs=%.3e cos=%.6f samples=%zu\n", max_abs, cos, all.size());
    return (max_abs < 5e-3 && cos > 0.999) ? 0 : 7;
}
