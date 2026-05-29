#include "ggml.h"
#include "vibevoice_tts.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

void fill_tensor(struct ggml_tensor* t, const std::vector<float>& values) {
    float* dst = static_cast<float*>(t->data);
    for (size_t i = 0; i < values.size(); ++i) dst[i] = values[i];
}

bool approx_eq(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

}  // namespace

int main() {
    struct ggml_init_params p{};
    p.mem_size = 1ull << 20;
    p.no_alloc = false;
    struct ggml_context* wctx = ggml_init(p);
    if (!wctx) {
        std::fprintf(stderr, "FAIL: ggml_init for connector weights\n");
        return 1;
    }

    vv::VibeVoiceConfig cfg;
    cfg.latent = 2;
    cfg.hidden = 2;

    vv::VibeVoiceWeights w;
    w.ac_fc1_w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, cfg.latent, cfg.hidden);
    w.ac_norm  = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, cfg.hidden);
    w.ac_fc2_w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, cfg.hidden, cfg.hidden);
    if (!w.ac_fc1_w || !w.ac_norm || !w.ac_fc2_w) {
        std::fprintf(stderr, "FAIL: failed to allocate connector tensors\n");
        ggml_free(wctx);
        return 2;
    }

    // All-ones weights make the reused connector path easy to predict:
    // fc1(x) -> [sum(x), sum(x)] ; rms_norm -> [1, 1] ; fc2 -> [2, 2].
    fill_tensor(w.ac_fc1_w, {1.0f, 1.0f,
                             1.0f, 1.0f});
    fill_tensor(w.ac_norm,  {1.0f, 1.0f});
    fill_tensor(w.ac_fc2_w, {1.0f, 1.0f,
                             1.0f, 1.0f});

    const std::vector<float> x = {
        3.0f, 4.0f,  // frame 0
        0.0f, 2.0f,  // frame 1
    };
    const auto out = vv::detail::run_speech_connector_for_test(cfg, w, x.data(), /*batch=*/2);
    if (out.size() != 4) {
        std::fprintf(stderr, "FAIL: connector output size=%zu want 4\n", out.size());
        ggml_free(wctx);
        return 3;
    }

    const std::vector<float> expected = {2.0f, 2.0f, 2.0f, 2.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!approx_eq(out[i], expected[i])) {
            std::fprintf(stderr, "FAIL: connector output[%zu]=%.6f want %.6f\n",
                         i, out[i], expected[i]);
            ggml_free(wctx);
            return 4;
        }
    }

    ggml_free(wctx);
    std::printf("KugelAudio reused connector path OK\n");
    return 0;
}
