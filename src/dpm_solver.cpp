#include "dpm_solver.hpp"
#include "backend.hpp"
#include "common.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace vv {

namespace {

float alpha_bar_cosine(float t) {
    const float pi = 3.14159265358979323846f;
    const float v  = std::cos((t + 0.008f) / 1.008f * pi * 0.5f);
    return v * v;
}

}  // namespace

void dpm_solver_init(const DPMSolverConfig& cfg, DPMSolverState* state) {
    const int N = cfg.num_train_timesteps;

    std::vector<float> betas(N);
    for (int i = 0; i < N; ++i) {
        const float t1 = static_cast<float>(i)     / N;
        const float t2 = static_cast<float>(i + 1) / N;
        const float a1 = alpha_bar_cosine(t1);
        const float a2 = alpha_bar_cosine(t2);
        betas[i] = std::min(1.0f - a2 / a1, 0.999f);
    }

    state->alpha_t.resize(N);
    state->sigma_t.resize(N);
    state->lambda_t.resize(N);
    double ac = 1.0;
    for (int i = 0; i < N; ++i) {
        ac *= (1.0 - betas[i]);
        state->alpha_t[i]  = static_cast<float>(std::sqrt(ac));
        state->sigma_t[i]  = static_cast<float>(std::sqrt(1.0 - ac));
        state->lambda_t[i] = std::log(state->alpha_t[i]) - std::log(state->sigma_t[i]);
    }

    // Uniform timestep selection: linspace(N-1, 0, M+1).round(), final entry = -1
    // (we model the final entry as -1 = sigma=0; see step logic).
    const int M = cfg.num_inference_steps;
    state->timesteps.assign(M + 1, 0);
    for (int i = 0; i <= M; ++i) {
        const float u = static_cast<float>(i) / M;       // 0..1
        const float t = (1.0f - u) * (N - 1);            // N-1..0
        state->timesteps[i] = static_cast<int>(std::round(t));
    }
    // Make the last step explicitly the "alpha=1, sigma=0" terminal state.
    state->timesteps.back() = -1;
}

namespace {

// Build a per-step graph: y = head(noisy, cond, t)   with all weights captured
// from the loader's ctx.
struct StepResult {
    std::vector<float> v;
};

bool run_head_step(const DiffusionHeadWeights& w,
                   const DiffusionHeadConfig&  cfg,
                   const std::vector<float>&   noisy,
                   const std::vector<float>&   cond,
                   int latent, int frames, int batch, int hidden,
                   float t_value,
                   StepResult*                 result) {
    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead();
    p.no_alloc = true;
    struct ggml_context* ctx = ggml_init(p);

    struct ggml_tensor* x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, latent, frames, batch);
    struct ggml_tensor* cd = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, frames, batch);
    struct ggml_tensor* ts = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.freq_size, batch);
    struct ggml_tensor* y  = diffusion_head_forward(ctx, x, cd, ts, w, cfg);
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx);
    if (!in_buf) { ggml_free(ctx); return false; }
    vv::backend_tensor_set(x,  noisy.data(), 0, sizeof(float) * noisy.size());
    vv::backend_tensor_set(cd, cond.data(),  0, sizeof(float) * cond.size());
    std::vector<float> ts_v(static_cast<size_t>(cfg.freq_size) * batch);
    for (int b = 0; b < batch; ++b) {
        timestep_sinusoidal(t_value, cfg.freq_size,
                            ts_v.data() + b * cfg.freq_size);
        for (int k = 0; k < cfg.freq_size; ++k) {
            float& v = ts_v[static_cast<size_t>(b) * cfg.freq_size + k];
            v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
        }
    }
    vv::backend_tensor_set(ts, ts_v.data(), 0, sizeof(float) * ts_v.size());

    if (!vv::compute_graph(gf)) {
        ggml_backend_buffer_free(in_buf);
        ggml_free(ctx);
        return false;
    }
    const size_t n = static_cast<size_t>(latent) * frames * batch;
    result->v.assign(n, 0.0f);
    ggml_backend_tensor_get(y, result->v.data(), 0, sizeof(float) * n);
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx);
    return true;
}

}  // namespace

int dpm_solver_sample_from_step(std::vector<float>&         x,
                                int                         start_step,
                                int                         latent,
                                int                         frames,
                                int                         batch,
                                const std::vector<float>&   cond,
                                int                         hidden,
                                const DiffusionHeadWeights& w,
                                const DiffusionHeadConfig&  head_cfg,
                                const DPMSolverConfig&      solver_cfg,
                                const DPMSolverState&       state,
                                const std::vector<float>&   cond_neg,
                                float                       cfg_scale,
                                const std::vector<float>*   variance_noise,
                                const std::function<void(int, const std::vector<float>&)>* trace_model_output,
                                const std::function<void(int, const std::vector<float>&)>* trace_step) {
    const int M       = solver_cfg.num_inference_steps;
    const size_t n_el = static_cast<size_t>(latent) * frames * batch;

    if (x.size() != n_el || cond.size() != static_cast<size_t>(hidden) * frames * batch) {
        VV_LOG_ERROR("dpm_solver_sample: bad shapes");
        return 1;
    }
    if (static_cast<int>(state.timesteps.size()) != M + 1) {
        VV_LOG_ERROR("dpm_solver_sample: state.timesteps mismatched length");
        return 2;
    }
    if (start_step < 0 || start_step >= M) {
        VV_LOG_ERROR("dpm_solver_sample: bad start_step=%d for M=%d", start_step, M);
        return 5;
    }
    const bool use_cfg = !cond_neg.empty() && cfg_scale > 1.0f &&
                         cond_neg.size() == cond.size();
    const bool use_sde = solver_cfg.sde_dpmsolver_plus_plus;
    if (use_sde && variance_noise && variance_noise->size() != static_cast<size_t>(M) * n_el) {
        VV_LOG_ERROR("dpm_solver_sample: variance_noise has bad shape");
        return 4;
    }

    std::vector<float> prev_x0(n_el, 0.0f);

    for (int i = start_step; i < M; ++i) {
        const int t = state.timesteps[i];
        const int s = state.timesteps[i + 1];

        const float a_t = state.alpha_t[t];
        const float sg_t = state.sigma_t[t];
        const float l_t  = state.lambda_t[t];
        const float l_prev = (i > start_step) ? state.lambda_t[state.timesteps[i - 1]] : 0.0f;

        float a_s, sg_s, l_s;
        if (s == -1) { a_s = 1.0f; sg_s = 0.0f; l_s = std::numeric_limits<float>::infinity(); }
        else         { a_s = state.alpha_t[s]; sg_s = state.sigma_t[s]; l_s = state.lambda_t[s]; }

        StepResult r;
        if (!run_head_step(w, head_cfg, x, cond, latent, frames, batch, hidden,
                           static_cast<float>(t), &r)) return 3;

        if (use_cfg) {
            StepResult r_neg;
            if (!run_head_step(w, head_cfg, x, cond_neg, latent, frames, batch, hidden,
                               static_cast<float>(t), &r_neg)) return 3;
            for (size_t k = 0; k < r.v.size(); ++k) {
                r.v[k] = r_neg.v[k] + cfg_scale * (r.v[k] - r_neg.v[k]);
            }
        }

        if (trace_model_output && *trace_model_output) {
            (*trace_model_output)(i, r.v);
        }

        std::vector<float> x0(n_el);
        for (size_t k = 0; k < n_el; ++k) x0[k] = a_t * x[k] - sg_t * r.v[k];

        const bool is_first = (i == start_step);
        const bool is_last  = (i == M - 1);
        const int  order    = (is_first || (solver_cfg.lower_order_final && is_last))
                              ? 1 : solver_cfg.solver_order;

        if (s == -1) {
            std::copy(x0.begin(), x0.end(), x.begin());
        } else if (use_sde) {
            const float h = l_s - l_t;
            const float exp_m_h = std::exp(-h);
            const float exp_m_2h = std::exp(-2.0f * h);
            const float A = (sg_s / sg_t) * exp_m_h;
            const float B = a_s * (1.0f - exp_m_2h);
            const float C = sg_s * std::sqrt(std::max(0.0f, 1.0f - exp_m_2h));
            const float* noise = variance_noise ? variance_noise->data() + static_cast<size_t>(i) * n_el : nullptr;
            if (order == 1) {
                for (size_t k = 0; k < n_el; ++k) {
                    const float eta = noise ? noise[k] : 0.0f;
                    x[k] = A * x[k] + B * x0[k] + C * eta;
                }
            } else {
                const float h_0 = l_t - l_prev;
                const float r0 = h_0 / h;
                const float Bd1 = 0.5f * B;
                for (size_t k = 0; k < n_el; ++k) {
                    const float D0 = x0[k];
                    const float D1 = (1.0f / r0) * (x0[k] - prev_x0[k]);
                    const float eta = noise ? noise[k] : 0.0f;
                    x[k] = A * x[k] + B * D0 + Bd1 * D1 + C * eta;
                }
            }
        } else if (order == 1) {
            const float h = l_s - l_t;
            const float A = sg_s / sg_t;
            const float B = a_s * (std::exp(-h) - 1.0f);
            for (size_t k = 0; k < n_el; ++k) x[k] = A * x[k] - B * x0[k];
        } else {
            const float h     = l_s - l_t;
            const float h_0   = l_t - l_prev;
            const float r0    = h_0 / h;
            const float A     = sg_s / sg_t;
            const float B     = a_s * (std::exp(-h) - 1.0f);
            const float Bd1   = 0.5f * B;
            for (size_t k = 0; k < n_el; ++k) {
                const float D0 = x0[k];
                const float D1 = (1.0f / r0) * (x0[k] - prev_x0[k]);
                x[k] = A * x[k] - B * D0 - Bd1 * D1;
            }
        }
        if (solver_cfg.cast_sample_to_f16) {
            for (float& v : x) {
                v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
            }
        }
        if (trace_step && *trace_step) {
            (*trace_step)(i, x);
        }
        std::copy(x0.begin(), x0.end(), prev_x0.begin());
    }
    return 0;
}

int dpm_solver_sample(std::vector<float>&         x,
                      int                         latent,
                      int                         frames,
                      int                         batch,
                      const std::vector<float>&   cond,
                      int                         hidden,
                      const DiffusionHeadWeights& w,
                      const DiffusionHeadConfig&  head_cfg,
                      const DPMSolverConfig&      solver_cfg,
                      const DPMSolverState&       state,
                      const std::vector<float>&   cond_neg,
                      float                       cfg_scale,
                      const std::vector<float>*   variance_noise,
                      const std::function<void(int, const std::vector<float>&)>* trace_model_output,
                      const std::function<void(int, const std::vector<float>&)>* trace_step) {
    return dpm_solver_sample_from_step(x, 0, latent, frames, batch, cond, hidden,
                                       w, head_cfg, solver_cfg, state, cond_neg,
                                       cfg_scale, variance_noise,
                                       trace_model_output, trace_step);
}

}  // namespace vv
