#include "backend.hpp"

#include "common.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace vv {

namespace {

ggml_backend_t  g_backend  = nullptr;
ggml_gallocr_t  g_gallocr  = nullptr;
std::string     g_name;
std::once_flag  g_once;
bool            g_flash_attn_supported = false;
std::string     g_requested_backend = "auto";
int             g_requested_device_index = -1;
std::string     g_selected_reason = "uninitialized";
std::string     g_available_devices;
bool            g_verbose = false;

std::string lower(const std::string& s) {
    std::string r = s;
    for (auto& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

const char* getenv_compat(const char* kugelaudio_name, const char* vibevoice_name) {
    if (const char* v = std::getenv(kugelaudio_name); v && *v) return v;
    return std::getenv(vibevoice_name);
}

bool env_truthy(const char* v) {
    if (!v || !*v) return false;
    const std::string s = lower(v);
    return !(s == "0" || s == "false" || s == "off" || s == "no");
}

const char* dev_type_name(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:  return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:  return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:return "accel";
        default:                            return "unknown";
    }
}

std::vector<ggml_backend_dev_t> all_devices() {
    std::vector<ggml_backend_dev_t> out;
    const size_t n = ggml_backend_dev_count();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (ggml_backend_dev_t dev = ggml_backend_dev_get(i)) {
            out.push_back(dev);
        }
    }
    return out;
}

std::vector<ggml_backend_dev_t> devices_by_type(enum ggml_backend_dev_type type) {
    std::vector<ggml_backend_dev_t> out;
    for (ggml_backend_dev_t dev : all_devices()) {
        if (ggml_backend_dev_type(dev) == type) out.push_back(dev);
    }
    return out;
}

bool device_matches_backend_name(ggml_backend_dev_t dev, const std::string& want_lower) {
    if (!dev) return false;
    if (want_lower.empty() || want_lower == "auto") return true;

    const std::string dev_name = lower(ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "");
    const std::string dev_desc = lower(ggml_backend_dev_description(dev) ? ggml_backend_dev_description(dev) : "");
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const std::string reg_name = lower(reg && ggml_backend_reg_name(reg) ? ggml_backend_reg_name(reg) : "");

    if (want_lower == "cpu") {
        return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    }
    if (want_lower == "gpu") {
        return ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
    }
    return dev_name.find(want_lower) != std::string::npos ||
           dev_desc.find(want_lower) != std::string::npos ||
           reg_name.find(want_lower) != std::string::npos;
}

std::vector<ggml_backend_dev_t> matching_devices(const std::string& want_lower) {
    std::vector<ggml_backend_dev_t> out;
    for (ggml_backend_dev_t dev : all_devices()) {
        if (device_matches_backend_name(dev, want_lower)) out.push_back(dev);
    }
    return out;
}

std::string summarize_devices(const std::vector<ggml_backend_dev_t>& devs) {
    std::ostringstream oss;
    for (size_t i = 0; i < devs.size(); ++i) {
        ggml_backend_dev_t dev = devs[i];
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (i) oss << "; ";
        oss << "[" << i << "]"
            << " reg=" << (reg && ggml_backend_reg_name(reg) ? ggml_backend_reg_name(reg) : "<none>")
            << " name=" << (ggml_backend_dev_name(dev) ? ggml_backend_dev_name(dev) : "<none>")
            << " type=" << dev_type_name(ggml_backend_dev_type(dev));
        if (g_verbose) {
            oss << " desc=" << (ggml_backend_dev_description(dev) ? ggml_backend_dev_description(dev) : "<none>");
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(dev, &free_b, &total_b);
            if (total_b > 0) {
                oss << " mem_free=" << free_b << " mem_total=" << total_b;
            }
        }
    }
    return oss.str();
}

int parse_device_index_env() {
    const char* env = getenv_compat("KUGELAUDIO_BACKEND_DEVICE_INDEX", "VIBEVOICE_BACKEND_DEVICE_INDEX");
    if (!env || !*env) return -1;
    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (!end || *end != '\0' || v < 0) {
        VV_LOG_WARN("backend: ignoring invalid KUGELAUDIO_BACKEND_DEVICE_INDEX/VIBEVOICE_BACKEND_DEVICE_INDEX=%s", env);
        return -1;
    }
    return static_cast<int>(v);
}

bool probe_flash_attn_support(ggml_backend_t backend, const std::string& backend_name) {
    const char* env = getenv_compat("KUGELAUDIO_FLASH_ATTN", "VIBEVOICE_FLASH_ATTN");
    if (env && (std::string(env) == "0" || lower(env) == "false" || lower(env) == "off")) {
        VV_LOG_INFO("backend: flash-attn disabled via KUGELAUDIO_FLASH_ATTN/VIBEVOICE_FLASH_ATTN=%s", env);
        return false;
    }
    if (!backend) return false;

    struct ggml_init_params ip {};
    ip.mem_size  = ggml_tensor_overhead() * 16;
    ip.no_alloc  = true;
    ggml_context* ctx = ggml_init(ip);
    const int hd = 128, n_h = 8, n_kv = 2, seq = 16;
    ggml_tensor* q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, n_h,  1);
    ggml_tensor* k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, n_kv, 1);
    ggml_tensor* v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, n_kv, 1);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, seq, seq);
    ggml_tensor* op   = ggml_flash_attn_ext(ctx, q, k, v, mask,
                                            1.0f / 11.31f, 0.0f, 0.0f);
    const bool supported = op && ggml_backend_supports_op(backend, op);
    ggml_free(ctx);

    VV_LOG_INFO("backend: flash-attn=%s selected=%s",
                supported ? "available" : "unavailable",
                backend_name.c_str());
    return supported;
}

ggml_backend_t init_device(ggml_backend_dev_t dev) {
    return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
}

void init() {
    ggml_backend_load_all();

    g_verbose = env_truthy(getenv_compat("KUGELAUDIO_BACKEND_VERBOSE", "VIBEVOICE_BACKEND_VERBOSE"));
    g_requested_device_index = parse_device_index_env();

    const char* env = getenv_compat("KUGELAUDIO_BACKEND", "VIBEVOICE_BACKEND");
    std::string want = env ? lower(env) : "";
    g_requested_backend = want.empty() ? "auto" : want;

    const std::vector<ggml_backend_dev_t> devices = all_devices();
    g_available_devices = summarize_devices(devices);

    if (!want.empty()) {
        std::vector<ggml_backend_dev_t> matches = matching_devices(want);
        if (want == "cpu" && matches.empty()) {
            ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu) matches.push_back(cpu);
        }

        if (!matches.empty()) {
            int match_index = g_requested_device_index >= 0 ? g_requested_device_index : 0;
            if (match_index < static_cast<int>(matches.size())) {
                g_backend = init_device(matches[match_index]);
                if (g_backend) {
                    std::ostringstream reason;
                    reason << "env-request";
                    if (g_requested_device_index >= 0) reason << "+device-index";
                    g_selected_reason = reason.str();
                }
            } else {
                VV_LOG_WARN("backend: requested backend=%s device_index=%d but only %zu matching device(s) are registered — falling back",
                            env, g_requested_device_index, matches.size());
                g_selected_reason = "fallback:requested device index out of range";
            }
        } else {
            VV_LOG_WARN("backend: requested backend=%s but no matching device registered — falling back", env);
            g_selected_reason = "fallback:requested backend unavailable";
        }
    } else if (g_requested_device_index >= 0) {
        std::vector<ggml_backend_dev_t> gpus = devices_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (g_requested_device_index < static_cast<int>(gpus.size())) {
            g_backend = init_device(gpus[g_requested_device_index]);
            if (g_backend) g_selected_reason = "auto+gpu-device-index";
        } else {
            VV_LOG_WARN("backend: KUGELAUDIO_BACKEND_DEVICE_INDEX=%d requested without KUGELAUDIO_BACKEND, but only %zu GPU device(s) are registered — falling back",
                        g_requested_device_index, gpus.size());
            g_selected_reason = "fallback:auto gpu device index out of range";
        }
    }

    if (!g_backend) {
        g_backend = ggml_backend_init_best();
        if (g_backend && g_selected_reason.rfind("fallback:", 0) != 0) {
            g_selected_reason = "auto-best-available";
        }
    }
    if (!g_backend) {
        g_backend = ggml_backend_dev_init(
            ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        g_selected_reason = "cpu-fallback";
    }
    if (!g_backend) {
        g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        g_selected_reason = "cpu-init-by-type-fallback";
    }

    if (g_backend) {
        const char* name = ggml_backend_name(g_backend);
        g_name = name ? name : "(unnamed)";
        if (ggml_backend_is_cpu(g_backend)) {
            ggml_backend_cpu_set_n_threads(g_backend, 0);
        }
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
        g_gallocr = ggml_gallocr_new(buft);
        VV_LOG_INFO("backend: requested=%s device_index=%d selected=%s reason=%s",
                    g_requested_backend.c_str(),
                    g_requested_device_index,
                    g_name.c_str(),
                    g_selected_reason.c_str());
        VV_LOG_INFO("backend: available_devices=%s",
                    g_available_devices.empty() ? "<none>" : g_available_devices.c_str());
        g_flash_attn_supported = probe_flash_attn_support(g_backend, g_name);
    } else {
        VV_LOG_ERROR("backend: failed to initialize any backend");
        g_name = "none";
        g_selected_reason = "failure:no-backend";
    }
}

}  // namespace

ggml_backend_t backend() {
    std::call_once(g_once, init);
    return g_backend;
}

const char* backend_name() {
    std::call_once(g_once, init);
    return g_name.c_str();
}

bool compute_graph(ggml_cgraph* graph) {
    ggml_backend_t b = backend();
    if (!b || !graph || !g_gallocr) return false;
    if (!ggml_gallocr_alloc_graph(g_gallocr, graph)) {
        VV_LOG_ERROR("backend: gallocr_alloc_graph failed");
        return false;
    }
    return ggml_backend_graph_compute(b, graph) == GGML_STATUS_SUCCESS;
}

ggml_backend_buffer_t allocate_ctx_tensors(ggml_context* ctx) {
    ggml_backend_t b = backend();
    if (!b || !ctx) return nullptr;
    return ggml_backend_alloc_ctx_tensors(ctx, b);
}

void backend_tensor_set(ggml_tensor* t, const void* data, size_t offset, size_t size) {
    ggml_backend_t b = backend();
    if (!b || !t || !data || size == 0) return;

    if (g_name.find("Vulkan") != std::string::npos) {
        constexpr size_t kChunkBytes = 1u << 20; // 1 MiB
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (size_t done = 0; done < size; done += kChunkBytes) {
            const size_t n = std::min(kChunkBytes, size - done);
            ggml_backend_tensor_set(t, p + done, offset + done, n);
            ggml_backend_synchronize(b);
        }
        return;
    }

    ggml_backend_tensor_set(t, data, offset, size);
}

bool backend_supports_flash_attn() {
    std::call_once(g_once, init);
    return g_flash_attn_supported;
}

}  // namespace vv
