// smoke + tiny end-to-end test for the quantize CLI override surface.

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

std::string run_capture(const std::string& cmd, int* status = nullptr) {
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) {
        if (status) *status = -1;
        return out;
    }
    char buf[4096];
    while (size_t n = std::fread(buf, 1, sizeof(buf), p)) {
        out.append(buf, n);
    }
    const int rc = pclose(p);
    if (status) *status = rc;
    return out;
}

std::string resolve_quantize_binary() {
    if (const char* explicit_bin = std::getenv("KUGELAUDIO_QUANTIZE")) {
        return explicit_bin;
    }
    const std::vector<std::string> cands = {
        "./kugelaudio-quantize",
        "../bin/kugelaudio-quantize",
        "./bin/kugelaudio-quantize",
        "../../build/bin/kugelaudio-quantize",
    };
    for (const auto& c : cands) {
        if (file_exists(c)) return c;
    }
    return "";
}

bool expect_contains(const std::string& s, const std::string& needle) {
    return s.find(needle) != std::string::npos;
}

std::string make_tmp_path(const char* stem) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "/tmp/%s-%d-%ld.gguf", stem, int(getpid()), random());
    return buf;
}

bool create_small_fixture(const std::string& path) {
    gguf_context* gguf = gguf_init_empty();
    if (!gguf) return false;
    gguf_set_val_str(gguf, "general.architecture", "vibevoice");
    gguf_set_val_str(gguf, "kugelaudio.architecture", "kugelaudio");

    ggml_init_params ip{};
    ip.mem_size = 1u << 20;
    ip.no_alloc = false;
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) {
        gguf_free(gguf);
        return false;
    }

    ggml_tensor* attn = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    ggml_set_name(attn, "lm.blk.0.attn_q.weight");
    ggml_tensor* keep = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    ggml_set_name(keep, "ac.fc1.weight");
    if (!attn || !keep) {
        ggml_free(ctx);
        gguf_free(gguf);
        return false;
    }

    float* a = static_cast<float*>(attn->data);
    float* k = static_cast<float*>(keep->data);
    for (int i = 0; i < 32; ++i) {
        a[i] = float(i + 1);
        k[i] = float(100 + i);
    }

    gguf_add_tensor(gguf, attn);
    gguf_add_tensor(gguf, keep);
    const bool ok = gguf_write_to_file(gguf, path.c_str(), false);
    ggml_free(ctx);
    gguf_free(gguf);
    return ok;
}

bool tensor_type_is(const std::string& path, const char* name, ggml_type want) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{};
    p.no_alloc = true;
    p.ctx = &ctx;
    gguf_context* gguf = gguf_init_from_file(path.c_str(), p);
    if (!gguf) return false;
    const int64_t idx = gguf_find_tensor(gguf, name);
    const bool ok = idx >= 0 && gguf_get_tensor_type(gguf, idx) == want;
    gguf_free(gguf);
    ggml_free(ctx);
    return ok;
}

bool kv_string_is(const std::string& path, const char* key, const char* want) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{};
    p.no_alloc = true;
    p.ctx = &ctx;
    gguf_context* gguf = gguf_init_from_file(path.c_str(), p);
    if (!gguf) return false;
    const int64_t idx = gguf_find_key(gguf, key);
    const bool ok = idx >= 0 && std::strcmp(gguf_get_val_str(gguf, idx), want) == 0;
    gguf_free(gguf);
    ggml_free(ctx);
    return ok;
}

}  // namespace

int main() {
    srandom(12345);

    const std::string q = resolve_quantize_binary();
    if (q.empty()) {
        std::fprintf(stderr, "skip: kugelaudio-quantize binary not found (set KUGELAUDIO_QUANTIZE)\n");
        return 77;
    }

    const std::string help = run_capture(std::string("\"") + q + "\" --help 2>&1");
    if (help.empty()) {
        std::fprintf(stderr, "FAIL: no output from --help\n");
        return 1;
    }

    const std::vector<std::string> required = {
        "--attn-type",
        "--ffn-type",
        "--lm-head-type",
        "--embed-type",
        "--ac-connector-type",
        "--sc-connector-type",
        "--dh-type",
        "--at-block-ffn-type",
        "--st-block-ffn-type",
    };
    for (const auto& k : required) {
        if (!expect_contains(help, k)) {
            std::fprintf(stderr, "FAIL: missing CLI flag %s in --help output\n", k.c_str());
            return 1;
        }
    }

    const std::string src = make_tmp_path("vv-quant-src");
    const std::string out = make_tmp_path("vv-quant-out");
    if (!create_small_fixture(src)) {
        std::fprintf(stderr, "FAIL: could not create tiny gguf fixture\n");
        return 1;
    }

    int status = 0;
    const std::string log = run_capture(
        std::string("\"") + q + "\" --src \"" + src + "\" --out \"" + out +
        "\" --type f16 --attn-type q8_0 2>&1",
        &status);
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "FAIL: quantize command failed\n%s\n", log.c_str());
        return 1;
    }
    if (!expect_contains(log, "loaded ") || !expect_contains(log, "progress: starting streamed quantization") ||
        !expect_contains(log, "progress: 2/2 tensors") || !expect_contains(log, "wrote ")) {
        std::fprintf(stderr, "FAIL: unexpected quantize log\n%s\n", log.c_str());
        return 1;
    }

    if (!tensor_type_is(out, "lm.blk.0.attn_q.weight", GGML_TYPE_Q8_0)) {
        std::fprintf(stderr, "FAIL: quantized tensor did not become Q8_0\n");
        return 1;
    }
    if (!tensor_type_is(out, "ac.fc1.weight", GGML_TYPE_F32)) {
        std::fprintf(stderr, "FAIL: passthrough tensor did not stay F32\n");
        return 1;
    }
    if (!kv_string_is(out, "general.architecture", "kugelaudio")) {
        std::fprintf(stderr, "FAIL: quantized KugelAudio fixture did not rewrite general.architecture to kugelaudio\n");
        return 1;
    }

    std::remove(src.c_str());
    std::remove(out.c_str());
    std::printf("quantize CLI mix options exposed and streaming rewrite succeeded\n");
    return 0;
}
