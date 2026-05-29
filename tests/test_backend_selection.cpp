#include "backend.hpp"
#include "vibevoice.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {

std::string g_logs;

void capture_log(vv_log_level, const char* msg, void*) {
    if (!msg) return;
    g_logs += msg;
    g_logs += "\n";
}

std::string self_path() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    buf[n] = '\0';
    return std::string(buf);
}

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int run_child(const std::string& envs, const std::string& tag, std::string* out) {
    const std::string exe = self_path();
    const std::string log = "/tmp/test_backend_selection_" + tag + ".log";
    const std::string cmd = envs + " \"" + exe + "\" --child > \"" + log + "\" 2>&1";
    const int rc = std::system(cmd.c_str());
    *out = slurp(log);
    return rc;
}

bool contains_all(const std::string& haystack, const char* const* needles, int n) {
    for (int i = 0; i < n; ++i) {
        if (haystack.find(needles[i]) == std::string::npos) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--child") {
        vv_set_log_callback(capture_log, nullptr);
        (void) vv::backend();
        std::printf("backend_name=%s\n", vv::backend_name());
        std::printf("flash_attn=%s\n", vv::backend_supports_flash_attn() ? "yes" : "no");
        std::printf("logs:\n%s", g_logs.c_str());
        return 0;
    }

    std::string out;

    if (run_child("env KUGELAUDIO_BACKEND=cpu KUGELAUDIO_BACKEND_VERBOSE=1 KUGELAUDIO_BACKEND_DEVICE_INDEX=0", "cpu", &out) != 0) {
        std::fprintf(stderr, "FAIL: cpu child failed\n%s\n", out.c_str());
        return 1;
    }
    const char* cpu_needles[] = {
        "backend_name=CPU",
        "backend: requested=cpu device_index=0 selected=CPU reason=env-request+device-index",
        "backend: available_devices=[0]",
        "backend: flash-attn=",
    };
    if (!contains_all(out, cpu_needles, 4)) {
        std::fprintf(stderr, "FAIL: cpu scenario missing expected backend diagnostics\n%s\n", out.c_str());
        return 2;
    }

    if (run_child("env KUGELAUDIO_BACKEND=definitely-not-a-backend KUGELAUDIO_BACKEND_VERBOSE=1", "fallback", &out) != 0) {
        std::fprintf(stderr, "FAIL: fallback child failed\n%s\n", out.c_str());
        return 3;
    }
    const char* fallback_needles[] = {
        "requested backend=definitely-not-a-backend but no matching device registered — falling back",
        "backend: requested=definitely-not-a-backend device_index=-1 selected=CPU reason=fallback:requested backend unavailable",
        "backend: available_devices=[0]",
    };
    if (!contains_all(out, fallback_needles, 3)) {
        std::fprintf(stderr, "FAIL: fallback scenario missing explicit fallback diagnostics\n%s\n", out.c_str());
        return 4;
    }

    std::printf("Backend selection/logging OK\n");
    return 0;
}
