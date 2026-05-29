#include "audio_io.hpp"
#include "vibevoice.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

bool file_ok(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

}  // namespace

int main() {
    const char* model = std::getenv("KUGELAUDIO_MODEL");
    const char* tok = std::getenv("KUGELAUDIO_TOKENIZER");
    const char* cli = std::getenv("KUGELAUDIO_CLI");
    if (!model || !tok || !cli || !file_ok(model) || !file_ok(tok) || !file_ok(cli)) {
        std::fprintf(stderr, "skip: set KUGELAUDIO_MODEL, KUGELAUDIO_TOKENIZER, and KUGELAUDIO_CLI\n");
        return 77;
    }

    const std::string ref_wav = "/tmp/kugelaudio_cli_ref.wav";
    const std::string out_wav = "/tmp/kugelaudio_cli_out.wav";
    const int sr = 16000;
    std::vector<float> stereo(static_cast<size_t>(sr) * 2);
    for (int i = 0; i < sr; ++i) {
        stereo[2 * i + 0] = 0.1f;
        stereo[2 * i + 1] = -0.1f;
    }
    vv_audio a{};
    a.samples = stereo.data();
    a.n_samples = sr;
    a.sample_rate = sr;
    a.channels = 2;
    if (vv_save_wav(ref_wav.c_str(), &a) != VV_OK) {
        std::fprintf(stderr, "FAIL: failed to write reference wav fixture\n");
        return 1;
    }

    std::string cmd =
        std::string("KUGELAUDIO_TEST_STOP_AFTER_CONDITIONING=1 ") +
        "\"" + cli + "\" " +
        "--model \"" + model + "\" " +
        "--tokenizer \"" + tok + "\" " +
        "--ref-audio \"" + ref_wav + "\" " +
        "--text \"Hello world.\" " +
        "--out \"" + out_wav + "\" > /tmp/kugelaudio_cli_conditioning.log 2>&1";

    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: kugelaudio-cli returned rc=%d\n", rc);
        return 2;
    }
    if (!file_ok(out_wav)) {
        std::fprintf(stderr, "FAIL: CLI did not produce output wav\n");
        return 3;
    }

    std::ifstream logf("/tmp/kugelaudio_cli_conditioning.log");
    std::string log((std::istreambuf_iterator<char>(logf)), std::istreambuf_iterator<char>());
    if (log.find("test hook stopping after preprocessing+conditioning encoders") == std::string::npos) {
        std::fprintf(stderr, "FAIL: CLI log does not show preprocessing+conditioning path was reached\n");
        return 4;
    }

    std::printf("KugelAudio CLI conditioning wiring OK\n");
    return 0;
}
