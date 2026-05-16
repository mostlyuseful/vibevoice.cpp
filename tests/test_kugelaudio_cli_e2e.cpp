#include "audio_io.hpp"

#include <cmath>
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
    const char* model = std::getenv("VIBEVOICE_KUGELAUDIO_MODEL");
    const char* tok   = std::getenv("VIBEVOICE_TOKENIZER");
    const char* ref   = std::getenv("VIBEVOICE_REF_WAV");
    const char* cli   = std::getenv("VIBEVOICE_CLI");
    if (!model || !tok || !ref || !cli ||
        !file_ok(model) || !file_ok(tok) || !file_ok(ref) || !file_ok(cli)) {
        std::fprintf(stderr,
                     "skip: set VIBEVOICE_KUGELAUDIO_MODEL, VIBEVOICE_TOKENIZER, VIBEVOICE_REF_WAV, and VIBEVOICE_CLI\n");
        return 77;
    }

    const std::string out_wav = "/tmp/kugelaudio_cli_e2e.wav";
    const std::string log_path = "/tmp/kugelaudio_cli_e2e.log";
    std::string cmd =
        std::string("\"") + cli + "\" tts " +
        "--model \"" + model + "\" " +
        "--tokenizer \"" + tok + "\" " +
        "--ref-audio \"" + ref + "\" " +
        "--text \"Hello world.\" " +
        "--out \"" + out_wav + "\" " +
        "--max-frames 32 --steps 8 --cfg 1.0 --seed 12345 > \"" + log_path + "\" 2>&1";

    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: vibevoice-cli tts returned rc=%d\n", rc);
        return 1;
    }
    if (!file_ok(out_wav)) {
        std::fprintf(stderr, "FAIL: CLI did not produce output wav\n");
        return 2;
    }

    std::vector<float> samples;
    if (vv::load_wav_24k_mono(out_wav, &samples) != 0) {
        std::fprintf(stderr, "FAIL: generated wav is not loadable via load_wav_24k_mono\n");
        return 3;
    }
    if (samples.empty()) {
        std::fprintf(stderr, "FAIL: generated wav is empty\n");
        return 4;
    }
    double sq = 0.0;
    bool finite = true;
    for (float v : samples) {
        finite = finite && std::isfinite(v);
        sq += static_cast<double>(v) * v;
    }
    const double rms = std::sqrt(sq / samples.size());
    std::printf("KugelAudio CLI e2e OK: %zu samples rms=%.6f\n", samples.size(), rms);
    if (!finite) {
        std::fprintf(stderr, "FAIL: generated wav contains non-finite samples\n");
        return 5;
    }
    if (rms < 1e-5) {
        std::fprintf(stderr, "FAIL: generated wav is effectively silent (rms=%.8f)\n", rms);
        return 6;
    }

    std::ifstream logf(log_path);
    std::string log((std::istreambuf_iterator<char>(logf)), std::istreambuf_iterator<char>());
    if (log.find("tts: wrote ") == std::string::npos) {
        std::fprintf(stderr, "FAIL: CLI log does not show successful write\n");
        return 7;
    }

    return 0;
}
