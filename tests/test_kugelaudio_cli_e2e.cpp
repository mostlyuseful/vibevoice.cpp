#include "audio_io.hpp"

#include <cmath>
#include <cstdint>
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

uint16_t read_le16(const std::vector<unsigned char>& b, size_t off) {
    return static_cast<uint16_t>(b[off]) |
           (static_cast<uint16_t>(b[off + 1]) << 8);
}

uint32_t read_le32(const std::vector<unsigned char>& b, size_t off) {
    return static_cast<uint32_t>(b[off]) |
           (static_cast<uint32_t>(b[off + 1]) << 8) |
           (static_cast<uint32_t>(b[off + 2]) << 16) |
           (static_cast<uint32_t>(b[off + 3]) << 24);
}

bool validate_wav_header(const std::string& path,
                         uint32_t expected_sample_rate,
                         uint16_t expected_channels,
                         std::string* error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (error) *error = "failed to open wav";
        return false;
    }
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
    if (b.size() < 44) {
        if (error) *error = "wav too small";
        return false;
    }
    if (std::string(reinterpret_cast<const char*>(b.data()), 4) != "RIFF" ||
        std::string(reinterpret_cast<const char*>(b.data() + 8), 4) != "WAVE") {
        if (error) *error = "missing RIFF/WAVE signature";
        return false;
    }
    size_t fmt_off = std::string::npos;
    size_t data_off = std::string::npos;
    size_t off = 12;
    while (off + 8 <= b.size()) {
        const std::string chunk(reinterpret_cast<const char*>(b.data() + off), 4);
        const uint32_t chunk_size = read_le32(b, off + 4);
        if (chunk == "fmt ") fmt_off = off;
        if (chunk == "data") { data_off = off; break; }
        off += 8 + chunk_size + (chunk_size & 1u);
    }
    if (fmt_off == std::string::npos) {
        if (error) *error = "missing fmt chunk";
        return false;
    }
    if (data_off == std::string::npos) {
        if (error) *error = "missing data chunk";
        return false;
    }
    if (fmt_off + 24 > b.size()) {
        if (error) *error = "truncated fmt chunk";
        return false;
    }
    const uint16_t audio_format = read_le16(b, fmt_off + 8);
    const uint16_t channels = read_le16(b, fmt_off + 10);
    const uint32_t sample_rate = read_le32(b, fmt_off + 12);
    const uint16_t bits_per_sample = read_le16(b, fmt_off + 22);
    const uint32_t data_size = read_le32(b, data_off + 4);
    if (audio_format != 1) {
        if (error) *error = "expected PCM format";
        return false;
    }
    if (channels != expected_channels) {
        if (error) *error = "unexpected channel count";
        return false;
    }
    if (sample_rate != expected_sample_rate) {
        if (error) *error = "unexpected sample rate";
        return false;
    }
    if (bits_per_sample != 16) {
        if (error) *error = "expected 16-bit PCM";
        return false;
    }
    if (data_size == 0) {
        if (error) *error = "empty data chunk";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const char* model = std::getenv("KUGELAUDIO_MODEL");
    const char* tok   = std::getenv("KUGELAUDIO_TOKENIZER");
    const char* ref   = std::getenv("KUGELAUDIO_REF_WAV");
    const char* cli   = std::getenv("KUGELAUDIO_CLI");
    if (!model || !tok || !ref || !cli ||
        !file_ok(model) || !file_ok(tok) || !file_ok(ref) || !file_ok(cli)) {
        std::fprintf(stderr,
                     "skip: set KUGELAUDIO_MODEL, KUGELAUDIO_TOKENIZER, KUGELAUDIO_REF_WAV, and KUGELAUDIO_CLI\n");
        return 77;
    }

    const std::string out_wav = "/tmp/kugelaudio_cli_e2e.wav";
    const std::string log_path = "/tmp/kugelaudio_cli_e2e.log";
    std::string cmd =
        std::string("\"") + cli + "\" " +
        "--model \"" + model + "\" " +
        "--tokenizer \"" + tok + "\" " +
        "--ref-audio \"" + ref + "\" " +
        "--text \"Hello world.\" " +
        "--out \"" + out_wav + "\" " +
        "--max-frames 32 --steps 8 --cfg 1.0 --seed 12345 --verbose > \"" + log_path + "\" 2>&1";

    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: kugelaudio-cli returned rc=%d\n", rc);
        return 1;
    }
    if (!file_ok(out_wav)) {
        std::fprintf(stderr, "FAIL: CLI did not produce output wav\n");
        return 2;
    }

    std::string wav_error;
    if (!validate_wav_header(out_wav, /*expected_sample_rate=*/24000,
                             /*expected_channels=*/1, &wav_error)) {
        std::fprintf(stderr, "FAIL: generated wav header invalid: %s\n", wav_error.c_str());
        return 3;
    }

    std::vector<float> samples;
    if (vv::load_wav_24k_mono(out_wav, &samples) != 0) {
        std::fprintf(stderr, "FAIL: generated wav is not loadable via load_wav_24k_mono\n");
        return 4;
    }
    if (samples.empty()) {
        std::fprintf(stderr, "FAIL: generated wav is empty\n");
        return 5;
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
        return 6;
    }
    if (rms < 1e-5) {
        std::fprintf(stderr, "FAIL: generated wav is effectively silent (rms=%.8f)\n", rms);
        return 7;
    }

    std::ifstream logf(log_path);
    std::string log((std::istreambuf_iterator<char>(logf)), std::istreambuf_iterator<char>());
    if (log.find("tts: wrote ") == std::string::npos) {
        std::fprintf(stderr, "FAIL: CLI log does not show successful write\n");
        return 8;
    }
    if (log.find("[tts_15b] progress:") == std::string::npos ||
        log.find("frame 1/32") == std::string::npos ||
        log.find("%") == std::string::npos) {
        std::fprintf(stderr, "FAIL: CLI log does not show frame-based progress\n%s\n", log.c_str());
        return 9;
    }
    if (log.find("next speech token=") != std::string::npos) {
        std::fprintf(stderr, "FAIL: CLI log still contains noisy token spam\n%s\n", log.c_str());
        return 10;
    }

    return 0;
}
