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

    const std::string ref_wav = "/tmp/kugelaudio_cli_chunking_ref.wav";
    const std::string out_wav = "/tmp/kugelaudio_cli_chunking_out.wav";
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
        std::string("KUGELAUDIO_TEST_STOP_AFTER_CONNECTORS=1 ") +
        "\"" + cli + "\" " +
        "--model \"" + model + "\" " +
        "--tokenizer \"" + tok + "\" " +
        "--ref-audio \"" + ref_wav + "\" " +
        "--text \"Alpha beta. Gamma delta.\" " +
        "--max-words-per-chunk 2 " +
        "--overlap-sentences 1 " +
        "--chunking-strategy syntax-aware " +
        "--pause-mode punctuation " +
        "--crossfade-ms 0 " +
        "--verbose " +
        "--out \"" + out_wav + "\" > /tmp/kugelaudio_cli_chunking.log 2>&1";

    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: kugelaudio-cli returned rc=%d\n", rc);
        return 2;
    }
    if (!file_ok(out_wav)) {
        std::fprintf(stderr, "FAIL: CLI did not produce output wav\n");
        return 3;
    }

    std::vector<float> samples;
    if (vv::load_wav_24k_mono(out_wav, &samples) != 0) {
        std::fprintf(stderr, "FAIL: generated wav is not loadable\n");
        return 4;
    }
    if (samples.size() != 6960) {
        std::fprintf(stderr, "FAIL: chunked hook output size=%zu want 6960\n", samples.size());
        return 5;
    }

    std::ifstream logf("/tmp/kugelaudio_cli_chunking.log");
    std::string log((std::istreambuf_iterator<char>(logf)), std::istreambuf_iterator<char>());
    if (log.find("chunking enabled chunks=3") == std::string::npos ||
        log.find("strategy=syntax-aware") == std::string::npos ||
        log.find("overlap_sentences=1") == std::string::npos ||
        log.find("[chunking] generating chunk 1/3") == std::string::npos ||
        log.find("[chunking] generating chunk 2/3") == std::string::npos ||
        log.find("[chunking] generating chunk 3/3") == std::string::npos) {
        std::fprintf(stderr, "FAIL: chunking log markers missing\n%s\n", log.c_str());
        return 6;
    }

    std::printf("KugelAudio CLI chunking OK\n");
    return 0;
}
