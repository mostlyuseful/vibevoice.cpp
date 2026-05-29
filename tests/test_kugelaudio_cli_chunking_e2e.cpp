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

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool validate_non_silent_wav(const std::string& wav_path, std::string* error) {
    std::vector<float> samples;
    if (vv::load_wav_24k_mono(wav_path, &samples) != 0) {
        if (error) *error = "generated wav is not loadable";
        return false;
    }
    if (samples.size() < 512) {
        if (error) *error = "generated wav is too short";
        return false;
    }
    double sq = 0.0;
    bool finite = true;
    for (float v : samples) {
        finite = finite && std::isfinite(v);
        sq += static_cast<double>(v) * v;
    }
    if (!finite) {
        if (error) *error = "generated wav contains non-finite samples";
        return false;
    }
    const double rms = std::sqrt(sq / samples.size());
    if (rms < 1e-5) {
        if (error) *error = "generated wav is effectively silent";
        return false;
    }
    return true;
}

int run_cli(const std::string& cli,
            const std::string& model,
            const std::string& tok,
            const std::string& ref,
            const std::string& text,
            const std::string& out_wav,
            const std::string& log_path,
            const std::string& extra_args) {
    const std::string cmd =
        std::string("KUGELAUDIO_BACKEND=cpu \"") + cli + "\" " +
        "--model \"" + model + "\" " +
        "--tokenizer \"" + tok + "\" " +
        "--ref-audio \"" + ref + "\" " +
        "--text \"" + text + "\" " +
        "--out \"" + out_wav + "\" " +
        "--max-frames 24 --steps 4 --cfg 1.0 --seed 12345 " +
        extra_args + " > \"" + log_path + "\" 2>&1";
    return std::system(cmd.c_str());
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

    {
        const std::string out_wav = "/tmp/kugelaudio_cli_chunking_e2e.wav";
        const std::string log_path = "/tmp/kugelaudio_cli_chunking_e2e.log";
        const int rc = run_cli(cli, model, tok, ref,
                               "Alpha beta gamma. Delta epsilon zeta.",
                               out_wav, log_path,
                               "--max-words-per-chunk 3 --overlap-sentences 1 --chunking-strategy syntax-aware --pause-mode punctuation --crossfade-ms 0 --verbose");
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: chunked CLI run rc=%d\n", rc);
            return 1;
        }
        std::string error;
        if (!validate_non_silent_wav(out_wav, &error)) {
            std::fprintf(stderr, "FAIL: chunked CLI output invalid: %s\n", error.c_str());
            return 2;
        }
        const std::string log = slurp(log_path);
        if (log.find("chunking enabled chunks=2") == std::string::npos ||
            log.find("strategy=syntax-aware") == std::string::npos ||
            log.find("overlap_sentences=1") == std::string::npos ||
            log.find("[chunking] generating chunk 1/2") == std::string::npos ||
            log.find("[chunking] generating chunk 2/2") == std::string::npos ||
            log.find("[tts_15b] progress:") == std::string::npos ||
            log.find("chunking stitched 2 chunks") == std::string::npos) {
            std::fprintf(stderr, "FAIL: chunked CLI log markers missing\n%s\n", log.c_str());
            return 3;
        }
        if (log.find("next speech token=") != std::string::npos) {
            std::fprintf(stderr, "FAIL: chunked CLI log still contains noisy token spam\n%s\n", log.c_str());
            return 4;
        }
    }

    {
        const std::string out_wav = "/tmp/kugelaudio_cli_chunking_bypass.wav";
        const std::string log_path = "/tmp/kugelaudio_cli_chunking_bypass.log";
        const int rc = run_cli(cli, model, tok, ref,
                               "Hello world.",
                               out_wav, log_path,
                               "--max-words-per-chunk 20 --overlap-sentences 1 --chunking-strategy syntax-aware --pause-mode punctuation --crossfade-ms 0 --verbose");
        if (rc != 0) {
            std::fprintf(stderr, "FAIL: single-chunk bypass CLI run rc=%d\n", rc);
            return 5;
        }
        std::string error;
        if (!validate_non_silent_wav(out_wav, &error)) {
            std::fprintf(stderr, "FAIL: single-chunk bypass CLI output invalid: %s\n", error.c_str());
            return 6;
        }
        const std::string log = slurp(log_path);
        if (log.find("chunking enabled but skipped: single chunk") == std::string::npos) {
            std::fprintf(stderr, "FAIL: single-chunk bypass log marker missing\n%s\n", log.c_str());
            return 7;
        }
    }

    std::printf("KugelAudio real-weight CLI chunking e2e OK\n");
    return 0;
}
