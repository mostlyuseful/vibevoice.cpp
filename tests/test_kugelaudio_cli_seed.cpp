#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

bool file_ok(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

std::vector<char> read_bytes(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int run_cli(const std::string& cli,
            const std::string& model,
            const std::string& tok,
            const std::string& ref,
            const std::string& out_wav,
            const std::string& log_path) {
    const std::string cmd =
        std::string("KUGELAUDIO_BACKEND=cpu \"") + cli + "\" " +
        "--model \"" + model + "\" " +
        "--tokenizer \"" + tok + "\" " +
        "--ref-audio \"" + ref + "\" " +
        "--text \"Hello world.\" " +
        "--out \"" + out_wav + "\" " +
        "--max-frames 32 --steps 8 --cfg 1.0 --seed 12345 --verbose > \"" + log_path + "\" 2>&1";
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

    const std::string out1 = "/tmp/kugelaudio_cli_seed_1.wav";
    const std::string out2 = "/tmp/kugelaudio_cli_seed_2.wav";
    const std::string log1 = "/tmp/kugelaudio_cli_seed_1.log";
    const std::string log2 = "/tmp/kugelaudio_cli_seed_2.log";

    int rc = run_cli(cli, model, tok, ref, out1, log1);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: first seeded CLI run rc=%d\n", rc);
        return 1;
    }
    rc = run_cli(cli, model, tok, ref, out2, log2);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: second seeded CLI run rc=%d\n", rc);
        return 2;
    }
    if (!file_ok(out1) || !file_ok(out2)) {
        std::fprintf(stderr, "FAIL: seeded CLI runs did not both produce WAVs\n");
        return 3;
    }

    const auto a = read_bytes(out1);
    const auto b = read_bytes(out2);
    if (a.empty() || b.empty()) {
        std::fprintf(stderr, "FAIL: seeded CLI outputs were empty\n");
        return 4;
    }
    if (a != b) {
        std::fprintf(stderr,
                     "FAIL: identical CLI seeds/settings did not produce identical WAV bytes (%zu vs %zu bytes)\n",
                     a.size(), b.size());
        return 5;
    }

    const std::string log = slurp(log1);
    if (log.find("[tts_15b] progress:") == std::string::npos ||
        log.find("frame 1/32") == std::string::npos) {
        std::fprintf(stderr, "FAIL: seeded CLI log missing frame-based progress\n%s\n", log.c_str());
        return 6;
    }
    if (log.find("next speech token=") != std::string::npos) {
        std::fprintf(stderr, "FAIL: seeded CLI log still contains noisy token spam\n%s\n", log.c_str());
        return 7;
    }

    std::printf("KugelAudio CLI seed plumbing OK: identical seeded WAV bytes (%zu bytes)\n", a.size());
    return 0;
}
