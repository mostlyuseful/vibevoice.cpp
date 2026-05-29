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
        "--text \"Alpha beta gamma. Delta epsilon zeta.\" " +
        "--out \"" + out_wav + "\" " +
        "--max-frames 24 --steps 4 --cfg 1.0 --seed 12345 " +
        "--max-words-per-chunk 3 --overlap-sentences 1 --chunking-strategy syntax-aware " +
        "--pause-mode punctuation --crossfade-ms 0 --verbose > \"" + log_path + "\" 2>&1";
    return std::system(cmd.c_str());
}

int run_cli_f16_policy_probe(const std::string& cli,
                             const std::string& model,
                             const std::string& tok,
                             const std::string& ref,
                             const std::string& out_wav,
                             const std::string& log_path) {
    const std::string model_alias = "/tmp/kugelaudio_cli_policy_f16.gguf";
    const std::string link_cmd = "ln -sf \"" + model + "\" \"" + model_alias + "\"";
    if (std::system(link_cmd.c_str()) != 0) return 1;
    const std::string cmd =
        std::string("KUGELAUDIO_BACKEND=cpu \"") + cli + "\" " +
        "--model \"" + model_alias + "\" " +
        "--tokenizer \"" + tok + "\" " +
        "--ref-audio \"" + ref + "\" " +
        "--text \"Alpha beta gamma. Delta epsilon zeta.\" " +
        "--out \"" + out_wav + "\" " +
        "--max-frames 2 --steps 1 --cfg 1.0 --seed 12345 " +
        "--chunk-continuity single-sequence --max-words-per-chunk 3 --overlap-sentences 1 " +
        "--final-decoder-backend stream --verbose > \"" + log_path + "\" 2>&1";
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

    const std::string out1 = "/tmp/kugelaudio_cli_chunking_seed_1.wav";
    const std::string out2 = "/tmp/kugelaudio_cli_chunking_seed_2.wav";
    const std::string log1 = "/tmp/kugelaudio_cli_chunking_seed_1.log";
    const std::string log2 = "/tmp/kugelaudio_cli_chunking_seed_2.log";

    int rc = run_cli(cli, model, tok, ref, out1, log1);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: first chunked seeded CLI run rc=%d\n", rc);
        return 1;
    }
    rc = run_cli(cli, model, tok, ref, out2, log2);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: second chunked seeded CLI run rc=%d\n", rc);
        return 2;
    }
    if (!file_ok(out1) || !file_ok(out2)) {
        std::fprintf(stderr, "FAIL: chunked seeded CLI runs did not both produce WAVs\n");
        return 3;
    }

    const auto a = read_bytes(out1);
    const auto b = read_bytes(out2);
    if (a.empty() || b.empty()) {
        std::fprintf(stderr, "FAIL: chunked seeded CLI outputs were empty\n");
        return 4;
    }
    if (a != b) {
        std::fprintf(stderr,
                     "FAIL: identical chunked CLI seeds/settings did not produce identical WAV bytes (%zu vs %zu bytes)\n",
                     a.size(), b.size());
        return 5;
    }

    const std::string log = slurp(log1);
    if (log.find("chunking enabled chunks=3") == std::string::npos ||
        log.find("strategy=syntax-aware") == std::string::npos ||
        log.find("overlap_sentences=1") == std::string::npos ||
        log.find("budget overlap_words=") == std::string::npos ||
        log.find("[tts_15b] progress:") == std::string::npos ||
        log.find("chunking stitched 3 chunks") == std::string::npos) {
        std::fprintf(stderr, "FAIL: chunked seeded CLI log markers missing\n%s\n", log.c_str());
        return 6;
    }
    if (log.find("next speech token=") != std::string::npos) {
        std::fprintf(stderr, "FAIL: chunked seeded CLI log still contains noisy token spam\n%s\n", log.c_str());
        return 7;
    }

    const std::string policy_out = "/tmp/kugelaudio_cli_policy_f16.wav";
    const std::string policy_log = "/tmp/kugelaudio_cli_policy_f16.log";
    rc = run_cli_f16_policy_probe(cli, model, tok, ref, policy_out, policy_log);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL: f16 policy probe CLI run rc=%d\n", rc);
        return 8;
    }
    const std::string policy = slurp(policy_log);
    if (policy.find("quantization_hint=f16") == std::string::npos ||
        policy.find("step_embed_f16=on") == std::string::npos ||
        policy.find("rounding generated step embeddings to f16 before LM feedback") == std::string::npos) {
        std::fprintf(stderr, "FAIL: f16 policy probe did not enable/log step embedding f16 feedback\n%s\n", policy.c_str());
        return 9;
    }
    if (policy.find("min_ratio=0.000") == std::string::npos) {
        std::fprintf(stderr, "FAIL: single-sequence default is not natural-stop min_ratio=0\n%s\n", policy.c_str());
        return 10;
    }

    std::printf("KugelAudio chunked CLI seed regression OK: identical seeded WAV bytes (%zu bytes)\n", a.size());
    return 0;
}
