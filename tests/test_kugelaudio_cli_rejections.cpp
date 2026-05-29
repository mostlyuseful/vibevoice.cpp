#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {

bool file_ok(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int run_expect_fail(const std::string& cmd, const std::string& log_path) {
    const int rc = std::system((cmd + " > \"" + log_path + "\" 2>&1").c_str());
    return rc;
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

    const std::string cli_q = std::string("\"") + cli + "\" --model \"" + model + "\" --tokenizer \"" + tok + "\" ";

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_voice.log";
        const std::string cmd = cli_q + "--voice \"/tmp/fake.voice.gguf\" --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: --voice on KugelAudio should be rejected\n");
            return 1;
        }
        const std::string out = slurp(log);
        if (out.find("--voice has been removed") == std::string::npos ||
            out.find("pre-baked voice.gguf conditioning is no longer supported") == std::string::npos) {
            std::fprintf(stderr, "FAIL: --voice rejection did not explain removal of pre-baked voice support\n%s\n", out.c_str());
            return 2;
        }
    }

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_multiref.log";
        const std::string cmd = cli_q +
            "--ref-audio \"a.wav\" --ref-audio \"b.wav\" --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: multiple --ref-audio on KugelAudio should be rejected\n");
            return 3;
        }
        const std::string out = slurp(log);
        if (out.find("unsupported KugelAudio runtime feature") == std::string::npos ||
            out.find("exactly one raw reference audio input") == std::string::npos) {
            std::fprintf(stderr, "FAIL: multi-ref rejection did not explain KugelAudio v1 shape\n%s\n", out.c_str());
            return 4;
        }
    }

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_speaker.log";
        const std::string cmd = cli_q +
            "--ref-audio \"a.wav\" --text \"Speaker 0: Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: Speaker-tagged text on KugelAudio should be rejected\n");
            return 5;
        }
        const std::string out = slurp(log);
        if (out.find("unsupported KugelAudio runtime feature") == std::string::npos ||
            out.find("plain untagged text") == std::string::npos) {
            std::fprintf(stderr, "FAIL: speaker-tagged rejection did not explain plain-text requirement\n%s\n", out.c_str());
            return 6;
        }
    }

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_overlap.log";
        const std::string cmd = cli_q +
            "--overlap-sentences -1 --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: negative --overlap-sentences should be rejected\n");
            return 7;
        }
        const std::string out = slurp(log);
        if (out.find("--overlap-sentences must be >= 0") == std::string::npos) {
            std::fprintf(stderr, "FAIL: negative overlap rejection missing clear message\n%s\n", out.c_str());
            return 8;
        }
    }

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_chunking_strategy.log";
        const std::string cmd = cli_q +
            "--chunking-strategy definitely-not-valid --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: invalid --chunking-strategy should be rejected\n");
            return 9;
        }
        const std::string out = slurp(log);
        if (out.find("invalid --chunking-strategy") == std::string::npos ||
            out.find("heuristic|syntax-aware") == std::string::npos) {
            std::fprintf(stderr, "FAIL: invalid chunking strategy rejection missing clear message\n%s\n", out.c_str());
            return 10;
        }
    }

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_chunk_continuity.log";
        const std::string cmd = cli_q +
            "--chunk-continuity definitely-not-valid --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: invalid --chunk-continuity should be rejected\n");
            return 11;
        }
        const std::string out = slurp(log);
        if (out.find("invalid --chunk-continuity") == std::string::npos ||
            out.find("none|single-sequence|segmented-state") == std::string::npos) {
            std::fprintf(stderr, "FAIL: invalid chunk continuity rejection missing clear message\n%s\n", out.c_str());
            return 12;
        }
    }

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_tail_reference.log";
        const std::string cmd = cli_q +
            "--chunk-continuity tail-reference --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: retired tail-reference continuity should be rejected\n");
            return 15;
        }
        const std::string out = slurp(log);
        if (out.find("tail-reference is retired") == std::string::npos ||
            out.find("decoded waveform noise") == std::string::npos) {
            std::fprintf(stderr, "FAIL: tail-reference retirement rejection missing clear message\n%s\n", out.c_str());
            return 16;
        }
    }

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_latent_prefix.log";
        const std::string cmd = cli_q +
            "--chunk-continuity latent-prefix --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: retired latent-prefix continuity should be rejected\n");
            return 17;
        }
        const std::string out = slurp(log);
        if (out.find("latent-prefix is retired") == std::string::npos ||
            out.find("worse noise and intonation drift") == std::string::npos) {
            std::fprintf(stderr, "FAIL: latent-prefix retirement rejection missing clear message\n%s\n", out.c_str());
            return 18;
        }
    }

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_clean_tail_reference.log";
        const std::string cmd = cli_q +
            "--chunk-continuity clean-tail-reference --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: retired clean-tail-reference continuity should be rejected\n");
            return 19;
        }
        const std::string out = slurp(log);
        if (out.find("clean-tail-reference is retired") == std::string::npos ||
            out.find("speaker drift") == std::string::npos) {
            std::fprintf(stderr, "FAIL: clean-tail-reference retirement rejection missing clear message\n%s\n", out.c_str());
            return 20;
        }
    }

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_prompt_instruction.log";
        const std::string cmd = cli_q +
            "--chunk-continuity prompt-instruction --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: retired prompt-instruction continuity should be rejected\n");
            return 21;
        }
        const std::string out = slurp(log);
        if (out.find("prompt-instruction is retired") == std::string::npos ||
            out.find("speaker identity drift") == std::string::npos) {
            std::fprintf(stderr, "FAIL: prompt-instruction retirement rejection missing clear message\n%s\n", out.c_str());
            return 22;
        }
    }

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_continuity_tail.log";
        const std::string cmd = cli_q +
            "--continuity-tail-ms -1 --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: negative --continuity-tail-ms should be rejected\n");
            return 13;
        }
        const std::string out = slurp(log);
        if (out.find("--continuity-tail-ms must be >= 0") == std::string::npos) {
            std::fprintf(stderr, "FAIL: negative continuity tail rejection missing clear message\n%s\n", out.c_str());
            return 14;
        }
    }

    std::printf("KugelAudio CLI rejection messaging OK\n");
    return 0;
}
