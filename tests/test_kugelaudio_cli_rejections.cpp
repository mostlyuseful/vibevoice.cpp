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
    const char* model = std::getenv("VIBEVOICE_KUGELAUDIO_MODEL");
    const char* tok = std::getenv("VIBEVOICE_KUGELAUDIO_PROMPT_TOKENIZER");
    const char* cli = std::getenv("VIBEVOICE_CLI");
    if (!model || !tok || !cli || !file_ok(model) || !file_ok(tok) || !file_ok(cli)) {
        std::fprintf(stderr, "skip: set VIBEVOICE_KUGELAUDIO_MODEL, VIBEVOICE_KUGELAUDIO_PROMPT_TOKENIZER, and VIBEVOICE_CLI\n");
        return 77;
    }

    const std::string cli_q = std::string("\"") + cli + "\" tts --model \"" + model + "\" --tokenizer \"" + tok + "\" ";

    {
        const std::string log = "/tmp/kugelaudio_cli_reject_voice.log";
        const std::string cmd = cli_q + "--voice \"/tmp/fake.voice.gguf\" --text \"Hello world.\"";
        if (run_expect_fail(cmd, log) == 0) {
            std::fprintf(stderr, "FAIL: --voice on KugelAudio should be rejected\n");
            return 1;
        }
        const std::string out = slurp(log);
        if (out.find("unsupported KugelAudio runtime feature") == std::string::npos ||
            out.find("pre-baked voice gguf conditioning") == std::string::npos) {
            std::fprintf(stderr, "FAIL: --voice rejection did not explain unsupported KugelAudio feature\n%s\n", out.c_str());
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

    std::printf("KugelAudio CLI rejection messaging OK\n");
    return 0;
}
