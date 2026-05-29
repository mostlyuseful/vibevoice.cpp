// Roundtrip test: synthesize a sine, save WAV, reload, verify samples.
// Also exercises the linear resampler.

#include "audio_io.hpp"
#include "vibevoice.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr float kPi = 3.14159265358979323846f;

bool roughly_equal(float a, float b, float tol) {
    return std::fabs(a - b) <= tol;
}
}  // namespace

int main() {
    const int   sr   = 16000;
    const float freq = 440.0f;
    const int   n    = sr;  // 1 second
    std::vector<float> sine(n);
    for (int i = 0; i < n; ++i) {
        sine[i] = 0.5f * std::sin(2.0f * kPi * freq * i / sr);
    }

    // Resample 16k -> 24k -> 16k roundtrip; correlation should be high.
    auto up   = vv::resample_linear(sine, sr, 24000);
    if (up.empty()) { std::fprintf(stderr, "resample up empty\n"); return 1; }
    auto down = vv::resample_linear(up,   24000, sr);
    if (down.size() < sine.size() - 4 || down.size() > sine.size() + 4) {
        std::fprintf(stderr, "roundtrip length mismatch: %zu vs %zu\n", down.size(), sine.size());
        return 2;
    }

    // Pearson-style correlation (centered) on the first 0.5 s.
    const size_t k = std::min<size_t>(sr / 2, down.size());
    double mu_a = 0, mu_b = 0;
    for (size_t i = 0; i < k; ++i) { mu_a += sine[i]; mu_b += down[i]; }
    mu_a /= k; mu_b /= k;
    double num = 0, da = 0, db = 0;
    for (size_t i = 0; i < k; ++i) {
        double xa = sine[i] - mu_a;
        double xb = down[i] - mu_b;
        num += xa * xb; da += xa * xa; db += xb * xb;
    }
    double corr = num / std::sqrt(da * db + 1e-12);
    if (corr < 0.95) {
        std::fprintf(stderr, "roundtrip correlation too low: %f\n", corr);
        return 3;
    }

    // WAV roundtrip.
    const std::string path = "/tmp/vibevoice_audio_io_test.wav";
    vv_audio out{};
    out.samples     = sine.data();
    out.n_samples   = sine.size();
    out.sample_rate = sr;
    out.channels    = 1;
    if (vv_save_wav(path.c_str(), &out) != VV_OK) {
        std::fprintf(stderr, "save_wav failed\n");
        return 4;
    }

    vv_audio loaded{};
    if (vv_load_wav(path.c_str(), &loaded) != VV_OK) {
        std::fprintf(stderr, "load_wav failed\n");
        return 5;
    }
    if (loaded.sample_rate != sr) {
        std::fprintf(stderr, "rate mismatch: %d vs %d\n", loaded.sample_rate, sr);
        vv_audio_free(&loaded);
        return 6;
    }
    if (loaded.n_samples != sine.size()) {
        std::fprintf(stderr, "length mismatch: %zu vs %zu\n", loaded.n_samples, sine.size());
        vv_audio_free(&loaded);
        return 7;
    }
    // 16-bit PCM quantization: max-abs error <= ~3e-5 expected.
    float max_err = 0;
    for (size_t i = 0; i < loaded.n_samples; ++i) {
        max_err = std::max(max_err, std::fabs(loaded.samples[i] - sine[i]));
    }
    vv_audio_free(&loaded);
    if (max_err > 1e-3f) {
        std::fprintf(stderr, "wav roundtrip max-err too high: %f\n", max_err);
        return 8;
    }

    // Runtime reference-audio path should accept non-24k, multi-channel WAVs
    // and convert them internally to 24 kHz mono.
    const std::string stereo_path = "/tmp/vibevoice_audio_io_stereo_16k.wav";
    std::vector<float> stereo(static_cast<size_t>(n) * 2);
    for (int i = 0; i < n; ++i) {
        stereo[2 * i + 0] = sine[i];
        stereo[2 * i + 1] = -sine[i];
    }
    vv_audio stereo_out{};
    stereo_out.samples = stereo.data();
    stereo_out.n_samples = sine.size();
    stereo_out.sample_rate = sr;
    stereo_out.channels = 2;
    if (vv_save_wav(stereo_path.c_str(), &stereo_out) != VV_OK) {
        std::fprintf(stderr, "save_wav stereo failed\n");
        return 9;
    }

    std::vector<float> ref_24k_mono;
    if (vv::load_wav_24k_mono(stereo_path, &ref_24k_mono) != VV_OK) {
        std::fprintf(stderr, "load_wav_24k_mono failed\n");
        return 10;
    }
    if (ref_24k_mono.size() < 23990 || ref_24k_mono.size() > 24010) {
        std::fprintf(stderr, "24k mono length mismatch: %zu\n", ref_24k_mono.size());
        return 11;
    }
    float max_abs = 0.0f;
    for (float v : ref_24k_mono) max_abs = std::max(max_abs, std::fabs(v));
    if (max_abs > 1e-3f) {
        std::fprintf(stderr, "stereo downmix expected near-silence after averaging channels, got max_abs=%f\n", max_abs);
        return 12;
    }

    // Canonical reference-audio normalization: target -25 dBFS, then avoid clipping.
    std::vector<float> quiet(24000, 0.01f);
    vv::normalize_dbfs(&quiet);
    double sq_norm = 0.0;
    for (float v : quiet) sq_norm += static_cast<double>(v) * v;
    const double rms_norm = std::sqrt(sq_norm / quiet.size());
    const double target_lin = std::pow(10.0, -25.0 / 20.0);
    if (std::fabs(rms_norm - target_lin) > 1e-3) {
        std::fprintf(stderr, "normalize_dbfs target mismatch: rms=%f want=%f\n", rms_norm, target_lin);
        return 13;
    }

    std::vector<float> loud(24000, 2.0f);
    vv::normalize_dbfs(&loud);
    float loud_max_abs = 0.0f;
    for (float v : loud) loud_max_abs = std::max(loud_max_abs, std::fabs(v));
    if (loud_max_abs > 1.0f + 1e-5f) {
        std::fprintf(stderr, "normalize_dbfs clipping guard failed: max_abs=%f\n", loud_max_abs);
        return 14;
    }

    std::printf("audio_io ok: roundtrip corr=%.4f, wav max-err=%.6f, ref_24k_mono=%zu samples, normalized_rms=%.6f\n",
                corr, max_err, ref_24k_mono.size(), rms_norm);
    return 0;
}
