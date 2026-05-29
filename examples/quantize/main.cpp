// kugelaudio-quantize — selective tensor quantization tool.
//
// Reads a GGUF model, quantizes matmul-heavy tensors to target dtypes via
// ggml_quantize_chunk, leaves everything else as source dtype, and writes a new gguf.
// The rewrite is streamed tensor-by-tensor: metadata is built first, then source
// payload is read and destination payload is written incrementally.
//
// Why selective: ggml_mul_mat handles quantized weights natively; our conv1d
// wrapper inline-casts kernels to fp16 (it doesn't dequantize on the fly), so
// quantizing conv/tensorwise-norm paths silently breaks outputs.
//
// Why this in addition to the python script: gguf-py only implements
// Q4_0 / Q5_0 / Q8_0 / BF16 in pure python. K-quants (Q4_K, Q5_K, Q6_K)
// need libggml linkage — that's what this tool provides.
//
// Usage:
//   kugelaudio-quantize --src in.gguf --out out.gguf --type q4_k_m

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <regex>
#include <string>
#include <vector>

namespace {

constexpr size_t STREAM_COPY_CHUNK = 1u << 20; // 1 MB

struct QuantSpec { const char* name; ggml_type type; };
const QuantSpec kKnown[] = {
    {"q4_0",   GGML_TYPE_Q4_0},
    {"q4_1",   GGML_TYPE_Q4_1},
    {"q5_0",   GGML_TYPE_Q5_0},
    {"q5_1",   GGML_TYPE_Q5_1},
    {"q8_0",   GGML_TYPE_Q8_0},
    {"q2_k",   GGML_TYPE_Q2_K},
    {"q3_k",   GGML_TYPE_Q3_K},
    {"q4_k",   GGML_TYPE_Q4_K},
    {"q5_k",   GGML_TYPE_Q5_K},
    {"q6_k",   GGML_TYPE_Q6_K},
    {"f16",    GGML_TYPE_F16},
    {"f32",    GGML_TYPE_F32},
    {"bf16",   GGML_TYPE_BF16},
};

struct FamilyConfig {
    ggml_type type = GGML_TYPE_F16;
    bool      enabled = false;
};

ggml_type parse_type(const std::string& s) {
    for (const auto& q : kKnown) {
        if (s == q.name) return q.type;
    }
    return GGML_TYPE_COUNT;
}

const std::vector<std::regex>& attn_patterns() {
    static const std::vector<std::regex> v = {
        std::regex(R"(^(lm|tlm)\.blk\.\d+\.attn_[qkvo]\.weight$)"),
    };
    return v;
}

const std::vector<std::regex>& ffn_patterns() {
    static const std::vector<std::regex> v = {
        std::regex(R"(^(lm|tlm)\.blk\.\d+\.ffn_(gate|up|down)\.weight$)"),
    };
    return v;
}

const std::vector<std::regex>& at_block_ffn_patterns() {
    static const std::vector<std::regex> v = {
        std::regex(R"(^(at\.(enc|dec))\.stage_\d+_block_\d+\.weight\.ffn_linear[12]$)"),
    };
    return v;
}

const std::vector<std::regex>& st_block_ffn_patterns() {
    static const std::vector<std::regex> v = {
        std::regex(R"(^st\.enc\.stage_\d+_block_\d+\.weight\.ffn_linear[12]$)"),
    };
    return v;
}

const std::vector<std::regex>& ac_connector_patterns() {
    static const std::vector<std::regex> v = {
        std::regex(R"(^ac\.fc[12]\.weight$)"),
    };
    return v;
}

const std::vector<std::regex>& sc_connector_patterns() {
    static const std::vector<std::regex> v = {
        std::regex(R"(^sc\.fc[12]\.weight$)"),
    };
    return v;
}

const std::vector<std::regex>& dh_patterns() {
    static const std::vector<std::regex> v = {
        std::regex(R"(^dh\.(noisy_proj|cond_proj|t_embed_lin1|t_embed_lin2|final\.proj|final\.adaln)$)"),
        std::regex(R"(^dh\.layer_\d+\.(norm|adaln|ffn_(gate|up|down))$)"),
    };
    return v;
}

bool matches_any(const std::string& s, const std::vector<std::regex>& pats) {
    for (const auto& re : pats) {
        if (std::regex_match(s, re)) return true;
    }
    return false;
}

bool should_quantize(const std::string& name,
                    const FamilyConfig& lm_attn,
                    const FamilyConfig& lm_ffn,
                    const FamilyConfig& lm_head,
                    const FamilyConfig& lm_emb,
                    const FamilyConfig& ac_conn,
                    const FamilyConfig& sc_conn,
                    const FamilyConfig& dh,
                    const FamilyConfig& at_block_ffn,
                    const FamilyConfig& st_block_ffn,
                    ggml_type* out_type) {
    if (matches_any(name, attn_patterns())) {
        *out_type = lm_attn.type;
        return lm_attn.enabled;
    }
    if (matches_any(name, ffn_patterns())) {
        *out_type = lm_ffn.type;
        return lm_ffn.enabled;
    }
    if (name == "lm_head.weight") {
        *out_type = lm_head.type;
        return lm_head.enabled;
    }
    if (name == "lm.tok_embd.weight") {
        *out_type = lm_emb.type;
        return lm_emb.enabled;
    }
    if (matches_any(name, ac_connector_patterns())) {
        *out_type = ac_conn.type;
        return ac_conn.enabled;
    }
    if (matches_any(name, sc_connector_patterns())) {
        *out_type = sc_conn.type;
        return sc_conn.enabled;
    }
    if (matches_any(name, dh_patterns())) {
        *out_type = dh.type;
        return dh.enabled;
    }
    if (matches_any(name, at_block_ffn_patterns()) || matches_any(name, st_block_ffn_patterns())) {
        // The same quant-family rule applies to at.enc/stages.* and at.dec/stages.*
        // block FFN linears, which are pure matmuls (safe to quantize).
        *out_type = (matches_any(name, at_block_ffn_patterns()) ? at_block_ffn.type : st_block_ffn.type);
        return matches_any(name, at_block_ffn_patterns()) ? at_block_ffn.enabled : st_block_ffn.enabled;
    }
    *out_type = GGML_TYPE_COUNT;
    return false;
}

bool read_at_offset(FILE* f, uint64_t offset, void* dst, size_t nbytes) {
    if (!f) return false;
    if (::fseeko(f, static_cast<off_t>(offset), SEEK_SET) != 0) {
        return false;
    }
    return std::fread(dst, 1, nbytes, f) == nbytes;
}

bool write_zero_fill(FILE* f, size_t nbytes) {
    if (nbytes == 0) return true;
    const size_t buf_size = 4096;
    static const uint8_t kZero[buf_size] = {};
    while (nbytes > 0) {
        const size_t n = std::min(nbytes, buf_size);
        if (std::fwrite(kZero, 1, n, f) != n) {
            return false;
        }
        nbytes -= n;
    }
    return true;
}

bool seek_or_pad(FILE* f, uint64_t expected_offset) {
    const auto cur = ::ftello(f);
    if (cur < 0) {
        return false;
    }
    const uint64_t cur_u = static_cast<uint64_t>(cur);
    if (cur_u == expected_offset) {
        return true;
    }
    if (cur_u > expected_offset) {
        return false;
    }
    return write_zero_fill(f, expected_offset - cur_u);
}

bool read_tensor_as_f32(FILE* src_fp,
                        uint64_t off,
                        const ggml_tensor* st,
                        std::vector<uint8_t>& src_bytes,
                        std::vector<float>& out_f32) {
    const size_t n_elems = ggml_nelements(st);
    const size_t nbytes = ggml_nbytes(st);
    out_f32.resize(n_elems);
    src_bytes.resize(nbytes);

    if (n_elems == 0) {
        return true;
    }

    if (!read_at_offset(src_fp, off, src_bytes.data(), nbytes)) {
        return false;
    }

    if (st->type == GGML_TYPE_F32) {
        std::memcpy(out_f32.data(), src_bytes.data(), nbytes);
        return true;
    }

    if (st->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = reinterpret_cast<const ggml_fp16_t*>(src_bytes.data());
        for (size_t i = 0; i < n_elems; ++i) {
            out_f32[i] = ggml_fp16_to_fp32(src[i]);
        }
        return true;
    }

    if (st->type == GGML_TYPE_BF16) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(src_bytes.data());
        for (size_t i = 0; i < n_elems; ++i) {
            const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            out_f32[i] = f;
        }
        return true;
    }

    std::fprintf(stderr,
        "to_f32: unsupported source dtype %d for tensor '%s'\n",
        static_cast<int>(st->type), st->name);
    return false;
}

bool write_tensor_data_passthrough(FILE* src_fp,
                                  uint64_t src_off,
                                  FILE* dst_fp,
                                  size_t nbytes) {
    if (nbytes == 0) return true;

    std::vector<uint8_t> buf(STREAM_COPY_CHUNK);
    if (::fseeko(src_fp, static_cast<off_t>(src_off), SEEK_SET) != 0) {
        return false;
    }

    uint64_t remain = nbytes;
    while (remain > 0) {
        const size_t n = static_cast<size_t>(std::min<uint64_t>(STREAM_COPY_CHUNK, remain));
        if (std::fread(buf.data(), 1, n, src_fp) != n) {
            return false;
        }
        if (std::fwrite(buf.data(), 1, n, dst_fp) != n) {
            return false;
        }
        remain -= n;
    }
    return true;
}

double bytes_to_gib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

void report_progress(int64_t index,
                     int64_t total,
                     const char* name,
                     bool quantized,
                     size_t bytes_done,
                     size_t bytes_total) {
    const double pct = bytes_total == 0 ? 100.0 :
        100.0 * static_cast<double>(bytes_done) / static_cast<double>(bytes_total);
    std::fprintf(stderr,
        "progress: %lld/%lld tensors %5.1f%% mode=%s tensor=%s done=%.2f/%.2f GiB\n",
        static_cast<long long>(index),
        static_cast<long long>(total),
        pct,
        quantized ? "quant" : "copy",
        name,
        bytes_to_gib(bytes_done),
        bytes_to_gib(bytes_total));
}

}  // namespace

int main(int argc, char** argv) {
    std::string src, out, type_str = "q4_k";
    std::string attn_type_str, ffn_type_str, lm_head_type_str;
    std::string embed_type_str, ac_conn_type_str, sc_conn_type_str;
    std::string dh_type_str, at_block_ffn_type_str, st_block_ffn_type_str;
    bool include_embed = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--src"             && i + 1 < argc) src                 = argv[++i];
        else if (a == "--out"             && i + 1 < argc) out                 = argv[++i];
        else if (a == "--type"            && i + 1 < argc) type_str            = argv[++i];
        else if (a == "--attn-type"       && i + 1 < argc) attn_type_str       = argv[++i];
        else if (a == "--ffn-type"        && i + 1 < argc) ffn_type_str        = argv[++i];
        else if (a == "--lm-head-type"    && i + 1 < argc) lm_head_type_str    = argv[++i];
        else if (a == "--embed-type"      && i + 1 < argc) embed_type_str      = argv[++i];
        else if (a == "--ac-connector-type" && i + 1 < argc) ac_conn_type_str  = argv[++i];
        else if (a == "--sc-connector-type" && i + 1 < argc) sc_conn_type_str  = argv[++i];
        else if (a == "--dh-type"         && i + 1 < argc) dh_type_str         = argv[++i];
        else if (a == "--at-block-ffn-type"&& i + 1 < argc) at_block_ffn_type_str = argv[++i];
        else if (a == "--st-block-ffn-type"&& i + 1 < argc) st_block_ffn_type_str = argv[++i];
        else if (a == "--include-embed")                      include_embed       = true;
        else if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
                "usage: %s --src in.gguf --out out.gguf --type <type>\n"
                "  --type one of q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k f16 f32 bf16\n"
                "  --attn-type <type>      override quant type for attention weights\n"
                "                          (lm/tlm.blk.*.attn_[qkvo].weight) only.\n"
                "  --ffn-type <type>       override quant type for FFN weights\n"
                "                          (lm/tlm.blk.*.ffn_[gate|up|down].weight) only.\n"
                "  --lm-head-type <type>   override quant type for lm_head.weight only.\n"
                "  --embed-type <type>     override quant type for lm.tok_embd.weight.\n"
                "                          (legacy flag: --include-embed quantizes at --type)\n"
                "  --ac-connector-type <type>   quantize acoustic connector\n"
                "                               ac.fc1.weight and ac.fc2.weight (matmul paths).\n"
                "  --sc-connector-type <type>   quantize semantic connector\n"
                "                               sc.fc1.weight and sc.fc2.weight (matmul paths).\n"
                "  --dh-type <type>         quantize diffusion head matrices\n"
                "                               dh.noisy_proj/cond_proj/t_embed_lin1/t_embed_lin2\n"
                "                               dh.final.{proj,adaln}/dh.layer_*.{norm,adaln,ffn_*}.\n"
                "  --at-block-ffn-type <type>   quantize at.enc/at.dec stage FFN linears\n"
                "                               *.stage_*_block_*_weight.ffn_linear1/2\n"
                "  --st-block-ffn-type <type>   quantize st.enc stage FFN linears\n"
                "                               st.enc.stage_*_block_*_weight.ffn_linear1/2\n"
                "\n"
                "Only explicitly enabled families are quantized; unspecified families keep\n"
                "source dtype (typically fp16).\n",
                argv[0]);
            return 0;
        }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1;
        }
    }

    if (src.empty() || out.empty()) {
        std::fprintf(stderr, "--src and --out are required\n"); return 1;
    }
    if (src == out) {
        std::fprintf(stderr, "--src and --out must be different files\n"); return 1;
    }

    const ggml_type target = parse_type(type_str);
    if (target == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "unknown --type: %s\n", type_str.c_str()); return 1;
    }

    auto parse_override = [&](const std::string& s, const char* flag, ggml_type fallback) -> FamilyConfig {
        if (s.empty()) {
            return {fallback, false};
        }
        const ggml_type t = parse_type(s);
        if (t == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "unknown %s: %s\n", flag, s.c_str());
            std::exit(1);
        }
        return {t, true};
    };

    FamilyConfig lm_attn = parse_override(attn_type_str, "--attn-type", target);
    FamilyConfig lm_ffn  = parse_override(ffn_type_str, "--ffn-type", target);
    FamilyConfig lm_head = parse_override(lm_head_type_str, "--lm-head-type", target);
    FamilyConfig lm_emb  = parse_override(embed_type_str, "--embed-type", target);
    FamilyConfig ac_conn = parse_override(ac_conn_type_str, "--ac-connector-type", target);
    FamilyConfig sc_conn = parse_override(sc_conn_type_str, "--sc-connector-type", target);
    FamilyConfig dh      = parse_override(dh_type_str, "--dh-type", target);
    FamilyConfig at_ffn  = parse_override(at_block_ffn_type_str, "--at-block-ffn-type", target);
    FamilyConfig st_ffn  = parse_override(st_block_ffn_type_str, "--st-block-ffn-type", target);

    // Preserve old compatibility knob: --include-embed quantizes lm.tok_embd.weight
    // with the base --type, unless --embed-type is explicitly set.
    if (include_embed && embed_type_str.empty()) lm_emb = {target, true};
    // Always keep default core LM-path families quantized to `--type` unless overridden.
    if (!lm_attn.enabled) lm_attn = {target, true};
    if (!lm_ffn.enabled)  lm_ffn  = {target, true};
    if (!lm_head.enabled) lm_head = {target, true};

    const auto reject_if_imatrix = [](bool enabled, ggml_type t, const char* flag) {
        if (!enabled) return false;
        if (ggml_quantize_requires_imatrix(t)) {
            for (const auto& q : kKnown) {
                if (q.type == t) {
                    std::fprintf(stderr,
                        "type %s for %s requires an importance matrix (imatrix) — not supported here\n",
                        q.name, flag);
                    break;
                }
            }
            return true;
        }
        return false;
    };

    if (reject_if_imatrix(lm_attn.enabled, lm_attn.type, "--attn-type") ||
        reject_if_imatrix(lm_ffn.enabled,  lm_ffn.type,  "--ffn-type") ||
        reject_if_imatrix(lm_head.enabled, lm_head.type, "--lm-head-type") ||
        reject_if_imatrix(lm_emb.enabled,  lm_emb.type,  "--embed-type") ||
        reject_if_imatrix(ac_conn.enabled, ac_conn.type, "--ac-connector-type") ||
        reject_if_imatrix(sc_conn.enabled, sc_conn.type, "--sc-connector-type") ||
        reject_if_imatrix(dh.enabled,      dh.type,      "--dh-type") ||
        reject_if_imatrix(at_ffn.enabled,  at_ffn.type,  "--at-block-ffn-type") ||
        reject_if_imatrix(st_ffn.enabled,  st_ffn.type,  "--st-block-ffn-type") ) {
        return 1;
    }

    // 1) Build source metadata only (no tensor payload) and destination metadata.
    struct ggml_context* src_ctx = nullptr;
    struct gguf_init_params p {};
    p.no_alloc = true;
    p.ctx      = &src_ctx;
    struct gguf_context* src_gguf = gguf_init_from_file(src.c_str(), p);
    if (!src_gguf) {
        std::fprintf(stderr, "failed to read %s\n", src.c_str()); return 2;
    }

    const int64_t n = gguf_get_n_tensors(src_gguf);
    std::printf("loaded %s: %lld tensors\n", src.c_str(), static_cast<long long>(n));

    struct gguf_context* dst_gguf = gguf_init_empty();
    gguf_set_kv(dst_gguf, src_gguf);
    const int64_t kugel_arch_key = gguf_find_key(src_gguf, "kugelaudio.architecture");
    if (kugel_arch_key >= 0) {
        const char* kugel_arch = gguf_get_val_str(src_gguf, kugel_arch_key);
        if (kugel_arch && std::strcmp(kugel_arch, "kugelaudio") == 0) {
            gguf_set_val_str(dst_gguf, "general.architecture", "kugelaudio");
        }
    }

    const size_t mem = ggml_tensor_overhead() * (n + 1) + 16ull * 1024 * 1024;
    struct ggml_init_params ip {};
    ip.mem_size = mem;
    ip.no_alloc = true;
    struct ggml_context* dst_ctx = ggml_init(ip);
    if (!dst_ctx) {
        std::fprintf(stderr, "ggml_init(dst) failed\n");
        gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx);
        return 2;
    }

    // Determine target metadata and quantization plan without materializing tensor data.
    size_t bytes_in = 0;
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(src_gguf, i);
        if (!name) continue;
        ggml_tensor* st = ggml_get_tensor(src_ctx, name);
        if (!st) continue;

        const size_t in_size = ggml_nbytes(st);
        bytes_in += in_size;

        const std::string sname = name;
        ggml_type tensor_target = target;
        bool do_quant = should_quantize(sname,
                                       lm_attn, lm_ffn, lm_head, lm_emb,
                                       ac_conn, sc_conn, dh,
                                       at_ffn, st_ffn,
                                       &tensor_target);
        if (tensor_target == GGML_TYPE_COUNT) tensor_target = target;

        const int blk = ggml_blck_size(tensor_target);
        const int64_t n_per_row = st->ne[0];
        const bool can_quant = (n_per_row > 0) && (n_per_row % blk == 0);
        const bool do_quant_final = do_quant && can_quant;
        if (do_quant && !can_quant) {
            std::fprintf(stderr,
                "skip %s [ne0=%lld] — row not divisible by block size %d\n",
                name, static_cast<long long>(n_per_row), blk);
        }

        ggml_type out_type = do_quant_final ? tensor_target : st->type;

        ggml_tensor* dt = ggml_new_tensor(dst_ctx, out_type, ggml_n_dims(st), st->ne);
        ggml_set_name(dt, name);
        gguf_add_tensor(dst_gguf, dt);
    }

    // 2) Serialize only metadata first. This keeps memory bounded and then
    //    fills the tensor data section in-order, one tensor at a time.
    if (!gguf_write_to_file(dst_gguf, out.c_str(), true)) {
        std::fprintf(stderr, "failed to write metadata to %s\n", out.c_str());
        gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx); ggml_free(dst_ctx);
        return 5;
    }

    FILE* src_fp = std::fopen(src.c_str(), "rb");
    if (!src_fp) {
        std::fprintf(stderr, "failed to reopen source file %s\n", src.c_str());
        gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx); ggml_free(dst_ctx);
        return 2;
    }

    FILE* out_fp = std::fopen(out.c_str(), "rb+");
    if (!out_fp) {
        std::fprintf(stderr, "failed to open output for streaming append %s\n", out.c_str());
        std::fclose(src_fp);
        gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx); ggml_free(dst_ctx);
        return 2;
    }

    if (std::fseek(out_fp, 0, SEEK_END) != 0) {
        std::fprintf(stderr, "failed to seek output end\n");
        std::fclose(src_fp);
        std::fclose(out_fp);
        gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx); ggml_free(dst_ctx);
        return 2;
    }

    const uint64_t dst_data_base = static_cast<uint64_t>(gguf_get_meta_size(dst_gguf));
    const uint64_t src_data_base = static_cast<uint64_t>(gguf_get_data_offset(src_gguf));
    const uint64_t end_of_meta = static_cast<uint64_t>(::ftello(out_fp));
    if (end_of_meta != dst_data_base && !seek_or_pad(out_fp, dst_data_base)) {
        std::fprintf(stderr, "failed to position output at data section start\n");
        std::fclose(src_fp);
        std::fclose(out_fp);
        gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx); ggml_free(dst_ctx);
        return 5;
    }

    size_t bytes_out = 0;
    size_t bytes_done = 0;
    size_t n_quant = 0;
    std::vector<uint8_t> tmp_raw;
    std::vector<float>  tmp_f32;
    std::vector<uint8_t> tmp_out;

    std::fprintf(stderr,
        "progress: starting streamed quantization for %lld tensors (%.2f GiB input)\n",
        static_cast<long long>(n), bytes_to_gib(bytes_in));

    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(src_gguf, i);
        if (!name) continue;
        ggml_tensor* st = ggml_get_tensor(src_ctx, name);
        if (!st) continue;

        const size_t in_size = ggml_nbytes(st);
        const uint64_t src_off = src_data_base + gguf_get_tensor_offset(src_gguf, i);
        const uint64_t dst_off = dst_data_base + gguf_get_tensor_offset(dst_gguf, i);

        const std::string sname = name;
        ggml_type tensor_target = target;
        bool do_quant = should_quantize(sname,
                                       lm_attn, lm_ffn, lm_head, lm_emb,
                                       ac_conn, sc_conn, dh,
                                       at_ffn, st_ffn,
                                       &tensor_target);
        if (tensor_target == GGML_TYPE_COUNT) tensor_target = target;

        const int blk = ggml_blck_size(tensor_target);
        const int64_t n_per_row = st->ne[0];
        const int64_t nrows = (n_per_row == 0) ? 0 : (static_cast<int64_t>(ggml_nelements(st)) / n_per_row);
        const bool can_quant = (n_per_row > 0) && (n_per_row % blk == 0);
        const bool do_quant_final = do_quant && can_quant;
        const size_t out_size = do_quant_final
                                 ? ggml_row_size(tensor_target, n_per_row) * nrows
                                 : in_size;

        if (!seek_or_pad(out_fp, dst_off)) {
            std::fprintf(stderr, "failed to seek/pad to tensor %s offset\n", name);
            std::fclose(src_fp);
            std::fclose(out_fp);
            gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx); ggml_free(dst_ctx);
            return 5;
        }

        if (do_quant_final) {
            if (!read_tensor_as_f32(src_fp, src_off, st, tmp_raw, tmp_f32)) {
                std::fprintf(stderr, "failed to read source tensor data for %s\n", name);
                std::fclose(src_fp);
                std::fclose(out_fp);
                gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx); ggml_free(dst_ctx);
                return 3;
            }

            tmp_out.assign(out_size, 0);
            ggml_quantize_init(tensor_target);
            const size_t produced = ggml_quantize_chunk(
                tensor_target, tmp_f32.data(), tmp_out.data(),
                /*start=*/0, nrows, n_per_row, /*imatrix=*/nullptr);
            if (produced != out_size) {
                std::fprintf(stderr,
                    "ggml_quantize_chunk(%s, %s): produced %zu, expected %zu\n",
                    name, ggml_type_name(tensor_target), produced, out_size);
                std::fclose(src_fp);
                std::fclose(out_fp);
                ggml_quantize_free();
                gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx); ggml_free(dst_ctx);
                return 4;
            }
            ++n_quant;

            if (!tmp_out.empty() && std::fwrite(tmp_out.data(), 1, out_size, out_fp) != out_size) {
                std::fprintf(stderr, "failed writing quantized tensor %s\n", name);
                std::fclose(src_fp);
                std::fclose(out_fp);
                ggml_quantize_free();
                gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx); ggml_free(dst_ctx);
                return 5;
            }
        } else {
            if (!write_tensor_data_passthrough(src_fp, src_off, out_fp, in_size)) {
                std::fclose(src_fp);
                std::fclose(out_fp);
                gguf_free(src_gguf); gguf_free(dst_gguf); ggml_free(src_ctx); ggml_free(dst_ctx);
                return 5;
            }
        }

        bytes_out += out_size;
        bytes_done += in_size;
        report_progress(i + 1, n, name, do_quant_final, bytes_done, bytes_in);
    }

    std::fclose(src_fp);
    std::fclose(out_fp);

    const double saved_gb = (static_cast<double>(bytes_in) - bytes_out) / (1024.0*1024.0*1024.0);
    std::printf("wrote %s: quantized %zu tensors → %s, saved %.2f GB (%.1f%%)\n",
                out.c_str(), n_quant, type_str.c_str(), saved_gb,
                bytes_in == 0 ? 0.0 : 100.0 * (1.0 - static_cast<double>(bytes_out) / bytes_in));

    ggml_quantize_free();
    gguf_free(dst_gguf);
    gguf_free(src_gguf);
    ggml_free(dst_ctx);
    ggml_free(src_ctx);
    return 0;
}
