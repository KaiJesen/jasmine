/**
 * GPT-2 inference timing, built to be compared against PyTorch.
 *
 * Separate from bench_jasmine / bench_bpe because it needs an exported weight
 * file, so it cannot run on a clean checkout.
 *
 * Two things are measured, because they are not the same computation and a fair
 * comparison has to line them up with the right PyTorch call:
 *
 *   1. full forward over T tokens. jasmine `forward(seq)` vs `model(ids)` in
 *      torch: one pass, no cache, identical shapes.
 *   2. the generation path: prefill T tokens then decode N steps one token at a
 *      time. jasmine `prefill(seq)` + `forward_one`, i.e. it steps the prompt
 *      through the cache token by token, whereas torch's `use_cache=True` prefill
 *      is a single batched forward. That difference is real and is reported as
 *      such rather than hidden, so `--full-forward` is timed separately.
 *
 * Fairness knobs, all set from outside so both sides can be pinned identically:
 *   - precision: --dtype float (default) or double. torch runs fp32, so fp32 is
 *     the default; running jasmine in double against torch in float says nothing
 *     about the implementation.
 *   - threads: read from OMP_NUM_THREADS, shared by OpenMP and the BLAS slice
 *     loop inside jas_mat_gemm.hpp. Pair it with torch.set_num_threads(N).
 *   - weights: the same exported file feeds both sides.
 *
 * Usage:
 *   OMP_NUM_THREADS=4 ./build/benches/bench_gpt2_infer build/distilgpt2_weights.bin
 *   ... --prefill 128 --decode 64 --repeats 5 --dtype float --csv out.csv
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "jas_gpt2_t.hpp"
#include "jas_updator_t.hpp"
#include "jas_weight_io.hpp"

using namespace jasmine;

namespace
{

using clock_type = std::chrono::steady_clock;

// Same shape as examples/gpt2_generate.cpp: gpt2_model_t takes the updator as a
// template template parameter, so it needs an alias template rather than a
// concrete type.
template <typename val_type>
using gpt2_upr_tpl = cache_updator_t<val_type, nadam_t>;

template <typename val_type>
using gpt2_infer_model_t = gpt2_model_t<mat_t<val_type>, gpt2_upr_tpl>;

double ms_since(clock_type::time_point start)
{
    const auto end = clock_type::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/** Median of the samples, which is robust to the occasional scheduler hiccup. */
double median(std::vector<double> samples)
{
    if (samples.empty())
        return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

struct opts_t
{
    std::string weights;
    std::string csv;
    int prefill = 128;
    int decode = 64;
    int repeats = 5;
    int warmup = 2;
    bool dtype_float = true; // torch's default, so the fair default
};

/**
 * Runs the whole measurement for one precision. Templated because val_type is a
 * compile-time choice in jas_mat_t, so float and double need separate code.
 */
template <typename val_type>
int run(const opts_t& opt)
{
    using model_type = gpt2_infer_model_t<val_type>;

    weight_file_t wf;
    wf.load(opt.weights);
    const auto cfg = read_gpt2_config(wf);

    model_type model(cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_ff, cfg.vocab, cfg.n_pos);
    load_gpt2(model, wf);
    model.reserve_kv_cache(cfg.n_pos);

    const int T = std::min(opt.prefill, cfg.n_pos);
    // Deterministic token ids that stay inside the vocabulary: a fixed pattern is
    // enough, because this measures shapes and memory traffic, not content.
    mat_t<val_type> ids(1, T);
    for (int t = 0; t < T; ++t)
        ids(0, t) = static_cast<val_type>(100 + (t * 7919) % 40000);
    mat_t<val_type> next_id(1, 1, {static_cast<val_type>(50256 - 1)});
    const int next_pos = T; // where decode continues from, in absolute positions

    // --- warmup: also forces lazy allocation (KV cache, BLAS workspaces) ---
    mat_t<val_type> logits;
    for (int i = 0; i < opt.warmup; ++i)
    {
        logits = model.forward(ids);
        model.prefill(ids);
        model.clear_kv_cache();
        model.prefill(ids);
        for (int s = 0; s < opt.decode; ++s)
            logits = model.forward_one(next_id, next_pos + s);
        model.clear_kv_cache();
    }

    // --- 1. full forward over T tokens, no cache ---
    std::vector<double> full;
    for (int r = 0; r < opt.repeats; ++r)
    {
        const auto t0 = clock_type::now();
        logits = model.forward(ids);
        full.push_back(ms_since(t0));
    }

    // --- 2a. prefill the cache (one forward_one per prompt token) ---
    std::vector<double> prefill;
    for (int r = 0; r < opt.repeats; ++r)
    {
        const auto t0 = clock_type::now();
        model.prefill(ids);
        prefill.push_back(ms_since(t0));
    }

    // --- 2b. decode N single tokens against a warm cache ---
    // Measured per step so it can be compared with torch's per-step cost.
    std::vector<double> decode;
    for (int r = 0; r < opt.repeats; ++r)
    {
        model.clear_kv_cache();
        model.prefill(ids);
        const auto t0 = clock_type::now();
        for (int s = 0; s < opt.decode; ++s)
            logits = model.forward_one(next_id, next_pos + s);
        decode.push_back(ms_since(t0) / static_cast<double>(opt.decode));
    }

    // A checksum of the final logits, so the Python side can prove it ran the
    // same computation before any timing is believed.
    double sum = 0.0;
    double first = 0.0;
    {
        const mat_t<val_type> last = model.forward(ids);
        const int rows = last.row_num();
        const int cols = last.col_num();
        first = static_cast<double>(last(0, cols - 1));
        for (int v = 0; v < rows; ++v)
            sum += static_cast<double>(last(v, cols - 1));
    }

    const std::string dtype_name = opt.dtype_float ? "float" : "double";
    const double full_ms = median(full);
    const double prefill_ms = median(prefill);
    const double decode_ms = median(decode);

    std::printf("dtype=%s prefill_tokens=%d decode_steps=%d repeats=%d\n",
                dtype_name.c_str(), T, opt.decode, opt.repeats);
    std::printf("full_forward_ms=%.4f\n", full_ms);
    std::printf("prefill_ms=%.4f\n", prefill_ms);
    std::printf("decode_ms_per_token=%.4f\n", decode_ms);
    std::printf("prefill_plus_decode_ms=%.4f\n", prefill_ms + decode_ms * opt.decode);
    std::printf("logits_last_col_sum=%.10g\n", sum);
    std::printf("logits_first=%.10g\n", first);

    if (!opt.csv.empty())
    {
        std::ofstream out(opt.csv, std::ios::app);
        const bool empty = !out.tellp();
        if (empty)
            out << "impl,dtype,prefill_tokens,decode_steps,full_forward_ms,prefill_ms,"
                   "decode_ms_per_token,threads\n";
        const char* threads = std::getenv("OMP_NUM_THREADS");
        out << "jasmine," << dtype_name << "," << T << "," << opt.decode << ","
            << full_ms << "," << prefill_ms << "," << decode_ms << ","
            << (threads ? threads : "unset") << "\n";
    }
    return 0;
}

void usage(const char* argv0)
{
    std::fprintf(stderr,
                 "usage: %s <weights.bin> [--dtype float|double] [--prefill T] "
                 "[--decode N] [--repeats R] [--warmup W] [--csv FILE]\n",
                 argv0);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        usage(argv[0]);
        return 2;
    }
    opts_t opt;
    opt.weights = argv[1];

    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto next_arg = [&]() -> std::string {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--dtype")
        {
            const std::string v = next_arg();
            if (v == "float")
                opt.dtype_float = true;
            else if (v == "double")
                opt.dtype_float = false;
            else
            {
                std::fprintf(stderr, "--dtype must be float or double\n");
                return 2;
            }
        }
        else if (a == "--prefill") opt.prefill = std::stoi(next_arg());
        else if (a == "--decode")  opt.decode = std::stoi(next_arg());
        else if (a == "--repeats") opt.repeats = std::stoi(next_arg());
        else if (a == "--warmup")  opt.warmup = std::stoi(next_arg());
        else if (a == "--csv")     opt.csv = next_arg();
        else
        {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (opt.repeats < 1 || opt.decode < 1 || opt.prefill < 1)
    {
        std::fprintf(stderr, "--prefill/--decode/--repeats must be positive\n");
        return 2;
    }

    try
    {
        return opt.dtype_float ? run<float>(opt) : run<double>(opt);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "bench_gpt2_infer: %s\n", e.what());
        return 1;
    }
}
