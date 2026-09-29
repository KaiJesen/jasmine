#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "jas_bpe_t.hpp"

// BPE tokenizer benchmarks.
//
// This is a separate target from bench_jasmine on purpose:
//   - it needs an external vocabulary (JASMINE_GPT2_TOKENIZER_DIR), so it cannot
//     run in CI or on a clean checkout;
//   - it prints a one-shot chunk-length report before timing anything, which
//     needs a main() of its own because bench_jasmine takes its main from
//     benchmark_main.
// Keeping it separate leaves bench_jasmine runnable with no model files present.
//
// Usage:
//     JASMINE_GPT2_TOKENIZER_DIR=<dir with vocab.json + merges.txt> ./bench_bpe

namespace {

using jasmine::bpe_t;

/** Directory holding `vocab.json` and `merges.txt`; same variable as the tests. */
constexpr const char* kTokenizerDirEnv = "JASMINE_GPT2_TOKENIZER_DIR";

std::string g_dir;

/**
 * The tokenizer, loaded once on first use. Vocabulary parsing takes a few
 * milliseconds, and every benchmark here measures encode()/decode() only, so it
 * must not be inside any timed region.
 */
bpe_t& tokenizer()
{
    static bpe_t instance = [] {
        bpe_t tok;
        tok.load(g_dir, {"<|endoftext|>"});
        return tok;
    }();
    return instance;
}

/**
 * A deterministic corpus used for the chunk-length report.
 *
 * Built in code rather than read from files so the histogram is reproducible on
 * any machine. The mix matters more than the text itself: prose, code, CJK,
 * symbol runs, whitespace runs, and two pathological single-chunk inputs (a long
 * digit run and a long letter run) that are the reason the merge loop's worst
 * case is worth measuring at all.
 */
std::string synthetic_corpus()
{
    static const char* kWords[] = {
        "the",    "of",     "and",   "to",     "in",     "a",       "is",     "that",
        "for",    "it",     "as",    "was",    "with",   "be",      "by",     "on",
        "not",    "he",     "this",  "are",    "or",     "have",    "from",   "at",
        "token",  "merge",  "byte",  "rank",   "greedy", "encoding","sequence","model",
        "input",  "output", "layer", "attention", "position", "cache", "batch", "gradient",
    };
    constexpr std::size_t kWordCount = std::size(kWords);

    std::mt19937 rng(20260928);
    const auto pick_word = [&]() -> const char* { return kWords[rng() % kWordCount]; };

    std::string out;

    // Prose: mostly short chunks, the case the rescan loop is designed for.
    for (int i = 0; i < 12000; ++i)
    {
        out += pick_word();
        out += (i % 11 == 10) ? ". " : " ";
    }

    // Code: indentation runs, punctuation, identifiers.
    for (int i = 0; i < 400; ++i)
    {
        out += "    const std::vector<int> ids = tokenizer.encode(text, ";
        out += pick_word();
        out += ");\n";
    }

    // Symbol runs: never attach to a word, and merge among themselves.
    for (int i = 0; i < 300; ++i)
        out += "===+++---...///***###$$$%%%!!!\n";

    // CJK: 3 bytes per code point, no spaces, so each run is one long-ish chunk.
    for (int i = 0; i < 400; ++i)
        out += "\u4F60\u597D\uFF0C\u4E16\u754C\u3002\u6A21\u578B\u5206\u8BCD";

    // Whitespace runs: exercise the `\s+(?!\S)` backtracking path.
    for (int i = 0; i < 300; ++i)
        out += "a                    b \t\t\t c\n";

    // Pathological: one chunk each. These are what the timing benchmarks below
    // measure the scaling of.
    out += std::string(10000, '9');
    out += "\n";
    out += std::string(10000, 'a');
    out += "\n";

    return out;
}

/**
 * Prints the chunk-length distribution: the input to the "is the heap needed?" question.
 *
 * Count-based percentiles are not enough here. The merge loop is O(k^2), so a single
 * 10000-long chunk costs as much as a million 10-long ones: with a handful of outliers
 * among tens of thousands of chunks they never show up in p99.9, yet they dominate the
 * runtime. The report therefore also sums k^2 per bucket and lists the largest chunks,
 * which is the shape of the actual cost.
 */
void report_chunk_lengths(const std::string& corpus)
{
    const std::vector<std::string_view> chunks = bpe_t::pre_tokenize(corpus);

    std::vector<std::size_t> lens;
    lens.reserve(chunks.size());
    for (const std::string_view chunk : chunks)
        lens.push_back(chunk.size());
    if (lens.empty())
        return;
    std::sort(lens.begin(), lens.end());

    const auto pct = [&](double p) {
        return lens[static_cast<std::size_t>(static_cast<double>(lens.size() - 1) * p)];
    };

    std::size_t total = 0;
    std::size_t huge = 0;
    double work = 0.0;      // sum of len^2, proportional to the merge loop's work
    double work_huge = 0.0;
    for (const std::size_t len : lens)
    {
        const double cost = static_cast<double>(len) * static_cast<double>(len);
        total += len;
        work += cost;
        if (len > 1000)
        {
            ++huge;
            work_huge += cost;
        }
    }

    std::printf("\n=== pre-tokenizer chunk lengths ===\n");
    std::printf("corpus bytes : %zu\n", total);
    std::printf("chunks       : %zu   mean %.2f   max %zu\n", lens.size(),
                static_cast<double>(total) / static_cast<double>(lens.size()), lens.back());
    std::printf("lengths      : p50 %.0f   p90 %.0f   p99 %.0f\n",
                static_cast<double>(pct(0.50)), static_cast<double>(pct(0.90)),
                static_cast<double>(pct(0.99)));

    std::printf("longest 5    :");
    for (std::size_t i = 0; i < 5 && i < lens.size(); ++i)
        std::printf(" %zu", lens[lens.size() - 1 - i]);
    std::printf("\n");

    std::printf("chunks > 1000: %zu  (%.2f%% of chunks, %.1f%% of the merge work)\n", huge,
                100.0 * static_cast<double>(huge) / static_cast<double>(lens.size()),
                work > 0.0 ? 100.0 * work_huge / work : 0.0);
    std::printf("(the rescan merge loop is O(chunk^2), so read the work share, not the "
                "chunk share)\n\n");
}

} // namespace

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/** Reference case: normal prose, thousands of short chunks. */
static void BM_EncodeProse(benchmark::State& state)
{
    static const std::string text = [] {
        std::string out;
        for (int i = 0; i < 200; ++i)
            out += "The quick brown fox jumps over the lazy dog. ";
        return out;
    }();

    for (auto _ : state)
        benchmark::DoNotOptimize(tokenizer().encode(text));

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(text.size()));
}

/** CJK: 3-byte code points, no spaces, so chunks are longer and denser. */
static void BM_EncodeCjk(benchmark::State& state)
{
    static const std::string text = [] {
        std::string out;
        for (int i = 0; i < 200; ++i)
            out += "\u4F60\u597D\uFF0C\u4E16\u754C\u3002\u6A21\u578B\u5206\u8BCD";
        return out;
    }();

    for (auto _ : state)
        benchmark::DoNotOptimize(tokenizer().encode(text));

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(text.size()));
}

/**
 * One single chunk of length N, which is the worst case for the rescan merge
 * loop. Items/second falls roughly 4x per doubling while the merge loop stays
 * quadratic; the point of the benchmark is to make that visible, and to give a
 * number for deciding whether the doubly-linked-list + heap version is needed.
 */
static void BM_EncodeSingleChunkDigits(benchmark::State& state)
{
    const auto len = static_cast<std::size_t>(state.range(0));
    const std::string text(len, '9');

    for (auto _ : state)
        benchmark::DoNotOptimize(tokenizer().encode(text));

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(len));
}

/** Same shape, but letters: the merges that apply differ from the digit case. */
static void BM_EncodeSingleChunkLetters(benchmark::State& state)
{
    const auto len = static_cast<std::size_t>(state.range(0));
    const std::string text(len, 'a');

    for (auto _ : state)
        benchmark::DoNotOptimize(tokenizer().encode(text));

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(len));
}

/** Decode is linear in the token count; kept as the contrast to the merge loop. */
static void BM_Decode(benchmark::State& state)
{
    static const std::vector<int> ids = [] {
        std::string text;
        for (int i = 0; i < 200; ++i)
            text += "The quick brown fox jumps over the lazy dog. ";
        return tokenizer().encode(text);
    }();

    for (auto _ : state)
        benchmark::DoNotOptimize(tokenizer().decode(ids));

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(ids.size()));
}

BENCHMARK(BM_EncodeProse)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_EncodeCjk)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Decode)->Unit(benchmark::kMicrosecond);

/**
 * The two merge implementations head to head on the same single chunk. This is
 * what bpe_t's automatic threshold is calibrated against: below the crossover the
 * rescan wins because it allocates nothing, above it the quadratic term loses to
 * the linked list.
 */
static void merge_strategy_case(benchmark::State& state, jasmine::merge_strategy strategy)
{
    const auto len = static_cast<std::size_t>(state.range(0));
    const std::vector<int> base(len, tokenizer().symbol_of_byte('9'));

    for (auto _ : state)
    {
        std::vector<int> syms = base; // copying is identical for both strategies
        tokenizer().merge_symbols(syms, strategy);
        benchmark::DoNotOptimize(syms);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(len));
}

static void BM_MergeRescan(benchmark::State& state)
{
    merge_strategy_case(state, jasmine::merge_strategy::rescan);
}

static void BM_MergeLinkedHeap(benchmark::State& state)
{
    merge_strategy_case(state, jasmine::merge_strategy::linked_heap);
}

BENCHMARK(BM_MergeRescan)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_MergeLinkedHeap)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond);
// Up to 8000 keeps a full run under a minute; the quadratic trend is already
// unambiguous by then.
BENCHMARK(BM_EncodeSingleChunkDigits)
    ->Arg(250)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Arg(4000)
    ->Arg(8000)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_EncodeSingleChunkLetters)
    ->Arg(250)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Arg(4000)
    ->Arg(8000)
    ->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);

    const char* dir = std::getenv(kTokenizerDirEnv);
    if (dir == nullptr || *dir == '\0')
    {
        // Not an error: this is the expected state on a clean checkout.
        std::fprintf(stderr,
                     "bench_bpe: %s is not set; skipping.\n"
                     "           Point it at a directory containing vocab.json and "
                     "merges.txt\n"
                     "           (e.g. a distilgpt2 snapshot).\n",
                     kTokenizerDirEnv);
        return 0;
    }
    g_dir = dir;

    try
    {
        report_chunk_lengths(synthetic_corpus());
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "bench_bpe: chunk report failed: %s\n", e.what());
        return 1;
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
