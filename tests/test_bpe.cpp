#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "jas_bpe_t.hpp"

using namespace jasmine;

namespace {

/** UTF-8 encoding of one code point; the tests build vocabulary symbols with it. */
std::string utf8_of(unsigned int cp)
{
    std::string out;
    if (cp < 0x80)
    {
        out.push_back(static_cast<char>(cp));
    }
    else if (cp < 0x800)
    {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else if (cp < 0x10000)
    {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else
    {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

/**
 * Escapes a symbol for vocab.json. Only '"' and '\' need it here, and both are
 * members of the byte alphabet (they are printable, so they map to themselves),
 * which means the reader's escape handling is exercised by every load.
 */
std::string json_escape(const std::string& s)
{
    std::string out;
    for (const char c : s)
    {
        if (c == '"' || c == '\\')
            out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

/**
 * Writes a complete, self-contained byte-level tokenizer:
 *   - the 256 single-byte symbols, with id == byte value;
 *   - five merged symbols at ids 256..259 and 261;
 *   - one special token at id 260.
 *
 * The merge rules are chosen so that rank order and scan order disagree, and so
 * that a rule starting with '#' is present:
 *     rank 0: "#"  + "#"  -> "##"      (id 261)
 *     rank 1: "a"  + "b"  -> "ab"      (id 256)
 *     rank 2: "\u0120" + "a"  -> "\u0120a"  (id 257)
 *     rank 3: "ab" + "c"  -> "abc"     (id 258)
 *     rank 4: "\u0120a" + "b" -> "\u0120ab" (id 259)
 *
 * Encoding " ab" therefore has to pick rank 1 over rank 2 and produce
 * [space, "ab"], not ["\u0120a", "b"]. The trailing '#' rule guards the other
 * direction: HF's merges.txt has real rules whose left symbol is '#', and
 * treating every '#'-prefixed line as a comment silently drops them and shifts
 * the rank of everything after.
 */
std::string write_toy_tokenizer()
{
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "jasmine_test_bpe";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const std::string space = utf8_of(0x0120); // byte-level symbol for ' '

    // std::map keeps the output deterministic, which matters when diffing failures.
    std::map<std::string, int> vocab;
    const auto& alphabet = bpe_t::byte_to_unicode();
    for (unsigned int byte = 0; byte < 256; ++byte)
        vocab.emplace(utf8_of(alphabet[byte]), static_cast<int>(byte));
    vocab.emplace("ab", 256);
    vocab.emplace(space + "a", 257);
    vocab.emplace("abc", 258);
    vocab.emplace(space + "ab", 259);
    vocab.emplace("<|endoftext|>", 260);
    vocab.emplace("##", 261);

    {
        std::ofstream out(dir / "vocab.json", std::ios::binary);
        out << "{";
        bool first = true;
        for (const auto& entry : vocab)
        {
            if (!first)
                out << ",";
            first = false;
            out << "\"" << json_escape(entry.first) << "\":" << entry.second;
        }
        out << "}";
    }
    {
        std::ofstream out(dir / "merges.txt", std::ios::binary);
        out << "#version: 0.2\n";
        out << "# #\n";
        out << "a b\n";
        out << space << " a\n";
        out << "ab c\n";
        out << space << "a b\n";
    }
    return dir.string();
}

std::vector<std::string> as_strings(const std::vector<std::string_view>& views)
{
    std::vector<std::string> out;
    out.reserve(views.size());
    for (const std::string_view view : views)
        out.emplace_back(view);
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Byte alphabet
// ---------------------------------------------------------------------------

TEST(BpeByteAlphabet, IsBijectiveAndMatchesGpt2)
{
    const auto& map = bpe_t::byte_to_unicode();

    std::vector<unsigned int> seen(map.begin(), map.end());
    std::sort(seen.begin(), seen.end());
    EXPECT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end())
        << "the 256 code points must be distinct, or decode is ambiguous";

    // Spot values pinned against GPT-2's bytes_to_unicode.
    EXPECT_EQ(map[0x00], 0x0100u); // -> "\u0100"
    EXPECT_EQ(map[0x0A], 0x010Au); // newline -> "\u010A"
    EXPECT_EQ(map[0x20], 0x0120u); // space   -> "\u0120"
    EXPECT_EQ(map[0x21], 0x0021u); // '!' maps to itself
    EXPECT_EQ(map[0x7E], 0x007Eu);
    EXPECT_EQ(map[0xA1], 0x00A1u);
    EXPECT_EQ(map[0x7F], 0x0121u); // first code point after the printable ASCII run
    EXPECT_EQ(map[0xAD], 0x0143u); // 0xAD is the single hole in the high range
    EXPECT_EQ(map[0xFF], 0x00FFu);
}

// ---------------------------------------------------------------------------
// Pre-tokenizer
// ---------------------------------------------------------------------------

TEST(BpePreTokenizer, WordsNumbersAndPunctuation)
{
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("Hello, my dog is cute")),
              (std::vector<std::string>{"Hello", ",", " my", " dog", " is", " cute"}));
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("abc123def")),
              (std::vector<std::string>{"abc", "123", "def"}));
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("1,000")),
              (std::vector<std::string>{"1", ",", "000"}));
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("")), (std::vector<std::string>{}));
}

TEST(BpePreTokenizer, ContractionsAreTheirOwnChunks)
{
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("isn't it")),
              (std::vector<std::string>{"isn", "'t", " it"}));
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("we're")),
              (std::vector<std::string>{"we", "'re"}));
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("I'll")),
              (std::vector<std::string>{"I", "'ll"}));
}

TEST(BpePreTokenizer, WhitespaceRunLendsItsLastSpaceToTheNextWord)
{
    // `\s+(?!\S)` backtracks by one code point, so a run of two spaces becomes a
    // lone space chunk plus a space-prefixed word. Getting this wrong shifts every
    // whitespace token in the output.
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("  my")),
              (std::vector<std::string>{" ", " my"}));
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("my  you")),
              (std::vector<std::string>{"my", " ", " you"}));
    // A run that ends the text keeps all of its spaces.
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("my  ")),
              (std::vector<std::string>{"my", "  "}));
}

TEST(BpePreTokenizer, NonAsciiUsesUnicodeClasses)
{
    // CJK ideographs are \p{L} and CJK punctuation is not, so the two must land in
    // different chunks. An "all non-ASCII is a letter" approximation merges them
    // and silently changes every id after that point.
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("\u4F60\u597D\uFF0C\u4E16\u754C")),
              (std::vector<std::string>{"\u4F60\u597D", "\uFF0C", "\u4E16\u754C"}));

    // An ideographic space is \s but not the literal U+0020 the ` ?` prefix needs,
    // so it is not attached to the following word.
    EXPECT_EQ(as_strings(bpe_t::pre_tokenize("\u3000\u4E16")),
              (std::vector<std::string>{"\u3000", "\u4E16"}));
}

TEST(BpePreTokenizer, ChunksAlwaysCoverTheInput)
{
    const std::string sample =
        "Hi!\t\u4F60\u597D, 12345 \u00E9\u00E8 \U0001F600\u2026''  ";
    std::size_t total = 0;
    for (const std::string_view chunk : bpe_t::pre_tokenize(sample))
    {
        ASSERT_FALSE(chunk.empty()) << "an empty chunk would make the encoder loop forever";
        total += chunk.size();
    }
    EXPECT_EQ(total, sample.size());
}

// ---------------------------------------------------------------------------
// Merge engine
// ---------------------------------------------------------------------------

class BpeToyTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        s_dir = write_toy_tokenizer();
        s_tok.load(s_dir, {"<|endoftext|>"});
    }

    static void TearDownTestSuite() { std::filesystem::remove_all(s_dir); }

    static std::string s_dir;
    static bpe_t s_tok;
};

std::string BpeToyTest::s_dir;
bpe_t BpeToyTest::s_tok;

TEST_F(BpeToyTest, LoadsVocabularyAndSpecialTokens)
{
    EXPECT_EQ(s_tok.vocab_size(), 262);
    EXPECT_TRUE(s_tok.is_special(260));
    EXPECT_FALSE(s_tok.is_special(0));
    EXPECT_EQ(s_tok.id_to_symbol(256), "ab");
    EXPECT_THROW(s_tok.id_to_symbol(262), std::runtime_error);

    // special_id is how callers recover a stop id without hard-coding it.
    EXPECT_EQ(s_tok.special_id("<|endoftext|>"), 260);
    EXPECT_EQ(s_tok.special_id("ab"), -1); // in the vocabulary, but not registered
    EXPECT_EQ(s_tok.special_id("nonexistent"), -1);
}

TEST_F(BpeToyTest, MergesTheLowestRankPairFirst)
{
    // "abc": a+b (rank 1) then ab+c (rank 3).
    EXPECT_EQ(s_tok.encode("abc"), (std::vector<int>{258}));

    // " ab": the pair (space, a) has rank 2 but (a, b) has rank 1, so "ab" wins
    // even though it starts further right. A left-to-right scan would give
    // ["\u0120a", "b"] here and then fail to match any pretrained checkpoint.
    EXPECT_EQ(s_tok.encode(" ab"), (std::vector<int>{32, 256}));

    // "a bc": the space attaches to "bc", and no rule applies.
    EXPECT_EQ(s_tok.encode("a bc"), (std::vector<int>{97, 32, 98, 99}));
}

TEST_F(BpeToyTest, HonoursMergeRulesWhoseLeftSymbolIsHash)
{
    // "# #" is a real merge rule, not the "#version" banner. Dropping it (or
    // skipping every '#'-prefixed line) leaves [2, 2] here.
    EXPECT_EQ(s_tok.encode("##"), (std::vector<int>{261}));
    EXPECT_EQ(s_tok.decode(std::vector<int>{261}), "##");
}

TEST_F(BpeToyTest, EncodesQuotesAndBackslashesThroughJsonEscapes)
{
    // Both symbols are written escaped in vocab.json and must survive the reader.
    EXPECT_EQ(s_tok.encode("\""), (std::vector<int>{34}));
    EXPECT_EQ(s_tok.encode("\\"), (std::vector<int>{92}));
}

TEST_F(BpeToyTest, IsolatesSpecialTokensFromMerging)
{
    EXPECT_EQ(s_tok.encode("a<|endoftext|>b"), (std::vector<int>{97, 260, 98}));
    EXPECT_EQ(s_tok.encode("<|endoftext|>"), (std::vector<int>{260}));

    EXPECT_EQ(s_tok.decode(std::vector<int>{260}), "<|endoftext|>");
    EXPECT_EQ(s_tok.decode(std::vector<int>{260}, /*skip_special=*/true), "");
}

TEST_F(BpeToyTest, DecodeIsTheInverseOfEncode)
{
    const std::vector<std::string> samples = {
        "",
        "abc",
        " ab",
        "Hello, my dog is cute",
        "\u4F60\u597D\uFF0C\u4E16\u754C",
        "\u00E9\u00E8\u00EA",
        "\U0001F600",
        "line1\nline2\ttab",
        "  three  spaces  ",
    };
    for (const std::string& sample : samples)
    {
        const std::vector<int> ids = s_tok.encode(sample);
        EXPECT_EQ(s_tok.decode(ids), sample) << "round trip failed for: " << sample;
    }
}

TEST_F(BpeToyTest, RoundTripsRandomUnicodeText)
{
    // The toy vocabulary carries the full byte alphabet, so any valid UTF-8 input
    // must survive a round trip regardless of which chunks the pre-tokenizer picks.
    std::mt19937 rng(20260928);
    std::vector<unsigned int> pool;
    for (unsigned int cp = 0x20; cp < 0x7F; ++cp)
        pool.push_back(cp);
    pool.insert(pool.end(), {0x00E9, 0x0120, 0x4E16, 0x754C, 0x3000, 0x1F600, 0x2026, 0x0A, 0x09});

    for (int trial = 0; trial < 200; ++trial)
    {
        std::string text;
        const int len = static_cast<int>(rng() % 24);
        for (int i = 0; i < len; ++i)
            text += utf8_of(pool[rng() % pool.size()]);
        const std::vector<int> ids = s_tok.encode(text);
        EXPECT_EQ(s_tok.decode(ids), text) << "round trip failed for: " << text;
    }
}

TEST_F(BpeToyTest, RejectsMalformedUtf8)
{
    EXPECT_THROW(s_tok.encode("\xE4\xBD"), std::runtime_error); // truncated
    EXPECT_THROW(s_tok.encode("\xFF"), std::runtime_error);     // invalid lead byte
}

// ---------------------------------------------------------------------------
// Merge engine: the two implementations must be interchangeable
// ---------------------------------------------------------------------------

TEST_F(BpeToyTest, TieBreaksToTheLeftmostOccurrence)
{
    // Both occurrences of (a, b) carry the same rank, so the tie-break decides the
    // result: merging the leftmost first yields [ab, ab], while merging the right
    // one first yields [a, b, ab] and then stops, because no rule applies to
    // (b, ab). This is the case where the two merge implementations could diverge.
    EXPECT_EQ(s_tok.encode("abab"), (std::vector<int>{256, 256}));
}

/**
 * encode() routes each chunk to one of two merge implementations by length
 * (merge_strategy::automatic), so they have to agree exactly. Driving both over
 * the same symbol sequences is the only way to check that directly, and the toy
 * vocabulary is convenient for it because its single-byte ids equal the byte
 * values, so a byte string is already a symbol sequence.
 */
TEST_F(BpeToyTest, BothMergeStrategiesAgree)
{
    std::mt19937 rng(20260929);
    const char alphabet[] = {'a', 'b', 'c', ' ', '"', '\\', '#', 'A'};
    constexpr std::size_t kAlphabet = sizeof(alphabet);

    int checked = 0;
    for (int trial = 0; trial < 4000; ++trial)
    {
        // Spans both sides of the automatic threshold, and past it, so the linked
        // list really is exercised rather than only the rescan.
        const int len = 1 + static_cast<int>(rng() % 160);

        std::vector<int> base;
        base.reserve(static_cast<std::size_t>(len));
        for (int i = 0; i < len; ++i)
            base.push_back(static_cast<unsigned char>(alphabet[rng() % kAlphabet]));

        std::vector<int> by_rescan = base;
        std::vector<int> by_heap = base;
        s_tok.merge_symbols(by_rescan, merge_strategy::rescan);
        s_tok.merge_symbols(by_heap, merge_strategy::linked_heap);

        ASSERT_EQ(by_rescan, by_heap)
            << "strategies disagree on: " << std::string(base.begin(), base.end());
        ++checked;
    }
    EXPECT_GT(checked, 0);
}

// ---------------------------------------------------------------------------
// Differential check against a real HuggingFace tokenizer directory
// ---------------------------------------------------------------------------

/**
 * Point JASMINE_GPT2_TOKENIZER_DIR at a local GPT-2 tokenizer directory (the
 * `vocab.json` + `merges.txt` pair, e.g. the files in a distilgpt2 snapshot) to
 * verify the encoder against the reference tokenizer instead of the toy fixture.
 *
 * TODO: also drive tools/gpt2_tokenizer_server.py over a corpus so arbitrary text
 * can be diffed, not just the golden prompts below.
 */
TEST(BpeGpt2, MatchesHuggingFaceGoldenIds)
{
    const char* dir = std::getenv("JASMINE_GPT2_TOKENIZER_DIR");
    if (dir == nullptr || *dir == '\0')
        GTEST_SKIP() << "set JASMINE_GPT2_TOKENIZER_DIR to a local GPT-2 tokenizer dir";

    bpe_t tok;
    ASSERT_NO_THROW(tok.load(dir, {"<|endoftext|>"})) << "failed to load " << dir;
    EXPECT_EQ(tok.vocab_size(), 50257);

    // Ids produced by HuggingFace's own GPT-2 tokenizer. The cases are chosen for
    // the places a hand-written pre-tokenizer usually drifts: contractions, runs
    // of spaces, CJK text, code, tabs and special tokens.
    struct golden_t
    {
        std::string text;
        std::vector<int> ids;
    };
    const std::vector<golden_t> goldens = {
        {"Hello, my dog is cute", {15496, 11, 616, 3290, 318, 13779}},
        {"The quick brown fox jumps over the lazy dog.",
         {464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13}},
        {"<|endoftext|>", {50256}},
        {"Hello<|endoftext|>world", {15496, 50256, 6894}},
        {"isnt it", {271, 429, 340}},
        {"  double space", {220, 4274, 2272}},
        {"trailing   ", {9535, 4386, 220, 220, 220}},
        {"abc123def", {39305, 10163, 4299}},
        {"\u4F60\u597D\uFF0C\u4E16\u754C",
         {19526, 254, 25001, 121, 171, 120, 234, 10310, 244, 45911, 234}},
        {"caf\u00E9 na\u00EFve", {66, 1878, 2634, 41492}},
        {"\U0001F600 emoji", {47249, 222, 44805}},
        {"def f(x):\n    return x+1", {4299, 277, 7, 87, 2599, 198, 220, 220, 220, 1441, 2124, 10, 16}},
        {"1,000.50", {16, 11, 830, 13, 1120}},
        {"a\tb", {64, 197, 65}},
    };

    for (const golden_t& golden : goldens)
    {
        EXPECT_EQ(tok.encode(golden.text), golden.ids) << "text: " << golden.text;
        EXPECT_EQ(tok.decode(golden.ids), golden.text) << "text: " << golden.text;
    }
}
