#ifndef __JAS_BPE_T_HPP__
#define __JAS_BPE_T_HPP__

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "jas_unicode_class.hpp"

namespace jasmine {

/** Character class of a code point, mirroring the GPT-2 pre-tokenizer regex. */
enum class char_class
{
    letter,
    number,
    space,
    other
};

/**
 * Which implementation of the greedy merge loop to use.
 *
 * The two produce identical output; they differ only in cost.
 *   rescan      - rescan for the lowest-rank pair. O(k^2) but allocates nothing,
 *                 which wins on the short chunks real text produces.
 *   linked_heap - doubly linked list plus a min-heap. O(k log k), but allocates
 *                 per call, so it only wins once chunks get long.
 *   automatic   - rescan below k_heap_threshold, linked_heap above.
 */
enum class merge_strategy
{
    automatic,
    rescan,
    linked_heap
};

/**
 * Byte-level BPE tokenizer (the GPT-2 family): text <-> token ids.
 *
 * Deliberately independent of the numeric layers: a tokenizer turns text into
 * discrete ids, an embedding turns ids into vectors, and conflating the two is a
 * known trap. Nothing here includes mat_t or the network stack.
 *
 * Scope is the inference-side "merge engine": given a fixed vocabulary and a
 * fixed merge table, turn text into subword ids. Learning the merge table from a
 * corpus is a different algorithm (incremental pair counting) and does not live
 * here.
 *
 * Why bytes rather than characters: initial symbols are the 256 raw bytes, each
 * mapped onto a printable code point (space -> U+0120, newline -> U+010A, ...) so
 * that a merge table can be written as plain text. Every byte sequence is
 * therefore representable and no input ever produces an unknown token.
 *
 * The three pieces that have to agree with HuggingFace exactly, or the ids will
 * not match a pretrained checkpoint:
 *   1. the byte <-> code point mapping (see byte_to_unicode),
 *   2. the pre-tokenizer that bounds the merges (see pre_tokenize),
 *   3. the merge table and its rank order.
 *
 * Loading reads a HuggingFace tokenizer directory as-is: `vocab.json` plus
 * `merges.txt`. The JSON reader is purpose-built for a flat token -> id object
 * rather than a general parser; the file is machine generated and never nested,
 * and the alternative (an exporter step) would mean the ids cannot be checked
 * against the reference tokenizer without extra tooling.
 */
class bpe_t
{
public:
    // ------------------------------------------------------------------ loading

    /**
     * Reads `<dir>/vocab.json` and `<dir>/merges.txt`.
     *
     * `special_tokens` lists the tokens that must be matched literally and never
     * merged across; GPT-2 callers pass {"<|endoftext|>"}. They must already be
     * present in the vocabulary.
     */
    void load(const std::string& dir, const std::vector<std::string>& special_tokens = {});

    // -------------------------------------------------------------- text <-> id

    /** Encodes valid UTF-8 text into token ids. Throws on malformed UTF-8. */
    std::vector<int> encode(std::string_view text) const;

    /**
     * Decodes token ids back into the raw byte sequence.
     *
     * Returns bytes, not a validated UTF-8 string, on purpose: one character may
     * span several tokens, and only the caller knows how much of the tail is safe
     * to hand to a terminal (see the incremental-output handling in
     * examples/gpt2_chat.cpp). HuggingFace's lossy `clean_up_tokenization_spaces`
     * post-processing is intentionally not applied here.
     */
    std::string decode(std::span<const int> ids, bool skip_special = false) const;

    // ---------------------------------------------------------------- inspection

    int vocab_size() const { return static_cast<int>(m_symbol_of_id.size()); }

    const std::string& id_to_symbol(int id) const;

    bool is_special(int id) const;

    /**
     * Id of a special token that was registered at load time, or -1 when the
     * token is not among them. Lets callers recover a stop id (GPT-2's
     * `<|endoftext|>`) from the vocabulary instead of hard-coding the number.
     */
    int special_id(std::string_view token) const;

    /**
     * Splits text into the chunks the merge engine is allowed to work on. Merges
     * never cross a chunk boundary, so this is part of the tokenizer contract and
     * not an internal detail; exposing it also lets tests pin the boundaries
     * directly instead of inferring them from ids.
     */
    static std::vector<std::string_view> pre_tokenize(std::string_view text);

    // ------------------------------------------------------------------- engine

    /**
     * Merges a symbol-id sequence in place until no mergeable pair is left: the
     * same greedy step encode() runs on every pre-tokenizer chunk.
     *
     * Exposed because encode() picks between two implementations by chunk length
     * (see merge_strategy). Tests drive both on the same input to show they agree,
     * and benchmarks measure where the crossover sits.
     */
    void merge_symbols(std::vector<int>& syms,
                       merge_strategy strategy = merge_strategy::automatic) const;

    /**
     * The 256 code points raw bytes are mapped onto (space -> U+0120, newline ->
     * U+010A, ...), as in GPT-2's `bytes_to_unicode`. Public so tests and export
     * tooling can reproduce the mapping instead of re-deriving it.
     */
    static const std::array<unsigned int, 256>& byte_to_unicode();

    /** Token id of a single raw byte, i.e. its byte-level symbol. */
    int symbol_of_byte(unsigned char byte) const { return m_byte_token[byte]; }

private:
    /** Packs two symbol ids into the single key used by both merge maps. */
    static unsigned long long pair_key(int left, int right)
    {
        return (static_cast<unsigned long long>(static_cast<unsigned int>(left)) << 32) |
               static_cast<unsigned int>(right);
    }

    // --- merge table -----------------------------------------------------------

    // Taking the pair with the lowest rank (not the leftmost one) is what makes
    // the result independent of scan order; "newer" splits as ne|w|e|r precisely
    // because n+e was learned before w+e.
    std::unordered_map<unsigned long long, int> m_rank;   // (left, right) -> merge order
    std::unordered_map<unsigned long long, int> m_result; // (left, right) -> merged id

    // --- vocabulary ------------------------------------------------------------

    std::vector<std::string> m_symbol_of_id; // token id -> symbol text
    std::vector<char> m_is_special;          // token id -> matched literally, never merged
    int m_byte_token[256] = {};              // raw byte -> its single-byte token id

    // Code point of a byte-level symbol -> the byte it stands for. Built from the
    // 256-byte alphabet at load time, so decode never has to re-derive the mapping.
    std::unordered_map<unsigned int, unsigned char> m_unicode_to_byte;

    // Special tokens, longest first so that ties at the same position resolve to
    // the longest match.
    std::vector<std::pair<std::string, int>> m_specials;

    // --- scanning helpers ------------------------------------------------------

    static std::pair<unsigned int, std::size_t> decode_cp(std::string_view s, std::size_t pos);
    static void append_utf8(std::string& out, unsigned int cp);
    static bool in_ranges(const cp_range_t* ranges, std::size_t count, unsigned int cp);
    static char_class classify(unsigned int cp);

    static std::size_t next_chunk(std::string_view text, std::size_t pos);
    static std::size_t match_contraction(std::string_view text, std::size_t pos);
    static std::size_t match_space_prefixed_run(std::string_view text, std::size_t pos,
                                                char_class cls);
    static std::size_t match_whitespace(std::string_view text, std::size_t pos);

    // --- engine ----------------------------------------------------------------

    void encode_plain(std::string_view text, std::vector<int>& out) const;

    /** O(k^2) but allocation-free: the right default for the chunks real text yields. */
    void merge_rescan(std::vector<int>& syms) const;

    /** O(k log k): doubly linked list plus a lazy-deletion min-heap, for long chunks. */
    void merge_linked_heap(std::vector<int>& syms) const;

    // Chunk length at which merge_linked_heap starts to beat merge_rescan, measured
    // with benches/bench_bpe.cpp (BM_MergeRescan vs BM_MergeLinkedHeap):
    //
    //     length      8      16      32      64     256    1024
    //     rescan  0.12us  0.36us  1.18us  4.17us  62.5us  995us
    //     heap    0.22us  0.45us  0.94us  1.84us  8.87us  45us
    //
    // The crossover sits between 16 and 32: below it the rescan's lack of
    // allocation wins, above it the quadratic term loses fast (22x by 1024).
    // Real text barely notices either way (p99 chunk length is 30), but a single
    // pathological chunk (a ten-thousand digit run) would dominate the total work,
    // so long chunks must not take the quadratic path.
    static constexpr std::size_t k_heap_threshold = 32;

    // --- loading helpers -------------------------------------------------------

    static std::string path_join(const std::string& dir, const char* name);
    static std::string read_file(const std::string& path);
    static std::unordered_map<std::string, int> parse_vocab_json(const std::string& path);
    static void parse_json_string(std::string_view s, std::size_t& i, std::string& out);
    static unsigned int parse_hex4(std::string_view s, std::size_t pos);
};

// ---------------------------------------------------------------------------
// UTF-8
// ---------------------------------------------------------------------------

inline void bpe_t::append_utf8(std::string& out, unsigned int cp)
{
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
}

inline std::pair<unsigned int, std::size_t> bpe_t::decode_cp(std::string_view s, std::size_t pos)
{
    const unsigned char lead = static_cast<unsigned char>(s[pos]);
    if (lead < 0x80)
        return {lead, 1};

    std::size_t len = 0;
    unsigned int cp = 0;
    if ((lead & 0xE0) == 0xC0)
    {
        len = 2;
        cp = lead & 0x1Fu;
    }
    else if ((lead & 0xF0) == 0xE0)
    {
        len = 3;
        cp = lead & 0x0Fu;
    }
    else if ((lead & 0xF8) == 0xF0)
    {
        len = 4;
        cp = lead & 0x07u;
    }
    else
    {
        throw std::runtime_error("bpe: invalid UTF-8 lead byte");
    }

    if (pos + len > s.size())
        throw std::runtime_error("bpe: truncated UTF-8 sequence");

    for (std::size_t i = 1; i < len; ++i)
    {
        const unsigned char cont = static_cast<unsigned char>(s[pos + i]);
        if ((cont & 0xC0) != 0x80)
            throw std::runtime_error("bpe: invalid UTF-8 continuation byte");
        cp = (cp << 6) | (cont & 0x3Fu);
    }
    return {cp, len};
}

// ---------------------------------------------------------------------------
// Pre-tokenizer
// ---------------------------------------------------------------------------

inline bool bpe_t::in_ranges(const cp_range_t* ranges, std::size_t count, unsigned int cp)
{
    std::size_t lo = 0;
    std::size_t hi = count;
    while (lo < hi)
    {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid].lo)
            hi = mid;
        else if (cp > ranges[mid].hi)
            lo = mid + 1;
        else
            return true;
    }
    return false;
}

inline char_class bpe_t::classify(unsigned int cp)
{
    if (in_ranges(k_letter_ranges, std::size(k_letter_ranges), cp))
        return char_class::letter;
    if (in_ranges(k_number_ranges, std::size(k_number_ranges), cp))
        return char_class::number;
    if (in_ranges(k_space_ranges, std::size(k_space_ranges), cp))
        return char_class::space;
    return char_class::other;
}

inline std::size_t bpe_t::match_contraction(std::string_view text, std::size_t pos)
{
    static constexpr std::string_view kContractions[] = {"'s", "'t", "'re", "'ve",
                                                         "'m", "'ll", "'d"};
    for (const std::string_view contraction : kContractions)
        if (text.compare(pos, contraction.size(), contraction) == 0)
            return contraction.size();
    return 0;
}

inline std::size_t bpe_t::match_space_prefixed_run(std::string_view text, std::size_t pos,
                                                   char_class cls)
{
    const std::size_t n = text.size();
    std::size_t p = pos;
    // The ` ?` in the pattern is a literal U+0020, not the general \s class: an
    // NBSP or an ideographic space is a \s but must not be attached to the word.
    if (p < n && text[p] == ' ')
        ++p;
    if (p >= n)
        return 0;
    if (classify(decode_cp(text, p).first) != cls)
        return 0;

    std::size_t q = p;
    while (q < n)
    {
        const auto [cp, len] = decode_cp(text, q);
        if (classify(cp) != cls)
            break;
        q += len;
    }
    return q - pos;
}

inline std::size_t bpe_t::match_whitespace(std::string_view text, std::size_t pos)
{
    const std::size_t n = text.size();
    std::size_t q = pos;
    std::size_t last = pos;
    while (q < n)
    {
        const auto [cp, len] = decode_cp(text, q);
        if (classify(cp) != char_class::space)
            break;
        last = q; // byte offset of the final whitespace code point
        q += len;
    }

    if (q == pos)
        return 0;
    if (q == n)
        return q - pos; // the run ends the text, so `\s+(?!\S)` keeps all of it
    // `\s+(?!\S)` backtracks by one code point when the run is followed by a
    // non-space, leaving that final space to be picked up by the ` ?CLASS+`
    // alternatives. This is why "a  my" yields [" "," my"] rather than ["  ","my"].
    return last > pos ? last - pos : q - pos;
}

inline std::size_t bpe_t::next_chunk(std::string_view text, std::size_t pos)
{
    // Alternatives are tried in pattern order, because regex alternation is
    // ordered: the first one that matches wins, not the longest one.
    if (const std::size_t len = match_contraction(text, pos); len != 0)
        return len;

    for (const char_class cls : {char_class::letter, char_class::number, char_class::other})
        if (const std::size_t len = match_space_prefixed_run(text, pos, cls); len != 0)
            return len;

    if (const std::size_t len = match_whitespace(text, pos); len != 0)
        return len;

    // Every code point belongs to exactly one of the four classes, so this is
    // unreachable unless the input is not valid UTF-8.
    throw std::runtime_error("bpe: pre-tokenizer stalled on malformed UTF-8 input");
}

inline std::vector<std::string_view> bpe_t::pre_tokenize(std::string_view text)
{
    std::vector<std::string_view> chunks;
    std::size_t pos = 0;
    while (pos < text.size())
    {
        const std::size_t len = next_chunk(text, pos);
        chunks.push_back(text.substr(pos, len));
        pos += len;
    }
    return chunks;
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

inline void bpe_t::merge_symbols(std::vector<int>& syms, merge_strategy strategy) const
{
    if (strategy == merge_strategy::automatic)
        strategy = syms.size() >= k_heap_threshold ? merge_strategy::linked_heap
                                                  : merge_strategy::rescan;
    if (strategy == merge_strategy::linked_heap)
        merge_linked_heap(syms);
    else
        merge_rescan(syms);
}

inline void bpe_t::merge_rescan(std::vector<int>& syms) const
{
    // Repeatedly take the adjacent pair with the lowest rank, scanning for it from
    // scratch each round. Chunks handed over by the pre-tokenizer are short (a
    // word, a digit run, a punctuation run), so this beats a heap here: no
    // allocation, no staleness bookkeeping, and everything stays cache-resident.
    // Cost is quadratic in the chunk length, which is why encode() routes long
    // chunks to merge_linked_heap.
    while (syms.size() >= 2)
    {
        int best_rank = std::numeric_limits<int>::max();
        int best_i = -1;
        for (std::size_t i = 0; i + 1 < syms.size(); ++i)
        {
            const auto it = m_rank.find(pair_key(syms[i], syms[i + 1]));
            // Strictly less, scanning left to right, means "lowest rank, leftmost
            // occurrence wins". merge_linked_heap breaks ties the same way, which
            // is what makes the two implementations interchangeable.
            if (it != m_rank.end() && it->second < best_rank)
            {
                best_rank = it->second;
                best_i = static_cast<int>(i);
            }
        }
        if (best_i < 0)
            return; // nothing mergeable left

        const auto merged = m_result.find(pair_key(syms[best_i], syms[best_i + 1]));
        syms[best_i] = merged->second;
        syms.erase(syms.begin() + static_cast<std::ptrdiff_t>(best_i) + 1);
    }
}

inline void bpe_t::merge_linked_heap(std::vector<int>& syms) const
{
    const std::size_t n = syms.size();
    if (n < 2)
        return;

    // Symbols live in a doubly linked list built over index-stable arrays. Merging
    // relinks two nodes instead of moving a vector's tail, which is what takes the
    // loop from O(k^2) to O(k log k).
    std::vector<int> sym(syms);
    std::vector<int> next(n, -1);
    std::vector<int> prev(n, -1);
    // Generation of each node: bumped whenever a pair that touches it changes, so
    // that heap entries pushed earlier are recognisable as stale. Nodes are never
    // reused, so a mismatch means the entry describes a pair that no longer exists.
    std::vector<int> gen(n, 0);
    for (std::size_t i = 0; i + 1 < n; ++i)
        next[i] = static_cast<int>(i + 1);
    for (std::size_t i = 1; i < n; ++i)
        prev[i] = static_cast<int>(i - 1);

    struct candidate_t
    {
        int rank;
        int node; // index of the LEFT node of the pair
        int gen;  // gen[node] when the entry was pushed
    };
    struct lowest_first_t
    {
        bool operator()(const candidate_t& a, const candidate_t& b) const
        {
            // std::priority_queue is a max-heap, so invert to get the lowest rank
            // on top. Equal ranks fall back to the leftmost occurrence, which is
            // the tie-break merge_rescan applies.
            if (a.rank != b.rank)
                return a.rank > b.rank;
            return a.node > b.node;
        }
    };
    std::priority_queue<candidate_t, std::vector<candidate_t>, lowest_first_t> heap;

    const auto push_pair = [&](int node) {
        if (node < 0 || next[node] < 0)
            return;
        const auto it = m_rank.find(pair_key(sym[node], sym[next[node]]));
        if (it == m_rank.end())
            return;
        heap.push(candidate_t{it->second, node, gen[node]});
    };

    for (std::size_t i = 0; i + 1 < n; ++i)
        push_pair(static_cast<int>(i));

    while (!heap.empty())
    {
        const candidate_t top = heap.top();
        heap.pop();

        // Stale entries are dropped rather than deleted eagerly: a merge invalidates
        // the entries around it, and this is what notices. Deletion would need
        // decrease-key, which the generation counter replaces.
        if (top.node < 0 || static_cast<std::size_t>(top.node) >= n)
            continue;
        if (gen[static_cast<std::size_t>(top.node)] != top.gen)
            continue;
        const int node = top.node;
        const int right = next[node];
        if (right < 0)
            continue;
        const auto merged = m_result.find(pair_key(sym[node], sym[right]));
        if (merged == m_result.end())
            continue;

        // Merge `right` into `node`: the left node survives with the new symbol.
        const int before = prev[node];
        const int after = next[right];
        sym[node] = merged->second;
        next[node] = after;
        if (after >= 0)
            prev[after] = node;
        prev[right] = -1;
        next[right] = -1;

        // Exactly three pairs changed state: (before, node) because node's symbol
        // changed, (node, right) and (right, after) because they no longer exist.
        // Bumping the generation of each of their left nodes retires the matching
        // heap entries, and pushing the two surviving pairs replaces them. Note
        // that `after` is deliberately not bumped: its own pair is unaffected, and
        // invalidating it would silently drop a live candidate.
        if (before >= 0)
            ++gen[before];
        ++gen[node];
        ++gen[right];
        push_pair(before);
        push_pair(node);
    }

    syms.clear();
    for (int node = 0; node >= 0; node = next[node])
        syms.push_back(sym[node]);
}

inline void bpe_t::encode_plain(std::string_view text, std::vector<int>& out) const
{
    std::vector<int> syms;
    std::size_t pos = 0;
    while (pos < text.size())
    {
        const std::size_t len = next_chunk(text, pos);
        syms.clear();
        syms.reserve(len);
        for (std::size_t i = 0; i < len; ++i)
            syms.push_back(m_byte_token[static_cast<unsigned char>(text[pos + i])]);

        merge_symbols(syms);
        out.insert(out.end(), syms.begin(), syms.end());
        pos += len;
    }
}

inline std::vector<int> bpe_t::encode(std::string_view text) const
{
    std::vector<int> out;
    std::size_t pos = 0;
    while (pos < text.size())
    {
        // Special tokens are isolated, matching the `Split(..., Isolated)` step in
        // the HuggingFace pre-tokenizer: ordinary text never merges across one.
        std::size_t best_at = std::string_view::npos;
        int best_id = -1;
        std::size_t best_len = 0;
        for (const auto& special : m_specials)
        {
            const std::size_t at = text.find(special.first, pos);
            // m_specials is longest-first, so an equal position keeps the longer one.
            if (at != std::string_view::npos && at < best_at)
            {
                best_at = at;
                best_id = special.second;
                best_len = special.first.size();
            }
        }

        if (best_id < 0)
        {
            encode_plain(text.substr(pos), out);
            break;
        }
        if (best_at > pos)
            encode_plain(text.substr(pos, best_at - pos), out);
        out.push_back(best_id);
        pos = best_at + best_len;
    }
    return out;
}

inline const std::string& bpe_t::id_to_symbol(int id) const
{
    if (id < 0 || static_cast<std::size_t>(id) >= m_symbol_of_id.size())
        throw std::runtime_error("bpe: token id out of range: " + std::to_string(id));
    return m_symbol_of_id[static_cast<std::size_t>(id)];
}

inline bool bpe_t::is_special(int id) const
{
    return id >= 0 && static_cast<std::size_t>(id) < m_is_special.size() &&
           m_is_special[static_cast<std::size_t>(id)] != 0;
}

inline int bpe_t::special_id(std::string_view token) const
{
    // m_specials is short (GPT-2 has exactly one entry), so a linear scan is fine.
    for (const auto& special : m_specials)
        if (std::string_view(special.first) == token)
            return special.second;
    return -1;
}

inline std::string bpe_t::decode(std::span<const int> ids, bool skip_special) const
{
    std::string out;
    for (const int id : ids)
    {
        const std::string& sym = id_to_symbol(id);
        if (is_special(id))
        {
            // A special token is not byte-level encoded, so it is copied verbatim
            // and skipped entirely when the caller asks for plain text.
            if (!skip_special)
                out += sym;
            continue;
        }

        std::size_t pos = 0;
        while (pos < sym.size())
        {
            const auto [cp, len] = decode_cp(sym, pos);
            const auto it = m_unicode_to_byte.find(cp);
            if (it == m_unicode_to_byte.end())
                throw std::runtime_error("bpe: token " + std::to_string(id) +
                                         " is not a byte-level symbol");
            out.push_back(static_cast<char>(it->second));
            pos += len;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

inline const std::array<unsigned int, 256>& bpe_t::byte_to_unicode()
{
    static const std::array<unsigned int, 256> table = [] {
        std::array<unsigned int, 256> map{};
        std::array<bool, 256> taken{};
        // Printable bytes map to themselves.
        for (unsigned int b = 0x21; b <= 0x7E; ++b)
        {
            map[b] = b;
            taken[b] = true;
        }
        for (unsigned int b = 0xA1; b <= 0xAC; ++b)
        {
            map[b] = b;
            taken[b] = true;
        }
        for (unsigned int b = 0xAE; b <= 0xFF; ++b)
        {
            map[b] = b;
            taken[b] = true;
        }
        // The leftovers (controls, space, 0x7F-0xA0, 0xAD) are shifted into
        // U+0100.. in ascending byte order, which is what puts space on U+0120.
        unsigned int next = 256;
        for (unsigned int b = 0; b < 256; ++b)
            if (!taken[b])
                map[b] = next++;
        return map;
    }();
    return table;
}

inline std::string bpe_t::path_join(const std::string& dir, const char* name)
{
    if (!dir.empty() && dir.back() == '/')
        return dir + name;
    return dir + "/" + name;
}

inline std::string bpe_t::read_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("bpe: cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

inline unsigned int bpe_t::parse_hex4(std::string_view s, std::size_t pos)
{
    if (pos + 4 > s.size())
        throw std::runtime_error("bpe: truncated \\u escape in vocab.json");
    unsigned int value = 0;
    for (std::size_t i = 0; i < 4; ++i)
    {
        const char c = s[pos + i];
        unsigned int digit = 0;
        if (c >= '0' && c <= '9')
            digit = static_cast<unsigned int>(c - '0');
        else if (c >= 'a' && c <= 'f')
            digit = static_cast<unsigned int>(c - 'a') + 10;
        else if (c >= 'A' && c <= 'F')
            digit = static_cast<unsigned int>(c - 'A') + 10;
        else
            throw std::runtime_error("bpe: bad hex digit in \\u escape in vocab.json");
        value = (value << 4) | digit;
    }
    return value;
}

inline void bpe_t::parse_json_string(std::string_view s, std::size_t& i, std::string& out)
{
    // On entry s[i] is the opening quote. Appends the unescaped contents to `out`
    // and leaves `i` just past the closing quote.
    ++i;
    while (true)
    {
        if (i >= s.size())
            throw std::runtime_error("bpe: unterminated string in vocab.json");
        const char c = s[i];
        if (c == '"')
        {
            ++i;
            return;
        }
        if (c != '\\')
        {
            out.push_back(c);
            ++i;
            continue;
        }

        ++i;
        if (i >= s.size())
            throw std::runtime_error("bpe: dangling escape in vocab.json");
        switch (s[i])
        {
        case '"':
            out.push_back('"');
            ++i;
            break;
        case '\\':
            out.push_back('\\');
            ++i;
            break;
        case '/':
            out.push_back('/');
            ++i;
            break;
        case 'b':
            out.push_back('\b');
            ++i;
            break;
        case 'f':
            out.push_back('\f');
            ++i;
            break;
        case 'n':
            out.push_back('\n');
            ++i;
            break;
        case 'r':
            out.push_back('\r');
            ++i;
            break;
        case 't':
            out.push_back('\t');
            ++i;
            break;
        case 'u':
        {
            unsigned int cp = parse_hex4(s, i + 1);
            i += 5;
            // A code point above the BMP arrives as a UTF-16 surrogate pair.
            if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u')
            {
                const unsigned int low = parse_hex4(s, i + 2);
                if (low >= 0xDC00 && low <= 0xDFFF)
                {
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (low - 0xDC00u);
                    i += 6;
                }
            }
            append_utf8(out, cp);
            break;
        }
        default:
            throw std::runtime_error("bpe: unknown escape in vocab.json");
        }
    }
}

inline std::unordered_map<std::string, int> bpe_t::parse_vocab_json(const std::string& path)
{
    const std::string text = read_file(path);
    const std::string_view s = text;
    std::size_t i = 0;

    const auto skip_ws = [&] {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
            ++i;
    };
    const auto fail = [&](const char* what) {
        throw std::runtime_error(std::string("bpe: ") + what + " in " + path);
    };

    std::unordered_map<std::string, int> vocab;
    skip_ws();
    if (i >= s.size() || s[i] != '{')
        fail("expected a JSON object");
    ++i;

    while (true)
    {
        skip_ws();
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            break;
        }
        if (i >= s.size() || s[i] != '"')
            fail("expected a quoted token");
        std::string key;
        parse_json_string(s, i, key);

        skip_ws();
        if (i >= s.size() || s[i] != ':')
            fail("expected ':'");
        ++i;
        skip_ws();

        const std::size_t digits = i;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
            ++i;
        if (digits == i)
            fail("expected an integer id");
        const long long id = std::stoll(std::string(s.substr(digits, i - digits)));
        if (id < 0)
            fail("negative token id");
        vocab.emplace(std::move(key), static_cast<int>(id));

        skip_ws();
        if (i < s.size() && s[i] == ',')
        {
            ++i;
            continue;
        }
        if (i < s.size() && s[i] == '}')
        {
            ++i;
            break;
        }
        fail("expected ',' or '}'");
    }
    return vocab;
}

inline void bpe_t::load(const std::string& dir, const std::vector<std::string>& special_tokens)
{
    m_symbol_of_id.clear();
    m_is_special.clear();
    m_rank.clear();
    m_result.clear();
    m_unicode_to_byte.clear();
    m_specials.clear();
    std::fill(std::begin(m_byte_token), std::end(m_byte_token), 0);

    const auto symbol_to_id = parse_vocab_json(path_join(dir, "vocab.json"));
    if (symbol_to_id.empty())
        throw std::runtime_error("bpe: empty vocabulary in " + dir);

    int max_id = 0;
    for (const auto& entry : symbol_to_id)
        max_id = std::max(max_id, entry.second);
    m_symbol_of_id.assign(static_cast<std::size_t>(max_id) + 1, std::string());
    for (const auto& entry : symbol_to_id)
        m_symbol_of_id[static_cast<std::size_t>(entry.second)] = entry.first;
    for (std::size_t id = 0; id < m_symbol_of_id.size(); ++id)
        if (m_symbol_of_id[id].empty())
            throw std::runtime_error("bpe: vocabulary ids are not contiguous in " + dir);

    // Anchor the byte alphabet: every raw byte must have its own single-byte
    // token, otherwise the tokenizer is not byte-level and decoding would have
    // holes. This is also what makes m_unicode_to_byte a bijection.
    const auto& alphabet = byte_to_unicode();
    for (unsigned int byte = 0; byte < 256; ++byte)
    {
        std::string symbol;
        append_utf8(symbol, alphabet[byte]);
        const auto it = symbol_to_id.find(symbol);
        if (it == symbol_to_id.end())
            throw std::runtime_error("bpe: no single-byte token for byte " +
                                     std::to_string(byte) + " in " + dir);
        m_byte_token[byte] = it->second;
        m_unicode_to_byte.emplace(alphabet[byte], static_cast<unsigned char>(byte));
    }

    for (const std::string& special : special_tokens)
    {
        const auto it = symbol_to_id.find(special);
        if (it == symbol_to_id.end())
            throw std::runtime_error("bpe: special token not in vocabulary: " + special);
        m_specials.emplace_back(special, it->second);
    }
    std::sort(m_specials.begin(), m_specials.end(),
              [](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) {
                  return a.first.size() > b.first.size();
              });

    m_is_special.assign(m_symbol_of_id.size(), 0);
    for (const auto& special : m_specials)
        m_is_special[static_cast<std::size_t>(special.second)] = 1;

    // merges.txt is HuggingFace's native file: a `#version` header, then one rule
    // per line in learning order. The line index (header excluded) is the rank.
    std::ifstream in(path_join(dir, "merges.txt"));
    if (!in)
        throw std::runtime_error("bpe: cannot open " + path_join(dir, "merges.txt"));

    std::string line;
    int rank = 0;
    while (std::getline(in, line))
    {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        // Only the `#version: 0.2` banner is a comment. A plain "starts with #"
        // rule would also drop real merge rules, and '#' is a byte symbol that
        // legitimately appears on the left-hand side: GPT-2's file contains
        // "# #", "# $", "#### ####" and five more of the same.
        if (line.rfind("#version", 0) == 0 || line.empty())
            continue;

        const std::size_t split = line.find(' ');
        if (split == std::string::npos)
            throw std::runtime_error("bpe: malformed merges.txt line " + std::to_string(rank));
        // Splitting on the first space is unambiguous for byte-level vocabularies:
        // a literal space never appears inside a symbol, it is stored as U+0120.
        const std::string left = line.substr(0, split);
        const std::string right = line.substr(split + 1);

        const auto left_id = symbol_to_id.find(left);
        const auto right_id = symbol_to_id.find(right);
        const auto merged_id = symbol_to_id.find(left + right);
        if (left_id == symbol_to_id.end() || right_id == symbol_to_id.end() ||
            merged_id == symbol_to_id.end())
            throw std::runtime_error("bpe: merges.txt line " + std::to_string(rank) +
                                     " references a symbol outside the vocabulary");

        const unsigned long long key = pair_key(left_id->second, right_id->second);
        // Duplicated rules would only appear in a hand-edited file; keeping the
        // first occurrence preserves the lowest rank.
        m_rank.emplace(key, rank);
        m_result.emplace(key, merged_id->second);
        ++rank;
    }

    if (m_rank.empty())
        throw std::runtime_error("bpe: empty merge table in " + dir);
}

} // namespace jasmine

#endif // __JAS_BPE_T_HPP__
