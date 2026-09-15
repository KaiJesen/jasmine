#ifndef JASMINE_EXAMPLES_GPT2_GENERATE_HPP
#define JASMINE_EXAMPLES_GPT2_GENERATE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jas_mat_t.hpp"

namespace jasmine {

/** 生成超参 */
struct gpt2_sample_opts_t
{
    int max_new_tokens = 32;     // 最多新生成多少个 token
    int eos_id = 50256;          // 命中即停止；<0 表示不按 EOS 停止
    bool greedy = true;          // true = argmax；false = top-k + temperature 采样
    double temperature = 1.0;    // 仅在 greedy=false 时生效
    int top_k = 0;               // 0 = 不做 top-k 截断
    std::uint32_t seed = 1234u;
};

/**
 * 从 [vocab, 1] 的 logits 里挑下一个 token。
 * greedy 取 argmax；否则在 top-k 内按 softmax(logits / temperature) 采样。
 *
 * opts_type 只需提供 greedy / temperature / top_k 三个字段；模板化是为了让
 * examples/gpt2_chat.cpp 能直接复用自己的参数结构。
 */
template <typename val_type, typename opts_type>
inline int pick_next_token(const mat_t<val_type>& logits,
                           const opts_type& opt,
                           std::mt19937& rng)
{
    const int V = logits.row_num();
    if (V <= 0)
        throw std::runtime_error("pick_next_token: empty logits");

    if (opt.greedy)
    {
        int best = 0;
        for (int v = 1; v < V; ++v)
            if (logits(v, 0) > logits(best, 0))
                best = v;
        return best;
    }

    const double inv_t = 1.0 / std::max(opt.temperature, 1e-6);
    const int k = (opt.top_k > 0 && opt.top_k < V) ? opt.top_k : V;

    // 取出 top-k（更简单直观，V=50257 时排序开销可接受）
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits(a, 0) > logits(b, 0); });
    idx.resize(k);

    const double mx = static_cast<double>(logits(idx.front(), 0)) * inv_t;
    std::vector<double> probs(k);
    double sum = 0.0;
    for (int i = 0; i < k; ++i)
    {
        const double e = std::exp(static_cast<double>(logits(idx[i], 0)) * inv_t - mx);
        probs[i] = e;
        sum += e;
    }
    for (auto& p : probs) p /= sum;

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return idx[dist(rng)];
}

/**
 * KV-cache 自回归生成。
 *
 * 流程：prompt 逐 token 走 forward_one 填充 KV cache（prefill），
 * 之后每步只喂上一个新 token，位置用绝对下标（wpe 索引）。
 *
 * @param on_token 每生成一个 token 回调 (step_index, token_id)，可为空
 * @return 完整序列 = prompt + 新生成的 token（不含触发停止的 EOS）
 */
template <typename model_type>
std::vector<int> gpt2_generate(model_type& model,
                               const std::vector<int>& prompt,
                               const gpt2_sample_opts_t& opt,
                               const std::function<void(int, int)>& on_token = {})
{
    using val_type = typename model_type::val_type;
    if (prompt.empty())
        throw std::runtime_error("gpt2_generate: empty prompt");

    const int n_pos = model.n_pos();
    if (static_cast<int>(prompt.size()) >= n_pos)
        throw std::runtime_error("gpt2_generate: prompt longer than n_pos (" +
                                 std::to_string(n_pos) + ")");

    std::mt19937 rng(opt.seed);
    model.clear_kv_cache();

    // prefill：逐 token 前向并填充 KV cache
    mat_t<val_type> logits;
    for (int t = 0; t < static_cast<int>(prompt.size()); ++t)
    {
        mat_t<val_type> id(1, 1, {static_cast<val_type>(prompt[t])});
        logits = model.forward_one(id, t);
    }

    std::vector<int> out = prompt;
    const int limit = std::min(opt.max_new_tokens, n_pos - static_cast<int>(prompt.size()));

    for (int step = 0; step < limit; ++step)
    {
        const int next = pick_next_token(logits, opt, rng);
        if (next == opt.eos_id && opt.eos_id >= 0)
            break;

        out.push_back(next);
        if (on_token)
            on_token(step, next);

        const int pos = static_cast<int>(out.size()) - 1;
        mat_t<val_type> id(1, 1, {static_cast<val_type>(next)});
        logits = model.forward_one(id, pos);
    }

    return out;
}

/** 读取 id 文件：每行一个 id，或逗号/空格分隔；忽略空行与 # 注释 */
inline std::vector<int> read_ids_file(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("read_ids_file: cannot open " + path);
    std::vector<int> ids;
    std::string line;
    while (std::getline(in, line))
    {
        for (char& c : line)
            if (c == ',' || c == '\t') c = ' ';
        std::istringstream ls(line);
        std::string tok;
        while (ls >> tok)
        {
            if (tok.empty() || tok[0] == '#') break;
            ids.push_back(std::stoi(tok));
        }
    }
    return ids;
}

} // namespace jasmine
#endif
