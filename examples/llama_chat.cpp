/**
 * TinyLlama-Chat（LLaMA 系）交互式对话 demo（终端 REPL）。
 *
 * 与 gpt2_chat 的关键差别：这是一个**真正的指令微调对话模型**，所以
 *   1. 用 `tokenizer.apply_chat_template` 渲染对话（由常驻 tokenizer 服务提供），
 *      自己拼 `<|user|>\n...` 一定会错 —— SPM 的 `enc(a)+enc(b) != enc(a+b)`。
 *   2. 生成在 `</s>`（eos）处停止，而不是靠「遇到换行就停」这种对 base 模型的将就办法。
 *
 * KV cache 复用策略（**正确性由前缀校验保证**）：
 *   每轮把整段对话交给 apply_chat_template 重新渲染出 canonical token 序列，
 *   然后与「已经喂进 cache 的序列」求公共前缀：
 *     * 若 cache 序列是 canonical 的前缀 → 只喂新增的尾部（快路径）；
 *     * 否则清空 cache 重喂整段（安全路径）。
 *   为什么不能无条件增量：上一轮生成的内容要回填成 assistant 消息，再渲染时
 *   经过「decode 成文本 → 重新 encode」这一次往返，未必与当初生成的 token 完全一致。
 *   前缀校验让这两种情况都正确：能复用就复用，不能就重来，绝不会喂进互相矛盾的 token。
 *
 * 用法：
 *   ./llama_chat build/tinyllama_weights.bin [选项]
 *
 * 选项：
 *   --model NAME         tokenizer 用的 HF 模型名（默认从权重 manifest 读）
 *   --tokenizer-script P tokenizer 服务脚本（默认 tools/llama_tokenizer_server.py）
 *   --python CMD         python 解释器（默认 python3）
 *   --system TEXT        系统提示词（默认 "You are a helpful assistant."）
 *   --max-new N          每轮最多生成 N 个 token（默认 256）
 *   --temperature T      采样温度（默认 0.7；--greedy 时忽略）
 *   --top-k K            top-k 截断（默认 50）
 *   --top-p P            nucleus 采样阈值（默认 0.95；1.0 = 关闭）
 *   --greedy             argmax 解码
 *   --seed S             随机种子
 *   --max-context N      上下文 token 上限（默认取模型 n_pos；超出则丢弃最早的轮次）
 *   --no-stream          不开流式输出（默认逐 token 输出）
 *   --no-color           关闭 ANSI 颜色
 *
 * REPL 内命令：
 *   /help   /exit    /reset（清空对话与 cache）
 *   /context        显示上下文 token 数与当前对话
 *   /params         显示生成参数
 *   /set KEY VALUE  改参数：temperature / top_k / top_p / max_new / greedy / system
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "jas_llama_t.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_updator_t.hpp"
#include "jas_weight_io.hpp"
#include "chat_common.hpp"
#include "gpt2_generate.hpp"        // 复用 pick_next_token（与模型无关）

using namespace jasmine;
using namespace jasmine_chat;

namespace
{

template <typename val_type>
using llama_upr_tpl = cache_updator_t<val_type, nadam_t>;

// 用 float 而不是 double 存权重：TinyLlama 有 1.1B 参数，
// double 要 8.8GB、float 只要 4.4GB，而推理精度差别远小于对齐容差（见 TESTING.md）。
using llama_chat_t = llama_model_t<mat_t<float>, llama_upr_tpl>;

struct chat_params_t
{
    int max_new = 256;
    double temperature = 0.7;
    int top_k = 50;
    double top_p = 0.95;
    bool greedy = false;
    std::uint32_t seed = 1234;
    std::string system = "You are a helpful assistant.";
};

/**
 * top-p 采样：pick_next_token 只支持 top-k，而 TinyLlama 的推荐配置带 top_p。
 * 这里在它的基础上加一层 nucleus 截断：先按概率降序累积，砍掉累计概率超过 top_p 的尾巴。
 * 因为 pick_next_token 内部自己做 softmax，这里把被砍掉的 logits 直接置 -inf 即可。
 */
template <typename val_type>
void apply_top_p(mat_t<val_type>& logits, double top_p, double temperature)
{
    if (top_p >= 1.0 || logits.row_num() <= 1) return;

    const int V = logits.row_num();
    const double inv_t = 1.0 / std::max(temperature, 1e-6);

    std::vector<int> idx(V);
    for (int v = 0; v < V; ++v) idx[v] = v;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return logits(a, 0) > logits(b, 0); });

    const double mx = static_cast<double>(logits(idx.front(), 0)) * inv_t;
    double sum = 0.0;
    for (int v = 0; v < V; ++v)
        sum += std::exp(static_cast<double>(logits(v, 0)) * inv_t - mx);
    if (sum <= 0.0) return;

    double acc = 0.0;
    for (int rank = 0; rank < V; ++rank)
    {
        acc += std::exp(static_cast<double>(logits(idx[rank], 0)) * inv_t - mx) / sum;
        if (acc >= top_p)
        {
            // rank 之后（含尾部）全部砍掉，但至少保留 1 个
            for (int k = rank + 1; k < V; ++k)
                logits(idx[k], 0) = -std::numeric_limits<val_type>::infinity();
            break;
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    std::ios::sync_with_stdio(false);

    // --help / -h 放在最前面，方便不看源码就知道怎么用
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
    {
        std::printf("usage: %s <weights.bin> [options]\n\n", argv[0]);
        std::printf(
            "  --model NAME          tokenizer 用的 HF 模型名（默认从权重 manifest 读）\n"
            "  --tokenizer-script P  tokenizer 服务脚本（默认 tools/llama_tokenizer_server.py）\n"
            "  --python CMD          python 解释器（默认 python3）\n"
            "  --system TEXT         系统提示词\n"
            "  --max-new N           每轮最多生成 N 个 token（默认 256）\n"
            "  --temperature T       采样温度（默认 0.7）\n"
            "  --top-k K             top-k 截断（默认 50）\n"
            "  --top-p P             nucleus 采样阈值（默认 0.95；1.0 = 关闭）\n"
            "  --greedy              argmax 解码\n"
            "  --seed S              随机种子\n"
            "  --max-context N       上下文 token 上限（默认取模型 n_pos）\n"
            "  --no-stream           不开流式输出\n"
            "  --no-color            关闭 ANSI 颜色\n"
            "\n"
            "对话内命令：/exit 退出，/reset 清空上下文，/context 显示上下文长度\n");
        return 0;
    }

    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <weights.bin> [options]\n", argv[0]);
        std::fprintf(stderr, "       %s --help for the option list\n", argv[0]);
        return 2;
    }

    const std::string weights_path = argv[1];
    std::string model_id;
    std::string tokenizer_script = "tools/llama_tokenizer_server.py";
    std::string python_cmd = "python3";
    chat_params_t params;
    int max_context = 0;             // 0 = 用模型 n_pos
    bool stream = true;
    palette_t pal;

    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next_arg = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h")   { std::printf("see the header of examples/llama_chat.cpp\n"); return 0; }
        else if (a == "--model")          model_id = next_arg();
        else if (a == "--tokenizer-script") tokenizer_script = next_arg();
        else if (a == "--python")         python_cmd = next_arg();
        else if (a == "--system")         params.system = next_arg();
        else if (a == "--max-new")        params.max_new = std::stoi(next_arg());
        else if (a == "--temperature")    params.temperature = std::stod(next_arg());
        else if (a == "--top-k")          params.top_k = std::stoi(next_arg());
        else if (a == "--top-p")          params.top_p = std::stod(next_arg());
        else if (a == "--greedy")         params.greedy = true;
        else if (a == "--seed")           params.seed = static_cast<std::uint32_t>(std::stoul(next_arg()));
        else if (a == "--max-context")    max_context = std::stoi(next_arg());
        else if (a == "--no-stream")      stream = false;
        else if (a == "--no-color")       pal.on = false;
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); return 2; }
    }

    try
    {
        // ---- 权重 ----
        weight_file_t wf;
        wf.load(weights_path);
        const auto cfg = read_llama_config(wf);

        // RoPE 基频目前只支持 10000；导入别的变体要显式失败，不能默默算错
        require_supported_rope_theta(cfg.rope_theta);

        if (model_id.empty())
        {
            model_id = read_model_from_manifest(weights_path);
            if (model_id.empty()) model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
        }

        std::printf("%s[llama_chat]%s weights  %s (%d layers, d_model=%d, heads=%d/%d, vocab=%d)\n",
                    pal.dim(), pal.reset(), weights_path.c_str(),
                    cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.vocab);

        llama_chat_t model;
        model.set_param(cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_ff,
                        cfg.vocab, cfg.n_pos, cfg.n_kv_heads,
                        static_cast<float>(cfg.rms_eps));
        load_llama(model, wf);

        if (max_context <= 0) max_context = cfg.n_pos;
        max_context = std::min(max_context, cfg.n_pos);
        model.reserve_kv_cache(max_context);

        // 权重已全部读入模型，把这 4.4GB 的原始 blob 还给系统
        wf.release();

        // ---- tokenizer（常驻子进程；chat 模板由它渲染）----
        std::printf("%s[llama_chat]%s tokenizer %s\n",
                    pal.dim(), pal.reset(), model_id.c_str());
        tokenizer_client_t tokenizer(python_cmd, tokenizer_script, model_id);

        if (tokenizer.vocab_size() != cfg.vocab)
            std::printf("%s[llama_chat]%s warning: tokenizer vocab %d != weight vocab %d\n",
                        pal.red(), pal.reset(), tokenizer.vocab_size(), cfg.vocab);

        // 角色标记在 SPM 里各自是一个 token；模型若吐出它们说明该轮结束了
        std::vector<int> stop_tokens;
        for (const char* marker : {"<|user|>", "<|assistant|>", "<|system|>"})
        {
            const auto ids = tokenizer.encode(marker);
            if (ids.size() == 1) stop_tokens.push_back(ids[0]);
        }
        const int eos_id = tokenizer.eos_id() >= 0 ? tokenizer.eos_id() : 2;

        std::printf("%s[llama_chat]%s context limit %d tokens, eos=%d\n",
                    pal.dim(), pal.reset(), max_context, eos_id);
        std::printf("%s[llama_chat]%s type /help for commands, /exit to quit\n\n",
                    pal.dim(), pal.reset());

        // ---- 对话状态 ----
        std::vector<std::pair<std::string, std::string>> messages;   // (role, content)
        std::vector<int> ctx_ids;        // 已实际喂进 KV cache 的 token（跨轮）
        std::mt19937 rng(params.seed);

        auto reset = [&]() {
            messages.clear();
            ctx_ids.clear();
            model.clear_kv_cache();
        };

        /** 用当前 messages（+ 系统提示词）渲染出 canonical token 序列 */
        auto render_prompt = [&]() {
            std::vector<std::pair<std::string, std::string>> rendered;
            if (!params.system.empty())
                rendered.emplace_back("system", params.system);
            rendered.insert(rendered.end(), messages.begin(), messages.end());
            return tokenizer.render_chat(rendered);
        };

        /**
         * 上下文超限时从最早处成对丢弃（user+assistant），并重渲染。
         * 丢最旧的会让位置整体前移，因此调用方必须重建 KV cache —— 由下面
         * 「前缀校验」自然完成，这里不需要额外处理。
         */
        auto trim_to_fit = [&](std::vector<int>& canonical) {
            while (static_cast<int>(canonical.size()) + params.max_new > max_context &&
                   !messages.empty())
            {
                const std::size_t drop = std::min<std::size_t>(2, messages.size());
                messages.erase(messages.begin(), messages.begin() + drop);
                canonical = render_prompt();
            }
            if (static_cast<int>(canonical.size()) + params.max_new > max_context)
                throw std::runtime_error(
                    "context limit " + std::to_string(max_context) +
                    " is too small for the system prompt and one turn; raise --max-context");
        };

        auto feed_one = [&](int id) {
            mat_t<float> one(1, 1, {static_cast<float>(id)});
            auto logits = model.forward_one(one);
            ctx_ids.push_back(id);
            return logits;
        };

        auto print_help = [&]() {
            std::printf("%scommands%s\n", pal.bold(), pal.reset());
            std::printf("  /help            show this help\n");
            std::printf("  /exit            quit (Ctrl-D works too)\n");
            std::printf("  /reset           clear the conversation and the KV cache\n");
            std::printf("  /context         show context length and the conversation\n");
            std::printf("  /params          show sampling parameters\n");
            std::printf("  /set KEY VALUE   temperature | top_k | top_p | max_new | greedy | system\n");
            std::printf("\n");
        };

        auto print_params = [&]() {
            std::printf("  system      = %s\n", params.system.c_str());
            std::printf("  max_new     = %d\n", params.max_new);
            std::printf("  decoding    = %s\n", params.greedy ? "greedy" : "sample");
            if (!params.greedy)
            {
                std::printf("  temperature = %.2f\n", params.temperature);
                std::printf("  top_k       = %d\n", params.top_k);
                std::printf("  top_p       = %.2f\n", params.top_p);
            }
            std::printf("  seed        = %u\n", params.seed);
        };

        bool warned_rebuild = false;
        std::string line;
        while (true)
        {
            std::printf("%syou>%s ", pal.bold(), pal.reset());
            std::fflush(stdout);
            if (!std::getline(std::cin, line))
            {
                std::printf("\n");
                break;
            }

            if (line.empty()) continue;

            if (line[0] == '/')
            {
                std::istringstream ss(line);
                std::string cmd;
                ss >> cmd;

                if (cmd == "/exit" || cmd == "/quit") break;
                if (cmd == "/help") { print_help(); continue; }
                if (cmd == "/reset")
                {
                    reset();
                    std::printf("%s[reset]%s conversation cleared\n", pal.dim(), pal.reset());
                    continue;
                }
                if (cmd == "/params") { print_params(); continue; }
                if (cmd == "/context")
                {
                    std::printf("  context = %zu / %d tokens, %zu messages\n",
                                ctx_ids.size(), max_context, messages.size());
                    for (const auto& m : messages)
                    {
                        std::string c = m.second;
                        if (c.size() > 120) c = c.substr(0, 120) + "…";
                        for (char& ch : c) if (ch == '\n') ch = ' ';
                        std::printf("    %-9s %s\n", m.first.c_str(), c.c_str());
                    }
                    continue;
                }
                if (cmd == "/set")
                {
                    std::string key, value;
                    ss >> key;
                    std::getline(ss, value);
                    while (!value.empty() && value.front() == ' ') value.erase(value.begin());
                    try
                    {
                        if (key == "temperature") params.temperature = std::stod(value);
                        else if (key == "top_k")  params.top_k = std::stoi(value);
                        else if (key == "top_p")  params.top_p = std::stod(value);
                        else if (key == "max_new") params.max_new = std::stoi(value);
                        else if (key == "greedy") params.greedy = (value == "1" || value == "true");
                        else if (key == "system")
                        {
                            params.system = value;
                            // 系统提示词变了，之前渲染的序列作废
                            reset();
                        }
                        else throw std::runtime_error("unknown key '" + key + "'");
                        std::printf("%s[set]%s %s = %s\n", pal.dim(), pal.reset(),
                                    key.c_str(), value.c_str());
                    }
                    catch (const std::exception& e)
                    {
                        std::printf("%s[set]%s %s\n", pal.red(), pal.reset(), e.what());
                    }
                    continue;
                }
                std::printf("%sunknown command '%s'%s — try /help\n",
                            pal.red(), cmd.c_str(), pal.reset());
                continue;
            }

            // ---- 正常对话轮 ----
            messages.emplace_back("user", line);
            auto canonical = render_prompt();
            trim_to_fit(canonical);

            std::fflush(stdout);

            // 前缀校验：canonical 是否延续了 cache 里的序列？
            // 上一轮生成的内容要回填成 assistant 消息再重新渲染，中间的
            // decode→encode 往返未必与当初生成的 token 完全一致；不一致就整段重喂。
            // 这让「复用」成为纯粹的优化，正确性不依赖它。
            const std::size_t before = ctx_ids.size();
            const std::size_t shared = common_prefix_len(canonical, ctx_ids);
            const bool rebuilt = shared < before;
            if (rebuilt)
            {
                model.clear_kv_cache();
                ctx_ids.clear();
                if (before > 0 && !warned_rebuild)
                {
                    std::printf("%s[note]%s re-prefilled the whole context "
                                "(generated text did not re-encode identically)\n",
                                pal.dim(), pal.reset());
                    warned_rebuild = true;
                }
            }

            const std::size_t start = ctx_ids.size();
            if (start >= canonical.size())
                throw std::runtime_error("nothing left to feed after prompt alignment");
            mat_t<float> tail(1, static_cast<int>(canonical.size() - start));
            for (std::size_t k = start; k < canonical.size(); ++k)
                tail(0, static_cast<int>(k - start)) = static_cast<float>(canonical[k]);

            auto logits = model.forward_one(tail);
            ctx_ids.assign(canonical.begin(), canonical.end());

            std::printf("%sllama>%s ", pal.green(), pal.reset());
            std::fflush(stdout);

            std::vector<int> reply;
            std::string printed;
            rng.seed(params.seed + static_cast<std::uint32_t>(ctx_ids.size()));
            for (int step = 0; step < params.max_new; ++step)
            {
                if (!params.greedy)
                    apply_top_p(logits, params.top_p, params.temperature);

                const int next = pick_next_token(logits, params, rng);

                if (next == eos_id) break;
                if (std::find(stop_tokens.begin(), stop_tokens.end(), next) != stop_tokens.end())
                    break;
                if (static_cast<int>(ctx_ids.size()) >= max_context) break;

                reply.push_back(next);
                if (stream)
                    printed = emit_increment(tokenizer.decode(reply), printed);
                logits = feed_one(next);
            }

            if (!stream)
                std::printf("%s", tokenizer.decode(reply).c_str());
            else
                emit_remainder(tokenizer.decode(reply), printed);

            const std::string reply_text = tokenizer.decode(reply);
            // 回填成 assistant 消息，供下一轮重新渲染
            messages.emplace_back("assistant", reply_text);

            std::printf("\n%s[%zu new tokens, %zu in context]%s\n\n",
                        pal.dim(), reply.size(), ctx_ids.size(), pal.reset());
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s[llama_chat] error:%s %s\n", pal.red(), pal.reset(), e.what());
        return 1;
    }

    std::printf("%s[llama_chat]%s bye\n", pal.dim(), pal.reset());
    return 0;
}
