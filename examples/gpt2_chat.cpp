/**
 * GPT-2 交互式对话 demo（终端 REPL）。
 *
 * 模型权重只加载一次，KV cache 跨轮复用，因此每轮响应只需算新 token；
 * tokenizer 由常驻的 Python 子进程提供（见 tools/gpt2_tokenizer_server.py）。
 *
 * ⚠️ GPT-2 是 **base 语言模型，不是指令微调模型**：它不会「回答问题」，而是
 * **续写**你给的文字。--template chat 只是用 "User:/Assistant:" 这种文本格式
 * 引导它进入对话式的续写，效果远不如真正的 chat 模型。想看「问答」请换成
 * 指令微调模型；这个 demo 的价值在于验证 jasmine 的推理链路端到端可用。
 *
 * 用法：
 *   ./gpt2_chat build/distilgpt2_weights.bin [选项]
 *
 * 选项：
 *   --model NAME         tokenizer 用的 HF 模型名（默认从权重 manifest 读）
 *   --tokenizer-script P tokenizer 服务脚本路径（默认 tools/gpt2_tokenizer_server.py）
 *   --python CMD         python 解释器（默认 python3）
 *   --max-new N          每轮最多生成 N 个 token（默认 40）
 *   --temperature T      采样温度（默认 0.9；--greedy 时忽略）
 *   --top-k K            top-k 截断（默认 40）
 *   --greedy             argmax 解码
 *   --seed S             随机种子
 *   --template raw|chat  raw=直接续写（默认）；chat=包一层 User/Assistant 并遇到换行停止
 *   --max-context N      上下文 token 上限（默认取模型 n_pos；超出后丢弃最早的内容并重建 cache）
 *   --stop-token ID      生成时命中即停止（可重复；chat 模板会自动加换行 token）
 *   --no-stream          不开流式输出（默认逐 token 输出）
 *   --no-color           关闭 ANSI 颜色
 *
 * REPL 内命令：
 *   /help   /exit    /reset（清空对话与 cache）
 *   /context        显示当前上下文 token 数与最近对话
 *   /params         显示生成参数
 *   /set KEY VALUE  改参数：temperature / top_k / max_new / greedy(0|1) / template(raw|chat)
 */

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "jas_gpt2_t.hpp"
#include "jas_updator_t.hpp"
#include "jas_weight_io.hpp"
#include "chat_common.hpp"
#include "gpt2_generate.hpp"

using namespace jasmine;
using namespace jasmine_chat;

namespace
{

template <typename val_type>
using chat_upr_tpl = cache_updator_t<val_type, nadam_t>;
using chat_model_t = gpt2_model_t<mat_t<double>, chat_upr_tpl>;

// ---------------------------------------------------------------------------
// 生成参数
// ---------------------------------------------------------------------------
struct chat_params_t
{
    int max_new = 40;
    double temperature = 0.9;
    int top_k = 40;
    bool greedy = false;
    std::uint32_t seed = 1234;
    std::string templ = "raw";
};

} // namespace

int main(int argc, char** argv)
{
    std::ios::sync_with_stdio(false);

    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <weights.bin> [options]\n", argv[0]);
        std::fprintf(stderr, "       %s --help for the option list\n", argv[0]);
        return 2;
    }

    const std::string weights_path = argv[1];
    std::string model_id;
    std::string tokenizer_script = "tools/gpt2_tokenizer_server.py";
    std::string python_cmd = "python3";
    chat_params_t params;
    int max_context = 0;            // 0 = 用模型 n_pos
    std::vector<int> stop_tokens;
    bool stream = true;
    palette_t pal;

    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next_arg = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help")               { std::printf("see the header of examples/gpt2_chat.cpp\n"); return 0; }
        else if (a == "--model")         model_id = next_arg();
        else if (a == "--tokenizer-script") tokenizer_script = next_arg();
        else if (a == "--python")        python_cmd = next_arg();
        else if (a == "--max-new")       params.max_new = std::stoi(next_arg());
        else if (a == "--temperature")   params.temperature = std::stod(next_arg());
        else if (a == "--top-k")         params.top_k = std::stoi(next_arg());
        else if (a == "--greedy")        params.greedy = true;
        else if (a == "--seed")          params.seed = static_cast<std::uint32_t>(std::stoul(next_arg()));
        else if (a == "--template")      params.templ = next_arg();
        else if (a == "--max-context")   max_context = std::stoi(next_arg());
        else if (a == "--stop-token")    stop_tokens.push_back(std::stoi(next_arg()));
        else if (a == "--no-stream")     stream = false;
        else if (a == "--no-color")      pal.on = false;
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); return 2; }
    }

    if (params.templ != "raw" && params.templ != "chat")
    {
        std::fprintf(stderr, "--template must be 'raw' or 'chat'\n");
        return 2;
    }

    try
    {
        // ---- 权重 ----
        weight_file_t wf;
        wf.load(weights_path);
        const auto cfg = read_gpt2_config(wf);

        if (model_id.empty())
        {
            model_id = read_model_from_manifest(weights_path);
            if (model_id.empty()) model_id = "distilgpt2";
        }

        std::printf("%s[gpt2_chat]%s weights  %s (%d layers, d_model=%d, vocab=%d)\n",
                    pal.dim(), pal.reset(), weights_path.c_str(),
                    cfg.n_layers, cfg.d_model, cfg.vocab);

        chat_model_t model(cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_ff,
                           cfg.vocab, cfg.n_pos);
        load_gpt2(model, wf);
        model.reserve_kv_cache(cfg.n_pos);

        // ---- tokenizer（常驻子进程）----
        std::printf("%s[gpt2_chat]%s tokenizer %s\n",
                    pal.dim(), pal.reset(), model_id.c_str());
        tokenizer_client_t tokenizer(python_cmd, tokenizer_script, model_id);

        if (max_context <= 0) max_context = cfg.n_pos;
        max_context = std::min(max_context, cfg.n_pos);

        // chat 模板下默认在换行处停下，避免它一路续写到另一个 "User:"
        if (params.templ == "chat" && stop_tokens.empty())
        {
            const auto nl = tokenizer.encode("\n");
            if (nl.size() == 1)
                stop_tokens.push_back(nl[0]);
        }

        std::printf("%s[gpt2_chat]%s context limit %d tokens, template '%s'\n",
                    pal.dim(), pal.reset(), max_context, params.templ.c_str());
        std::printf("%s[gpt2_chat]%s GPT-2 is a base LM: it *continues* text, "
                    "it does not answer questions.\n", pal.dim(), pal.reset());
        std::printf("%s[gpt2_chat]%s type /help for commands, /exit to quit\n\n",
                    pal.dim(), pal.reset());

        // ---- 对话状态 ----
        std::vector<int> ctx_ids;       // 全部已喂给模型的 token（跨轮）
        chat_model_t* m = &model;
        const int eos_id = 50256;
        std::mt19937 rng(params.seed);

        auto reset = [&]() {
            ctx_ids.clear();
            m->clear_kv_cache();
        };

        auto chat_wrap = [&](const std::string& user_text) -> std::string {
            if (params.templ == "chat")
                return "User: " + user_text + "\nAssistant:";
            // raw 续写模式：轮与轮之间补一个换行，否则上一轮的结尾会和这一轮的开头
            // 粘成 "stillWhat" 这种词，模型会困惑（下一个 token 直接预测 endoftext）
            return ctx_ids.empty() ? user_text : ("\n" + user_text);
        };

        auto ensure_capacity = [&](int incoming) {
            if (static_cast<int>(ctx_ids.size()) + incoming <= max_context)
                return false;
            // 丢最早内容后位置会整体前移（绝对位置编码），必须重建 KV cache
            const int keep = std::max(1, std::min(max_context / 2, max_context - incoming - 1));
            const std::size_t n_keep = std::min<std::size_t>(
                static_cast<std::size_t>(std::max(0, keep)), ctx_ids.size());
            std::vector<int> kept(ctx_ids.end() - n_keep, ctx_ids.end());
            ctx_ids.clear();
            m->clear_kv_cache();
            for (std::size_t i = 0; i < kept.size(); ++i)
            {
                mat_t<double> id(1, 1, {static_cast<double>(kept[i])});
                m->forward_one(id, static_cast<int>(i));
                ctx_ids.push_back(kept[i]);
            }
            return true;
        };

        auto feed = [&](int id) {
            mat_t<double> one(1, 1, {static_cast<double>(id)});
            auto logits = m->forward_one(one, static_cast<int>(ctx_ids.size()));
            ctx_ids.push_back(id);
            return logits;
        };

        auto print_help = [&]() {
            std::printf("%scommands%s\n", pal.bold(), pal.reset());
            std::printf("  /help            show this help\n");
            std::printf("  /exit            quit (Ctrl-D works too)\n");
            std::printf("  /reset           clear the conversation and the KV cache\n");
            std::printf("  /context         show context length and the last few turns\n");
            std::printf("  /params          show sampling parameters\n");
            std::printf("  /set KEY VALUE   temperature | top_k | max_new | greedy | template\n");
            std::printf("\n");
        };

        auto print_params = [&]() {
            std::printf("  template    = %s\n", params.templ.c_str());
            std::printf("  max_new     = %d\n", params.max_new);
            std::printf("  decoding    = %s\n", params.greedy ? "greedy" : "sample");
            if (!params.greedy)
            {
                std::printf("  temperature = %.2f\n", params.temperature);
                std::printf("  top_k       = %d\n", params.top_k);
            }
            std::printf("  seed        = %u\n", params.seed);
            std::printf("  stop_tokens = ");
            if (stop_tokens.empty()) std::printf("(none)\n");
            else { for (int t : stop_tokens) std::printf("%d ", t); std::printf("\n"); }
        };

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
                    std::printf("  context = %zu / %d tokens\n",
                                ctx_ids.size(), max_context);
                    const std::size_t show = std::min<std::size_t>(ctx_ids.size(), 200);
                    std::vector<int> tail(ctx_ids.end() - show, ctx_ids.end());
                    std::printf("  tail    = %s\n", tokenizer.decode(tail).c_str());
                    continue;
                }
                if (cmd == "/set")
                {
                    std::string key, value;
                    ss >> key >> value;
                    try
                    {
                        if (key == "temperature") params.temperature = std::stod(value);
                        else if (key == "top_k")  params.top_k = std::stoi(value);
                        else if (key == "max_new") params.max_new = std::stoi(value);
                        else if (key == "greedy") params.greedy = (value == "1" || value == "true");
                        else if (key == "template")
                        {
                            if (value != "raw" && value != "chat")
                                throw std::runtime_error("template must be raw|chat");
                            params.templ = value;
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
            const std::string wrapped = chat_wrap(line);
            auto user_ids = tokenizer.encode(wrapped);
            if (user_ids.empty()) continue;

            if (ensure_capacity(static_cast<int>(user_ids.size()) + params.max_new))
                std::printf("%s[context full]%s dropped the oldest turns and rebuilt the cache\n",
                            pal.dim(), pal.reset());

            std::fflush(stdout);

            // 前向喂入本轮用户输入，拿到第一个预测
            mat_t<double> logits;
            for (int id : user_ids)
                logits = feed(id);

            std::printf("%sgpt2>%s ", pal.cyan(), pal.reset());
            std::fflush(stdout);

            // 自回归生成
            std::vector<int> reply;
            std::string printed;        // 已输出到终端的字节
            rng.seed(params.seed + static_cast<std::uint32_t>(ctx_ids.size()));
            for (int step = 0; step < params.max_new; ++step)
            {
                const int next = pick_next_token(logits, params, rng);
                if (next == eos_id) break;
                if (std::find(stop_tokens.begin(), stop_tokens.end(), next) != stop_tokens.end())
                    break;
                if (static_cast<int>(ctx_ids.size()) >= max_context) break;

                reply.push_back(next);
                if (stream)
                    printed = emit_increment(tokenizer.decode(reply), printed);
                logits = feed(next);
            }

            if (!stream)
            {
                std::printf("%s", tokenizer.decode(reply).c_str());
            }
            else
            {
                // 收尾：把之前为等补全而扣下的尾巴（含可能的收尾 U+FFFD）一并输出，
                // 与全量 decode 的结果保持一致
                emit_remainder(tokenizer.decode(reply), printed);
            }

            std::printf("\n%s[%zu new tokens, %zu in context]%s\n\n",
                        pal.dim(), reply.size(), ctx_ids.size(), pal.reset());
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s[gpt2_chat] error:%s %s\n", pal.red(), pal.reset(), e.what());
        return 1;
    }

    std::printf("%s[gpt2_chat]%s bye\n", pal.dim(), pal.reset());
    return 0;
}
