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
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "jas_gpt2_t.hpp"
#include "jas_updator_t.hpp"
#include "jas_weight_io.hpp"
#include "gpt2_generate.hpp"

using namespace jasmine;

namespace
{

template <typename val_type>
using chat_upr_tpl = cache_updator_t<val_type, nadam_t>;
using chat_model_t = gpt2_model_t<mat_t<double>, chat_upr_tpl>;

// ---------------------------------------------------------------------------
// ANSI 颜色
// ---------------------------------------------------------------------------
struct palette_t
{
    bool on = true;
    const char* dim()   const { return on ? "\033[2m" : ""; }
    const char* bold()  const { return on ? "\033[1m" : ""; }
    const char* cyan()  const { return on ? "\033[36m" : ""; }
    const char* green() const { return on ? "\033[32m" : ""; }
    const char* red()   const { return on ? "\033[31m" : ""; }
    const char* reset() const { return on ? "\033[0m" : ""; }
};

// ---------------------------------------------------------------------------
// base64（与 tokenizer 服务协议配套；文本走 base64 以免任何转义/编码问题）
// ---------------------------------------------------------------------------
std::string base64_encode(const std::string& in)
{
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    std::size_t i = 0;
    while (i + 2 < in.size())
    {
        const unsigned v = (static_cast<unsigned char>(in[i]) << 16) |
                           (static_cast<unsigned char>(in[i + 1]) << 8) |
                           static_cast<unsigned char>(in[i + 2]);
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += tbl[(v >> 6) & 63];
        out += tbl[v & 63];
        i += 3;
    }
    const std::size_t rem = in.size() - i;
    if (rem == 1)
    {
        const unsigned v = static_cast<unsigned char>(in[i]) << 16;
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += "==";
    }
    else if (rem == 2)
    {
        const unsigned v = (static_cast<unsigned char>(in[i]) << 16) |
                           (static_cast<unsigned char>(in[i + 1]) << 8);
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += tbl[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

std::string base64_decode(const std::string& in)
{
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string out;
    int buf = 0, bits = 0;
    for (char c : in)
    {
        if (c == '=' || c == '\n' || c == '\r') continue;
        const int v = val(c);
        if (v < 0) continue;
        buf = (buf << 6) | v;
        bits += 6;
        if (bits >= 8)
        {
            bits -= 8;
            out += static_cast<char>((buf >> bits) & 0xFF);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 流式输出的增量拼接
//
// GPT-2 是 byte-level BPE：一个多字节字符可能跨多个 token。**不能**逐 token 调
// decode —— HF 的 decode 对不完整的字节序列会输出 U+FFFD（EF BF BD），原始字节就
// 丢了，终端上表现为 "I��m" 且与全量 decode 结果不一致。
//
// 正确做法：每步都对「到目前为止的全部 token」重新 decode，然后
//   1) 只输出相对上次已输出内容的新增部分；
//   2) 掐掉末尾的 U+FFFD —— 它可能只是某个字符的前半截，等下一个 token 补齐。
//      （若模型真的输出了 U+FFFD，下一步会因为不再位于末尾而照常输出，不会丢。）
// 代价是每步重解一次整个序列，max_new 只有几十，可接受。
// ---------------------------------------------------------------------------
constexpr char kReplacementChar[] = "\xef\xbf\xbd";   // U+FFFD
constexpr std::size_t kReplacementLen = 3;

/** 返回 s 中「可以安全输出」的前缀长度（抹掉末尾的 U+FFFD） */
std::size_t stable_prefix_len(const std::string& s)
{
    std::size_t stable = s.size();
    while (stable >= kReplacementLen &&
           s.compare(stable - kReplacementLen, kReplacementLen, kReplacementChar) == 0)
        stable -= kReplacementLen;
    return stable;
}

/** 把 s 相对 already_printed 的新增且稳定的部分写出去；返回新的 already_printed */
std::string emit_increment(const std::string& s, const std::string& already_printed)
{
    const std::size_t stable = stable_prefix_len(s);
    if (stable <= already_printed.size()) return already_printed;
    if (s.compare(0, already_printed.size(), already_printed) != 0)
        return already_printed;             // 不该发生；保守起见不输出
    std::fwrite(s.data() + already_printed.size(), 1, stable - already_printed.size(), stdout);
    std::fflush(stdout);
    return s.substr(0, stable);
}

// ---------------------------------------------------------------------------
// 常驻 tokenizer 子进程（fork/exec + 双向管道）
// ---------------------------------------------------------------------------
class tokenizer_client_t
{
public:
    tokenizer_client_t(const std::string& python_cmd,
                       const std::string& script,
                       const std::string& model)
    {
        int in_pipe[2];     // parent writes -> child stdin
        int out_pipe[2];    // child stdout -> parent reads
        if (::pipe(in_pipe) != 0 || ::pipe(out_pipe) != 0)
            throw std::runtime_error("pipe() failed");

        m_pid = ::fork();
        if (m_pid < 0)
            throw std::runtime_error("fork() failed");

        if (m_pid == 0)
        {
            // 子进程
            ::dup2(in_pipe[0], STDIN_FILENO);
            ::dup2(out_pipe[1], STDOUT_FILENO);
            ::close(in_pipe[0]); ::close(in_pipe[1]);
            ::close(out_pipe[0]); ::close(out_pipe[1]);
            ::execlp(python_cmd.c_str(), python_cmd.c_str(), script.c_str(),
                     "--model", model.c_str(), static_cast<char*>(nullptr));
            std::fprintf(stderr, "failed to exec %s\n", python_cmd.c_str());
            ::_exit(127);
        }

        ::close(in_pipe[0]);
        ::close(out_pipe[1]);
        m_wfd = in_pipe[1];
        m_out = ::fdopen(out_pipe[0], "r");
        if (m_out == nullptr)
            throw std::runtime_error("fdopen() failed");

        // 握手：等 "ready <vocab> <n_pos>"
        std::string line = read_line();
        std::istringstream hs(line);
        std::string tag;
        hs >> tag;
        if (tag == "ready")
        {
            hs >> m_vocab >> m_n_pos;
        }
        else
        {
            // 服务端把错误放在 base64 里报回来
            std::string b64;
            hs >> b64;
            throw std::runtime_error("tokenizer server failed: " +
                                     (b64.empty() ? line : base64_decode(b64)));
        }
    }

    ~tokenizer_client_t()
    {
        if (m_wfd >= 0)
        {
            const std::string q = "Q\n";
            ssize_t ignored = ::write(m_wfd, q.data(), q.size());
            (void)ignored;
            ::close(m_wfd);
        }
        if (m_out) { if (::fclose(m_out) != 0) { /* ignore */ } }
        if (m_pid > 0) ::waitpid(m_pid, nullptr, 0);
    }

    tokenizer_client_t(const tokenizer_client_t&) = delete;
    tokenizer_client_t& operator=(const tokenizer_client_t&) = delete;

    int vocab_size() const { return m_vocab; }
    int n_pos() const { return m_n_pos; }

    std::vector<int> encode(const std::string& text)
    {
        request("E " + base64_encode(text));
        std::istringstream rs(m_last);
        std::string tag;
        rs >> tag;
        if (tag != "ok")
            throw std::runtime_error("encode failed: " + m_last);
        std::vector<int> ids;
        int id;
        while (rs >> id) ids.push_back(id);
        return ids;
    }

    std::string decode(const std::vector<int>& ids)
    {
        std::string payload = "D";
        for (int id : ids)
            payload += " " + std::to_string(id);
        request(payload);
        std::istringstream rs(m_last);
        std::string tag, b64;
        rs >> tag >> b64;
        if (tag != "ok")
            throw std::runtime_error("decode failed: " + m_last);
        return base64_decode(b64);
    }

private:
    std::string read_line()
    {
        char* buf = nullptr;
        std::size_t cap = 0;
        const ssize_t n = ::getline(&buf, &cap, m_out);
        std::string line = (n >= 0 && buf) ? std::string(buf, static_cast<std::size_t>(n)) : std::string();
        std::free(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();
        return line;
    }

    void request(const std::string& line)
    {
        const std::string req = line + "\n";
        std::size_t off = 0;
        while (off < req.size())
        {
            const ssize_t n = ::write(m_wfd, req.data() + off, req.size() - off);
            if (n <= 0)
                throw std::runtime_error("tokenizer pipe write failed");
            off += static_cast<std::size_t>(n);
        }
        m_last = read_line();
        if (m_last.empty())
            throw std::runtime_error("tokenizer server closed the pipe");
    }

    pid_t m_pid = -1;
    int m_wfd = -1;
    FILE* m_out = nullptr;
    int m_vocab = 0;
    int m_n_pos = 1024;
    std::string m_last;
};

// ---------------------------------------------------------------------------
// 从权重 manifest（<weights>.json）里读模型名
// ---------------------------------------------------------------------------
std::string read_model_from_manifest(const std::string& weights_path)
{
    std::ifstream in(weights_path + ".json");
    if (!in) return {};
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    const std::string key = "\"model\"";
    auto pos = text.find(key);
    if (pos == std::string::npos) return {};
    pos = text.find(':', pos);
    if (pos == std::string::npos) return {};
    pos = text.find('"', pos);
    if (pos == std::string::npos) return {};
    const auto end = text.find('"', pos + 1);
    if (end == std::string::npos) return {};
    return text.substr(pos + 1, end - pos - 1);
}

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
                const std::string full = tokenizer.decode(reply);
                if (full.size() > printed.size())
                    std::fwrite(full.data() + printed.size(), 1,
                                full.size() - printed.size(), stdout);
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
