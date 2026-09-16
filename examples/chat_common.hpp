#ifndef JASMINE_EXAMPLES_CHAT_COMMON_HPP
#define JASMINE_EXAMPLES_CHAT_COMMON_HPP

/**
 * gpt2_chat 与 llama_chat 共用的终端 REPL 基础设施。
 *
 * 抽出来的原因：ANSI 颜色、base64、流式增量拼接、常驻 tokenizer 子进程这三块
 * 与具体模型无关，两份拷贝没有意义；而 tokenizer 客户端的协议本来就要同时兼容
 * GPT-2（E/D/V）与 LLaMA（多一个 M = apply_chat_template）。
 *
 * 这里的东西刻意做成「协议即接口」的薄封装，模型相关的部分留在各自的 demo 文件里。
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace jasmine_chat {

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
inline std::string base64_encode(const std::string& in)
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

inline std::string base64_decode(const std::string& in)
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
// 一个多字节字符可能跨多个 token。**不能**逐 token 调 decode —— HF 的 decode 对不完整的
// 字节序列会输出 U+FFFD（EF BF BD），原始字节就丢了，终端上表现为 "I��m"，且与全量 decode
// 的结果不一致（GPT-2 的 byte-level BPE 尤其明显；LLaMA 的 SPM 也会遇到）。
//
// 正确做法：每步都对「到目前为止的全部 token」重新 decode，然后
//   1) 只输出相对上次已输出内容的新增部分；
//   2) 掐掉末尾的 U+FFFD —— 它可能只是某个字符的前半截，等下一个 token 补齐。
//      （若模型真的输出了 U+FFFD，下一步因为它不再位于末尾而照常输出，不会丢。）
// 代价是每步重解一次整个序列，max_new 只有几十，可接受。
// ---------------------------------------------------------------------------
constexpr char kReplacementChar[] = "\xef\xbf\xbd";   // U+FFFD
constexpr std::size_t kReplacementLen = 3;

/** 返回 s 中「可以安全输出」的前缀长度（抹掉末尾的 U+FFFD） */
inline std::size_t stable_prefix_len(const std::string& s)
{
    std::size_t stable = s.size();
    while (stable >= kReplacementLen &&
           s.compare(stable - kReplacementLen, kReplacementLen, kReplacementChar) == 0)
        stable -= kReplacementLen;
    return stable;
}

/** 把 s 相对 already_printed 的新增且稳定的部分写出去；返回新的 already_printed */
inline std::string emit_increment(const std::string& s, const std::string& already_printed)
{
    const std::size_t stable = stable_prefix_len(s);
    if (stable <= already_printed.size()) return already_printed;
    if (s.compare(0, already_printed.size(), already_printed) != 0)
        return already_printed;             // 不该发生；保守起见不输出
    std::fwrite(s.data() + already_printed.size(), 1, stable - already_printed.size(), stdout);
    std::fflush(stdout);
    return s.substr(0, stable);
}

/** 收尾：把之前为等补全而扣下的尾巴（含可能的 U+FFFD）一并输出 */
inline void emit_remainder(const std::string& full, const std::string& printed)
{
    if (full.size() > printed.size())
        std::fwrite(full.data() + printed.size(), 1, full.size() - printed.size(), stdout);
}

// ---------------------------------------------------------------------------
// tokenizer 服务脚本定位
//
// 默认值是相对仓库根的路径（tools/xxx_server.py）。直接从仓库根跑没问题，
// 但从 build/ 或别处跑就会找不到。这里做个兜底：按 /proc/self/exe 定位可执行文件，
// 自下而上找 tools/<脚本名>，找到就用绝对路径。找不到则原样返回，
// 让 exec 的报错信息仍然指向用户给的那个路径。
// ---------------------------------------------------------------------------
inline std::string resolve_script_path(const std::string& script)
{
    if (script.empty())
        return script;

    auto exists = [](const std::string& p) {
        return ::access(p.c_str(), R_OK) == 0;
    };
    if (exists(script))
        return script;

    // 只对默认的仓库内相对路径做兜底；用户显式给的路径不猜
    const std::string rel_prefix = "tools/";
    if (script.rfind(rel_prefix, 0) != 0)
        return script;
    const std::string base_name = script.substr(rel_prefix.size());

    char exe[4096];
    const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0)
        return script;
    exe[n] = '\0';

    std::string dir(exe);
    const auto slash = dir.find_last_of('/');
    if (slash == std::string::npos)
        return script;
    dir.resize(slash);

    // 从 build/examples/ 往上最多 4 层找 tools/
    for (int up = 0; up < 4 && !dir.empty(); ++up)
    {
        const std::string cand = dir + "/" + rel_prefix + base_name;
        if (exists(cand))
            return cand;
        const auto pos = dir.find_last_of('/');
        if (pos == std::string::npos)
            break;
        dir.resize(pos);
    }
    return script;
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
            const std::string resolved = resolve_script_path(script);
            ::execlp(python_cmd.c_str(), python_cmd.c_str(), resolved.c_str(),
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

        // 握手：等 "ready <vocab> <n_pos> [eos]"（eos 是 LLaMA 服务才有的可选字段）
        std::string line = read_line();
        std::istringstream hs(line);
        std::string tag;
        hs >> tag;
        if (tag == "ready")
        {
            hs >> m_vocab >> m_n_pos >> m_eos;
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

    /** 服务端上报的 eos（LLaMA 服务才有；GPT-2 服务不上报，返回 -1） */
    int eos_id() const { return m_eos; }

    std::vector<int> encode(const std::string& text)
    {
        request("E " + base64_encode(text));
        return parse_ids();
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

    /**
     * 用 apply_chat_template 渲染整段对话（含生成提示），返回 token id。
     * 只有 LLaMA 系的服务支持（op `M`）。
     *
     * 为什么必须交给 Python 做：LLaMA 用 SentencePiece，`enc(a) + enc(b) != enc(a + b)`
     * （SPM 给每段文本前面补一个 ▁）。手写模板拼接一定会产出错误的 token 序列。
     */
    std::vector<int> render_chat(const std::vector<std::pair<std::string, std::string>>& messages)
    {
        std::string payload = "M " + std::to_string(messages.size());
        for (const auto& m : messages)
            payload += " " + base64_encode(m.first) + " " + base64_encode(m.second);
        request(payload);
        return parse_ids();
    }

private:
    std::vector<int> parse_ids()
    {
        std::istringstream rs(m_last);
        std::string tag;
        rs >> tag;
        if (tag != "ok")
            throw std::runtime_error("request failed: " + m_last);
        std::vector<int> ids;
        int id;
        while (rs >> id) ids.push_back(id);
        return ids;
    }

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
    int m_eos = -1;
    std::string m_last;
};

// ---------------------------------------------------------------------------
// 从权重 manifest（<weights>.json）里读模型名
// ---------------------------------------------------------------------------
inline std::string read_model_from_manifest(const std::string& weights_path)
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

/** 公共前缀长度（用于判断 canonical prompt 是否延续了已有 KV cache） */
template <typename T>
inline std::size_t common_prefix_len(const std::vector<T>& a, const std::vector<T>& b)
{
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

} // namespace jasmine_chat
#endif
