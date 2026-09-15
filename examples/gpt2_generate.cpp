/**
 * GPT-2 推理 demo：加载导出的权重 + prompt token ids，做 KV-cache 自回归生成。
 *
 * 用法：
 *   ./gpt2_generate <weights.bin> <prompt_ids.txt> [选项]
 *
 * 选项：
 *   --max-new N         最多生成 N 个 token（默认 32）
 *   --eos ID            EOS token id（默认 50256），<0 关闭
 *   --greedy            argmax 解码（默认）
 *   --sample            采样解码
 *   --temperature T     采样温度（默认 1.0）
 *   --top-k K           top-k 截断（默认 0 = 不截断）
 *   --seed S            随机种子（默认 1234）
 *   --out-ids FILE      把完整 id 序列写到文件（每行一个）
 *   --dump-logits FILE  把整段序列的 logits dump 成权重文件格式（供 tools/verify_gpt2.py 比对）
 *   --quiet             不逐 token 打印进度
 *
 * prompt ids 由 tools/gpt2_tokenize.py 生成；生成结果可用同脚本 decode 回文本：
 *   python tools/gpt2_tokenize.py encode --model distilgpt2 --text "Hello" --out prompt_ids.txt
 *   ./build/examples/gpt2_generate build/distilgpt2_weights.bin prompt_ids.txt --out-ids out.txt
 *   python tools/gpt2_tokenize.py decode --model distilgpt2 --ids-file out.txt
 */
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "jas_gpt2_t.hpp"
#include "jas_updator_t.hpp"
#include "jas_weight_io.hpp"
#include "gpt2_generate.hpp"

using namespace jasmine;

namespace
{

template <typename val_type>
using gpt2_upr_tpl = cache_updator_t<val_type, nadam_t>;

void print_usage(const char* argv0)
{
    std::fprintf(stderr,
        "usage: %s <weights.bin> <prompt_ids.txt> [--max-new N] [--eos ID] "
        "[--greedy|--sample] [--temperature T] [--top-k K] [--seed S] "
        "[--out-ids FILE] [--quiet]\n", argv0);
}

void write_ids_file(const std::string& path, const std::vector<int>& ids)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot write " + path);
    for (int id : ids)
        out << id << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        print_usage(argv[0]);
        return 2;
    }

    const std::string weights_path = argv[1];
    const std::string prompt_path = argv[2];

    gpt2_sample_opts_t opt;
    std::string out_ids;
    std::string dump_logits;
    bool quiet = false;

    for (int i = 3; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next_arg = [&](void) -> std::string {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--max-new")          opt.max_new_tokens = std::stoi(next_arg());
        else if (a == "--eos")         opt.eos_id = std::stoi(next_arg());
        else if (a == "--greedy")      opt.greedy = true;
        else if (a == "--sample")      opt.greedy = false;
        else if (a == "--temperature") opt.temperature = std::stod(next_arg());
        else if (a == "--top-k")       opt.top_k = std::stoi(next_arg());
        else if (a == "--seed")        opt.seed = static_cast<std::uint32_t>(std::stoul(next_arg()));
        else if (a == "--out-ids")     out_ids = next_arg();
        else if (a == "--dump-logits") dump_logits = next_arg();
        else if (a == "--quiet")       quiet = true;
        else
        {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    try
    {
        weight_file_t wf;
        wf.load(weights_path);
        const auto cfg = read_gpt2_config(wf);

        std::printf("[gpt2] config: layers=%d heads=%d d_model=%d d_ff=%d vocab=%d n_pos=%d\n",
                    cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_ff, cfg.vocab, cfg.n_pos);

        gpt2_model_t<mat_t<double>, gpt2_upr_tpl> model(
            cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_ff, cfg.vocab, cfg.n_pos);
        load_gpt2(model, wf);
        std::printf("[gpt2] weights loaded from %s\n", weights_path.c_str());

        // 固定容量 KV cache，避免生成过程中扩容
        model.reserve_kv_cache(cfg.n_pos);

        const std::vector<int> prompt = read_ids_file(prompt_path);
        if (prompt.empty())
        {
            std::fprintf(stderr, "[gpt2] prompt is empty: %s\n", prompt_path.c_str());
            return 1;
        }
        if (static_cast<int>(prompt.size()) > cfg.n_pos)
        {
            std::fprintf(stderr, "[gpt2] prompt has %zu tokens > n_pos %d\n",
                         prompt.size(), cfg.n_pos);
            return 1;
        }

        std::printf("[gpt2] prompt: %zu tokens\n", prompt.size());
        if (!quiet)
        {
            std::printf("[gpt2] prompt ids:");
            for (int id : prompt) std::printf(" %d", id);
            std::printf("\n");
        }

        std::string decode_desc = opt.greedy ? "greedy"
                                             : ("sample (temperature=" + std::to_string(opt.temperature) +
                                                ", top_k=" + std::to_string(opt.top_k) + ")");
        std::printf("[gpt2] decoding: %s\n", decode_desc.c_str());

        const auto full = gpt2_generate(
            model, prompt, opt,
            [&](int step, int id) {
                if (!quiet)
                    std::printf("[gpt2] +%d: %d\n", step + 1, id);
            });

        const int new_tokens = static_cast<int>(full.size()) - static_cast<int>(prompt.size());
        std::printf("[gpt2] generated %d new tokens (total %zu)\n", new_tokens, full.size());

        std::printf("[gpt2] full ids:");
        for (int id : full) std::printf(" %d", id);
        std::printf("\n");

        if (!out_ids.empty())
        {
            write_ids_file(out_ids, full);
            std::printf("[gpt2] wrote ids -> %s\n", out_ids.c_str());
        }

        // 整段序列（prompt + 生成）的 logits，供 Python 侧与 HF 的同一次 forward 比对
        if (!dump_logits.empty())
        {
            model.clear_kv_cache();
            mat_t<double> seq(1, static_cast<int>(full.size()));
            for (std::size_t i = 0; i < full.size(); ++i)
                seq(0, static_cast<int>(i)) = static_cast<double>(full[i]);

            const auto logits = model.forward(seq);

            weight_writer_t writer;
            writer.add("ids", seq);
            writer.add("logits", logits);
            writer.write(dump_logits);
            std::printf("[gpt2] wrote logits %dx%d -> %s\n",
                        logits.row_num(), logits.col_num(), dump_logits.c_str());
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "[gpt2] error: %s\n", e.what());
        return 1;
    }
}
