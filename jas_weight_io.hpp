#ifndef __JAS_WEIGHT_IO_HPP__
#define __JAS_WEIGHT_IO_HPP__

#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "jas_mat_t.hpp"
#include "jas_gpt2_t.hpp"
#include "jas_llama_t.hpp"

namespace jasmine {

/**
 * jasmine 权重文件：单文件「文本索引 + 二进制 float32 数据」。
 *
 * 格式：
 *     JASMINE_WEIGHTS_V1\n
 *     f32\n
 *     <count>\n
 *     <name> <rows> <cols> <byte_offset>\n      (count 行，offset 相对数据区起点)
 *     \n                                        (空行分隔)
 *     <二进制 float32 数据，行优先>
 *
 * 所有 tensor 均为 jasmine 原生布局（已由导出脚本完成转置 / c_attn 拆分 / 权重绑定），
 * 因此这里只做「按名字取 + shape 校验 + 类型转换」。
 *
 * 之所以自定义格式而不直接解析 safetensors / JSON：避免 C++ 侧引入 JSON 依赖，
 * 把布局换算集中在 Python 导出脚本中，便于对照 HuggingFace 实现。
 */
class weight_file_t
{
public:
    struct entry_t
    {
        int rows = 0;
        int cols = 0;
        std::size_t offset = 0;
    };

private:
    std::vector<char> m_blob;
    std::size_t m_data_start = 0;
    std::unordered_map<std::string, entry_t> m_index;

    static std::string trim_cr(std::string s)
    {
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n'))
            s.pop_back();
        return s;
    }

    void parse()
    {
        std::size_t pos = 0;
        auto next_line = [&]() -> std::string {
            if (pos > m_blob.size())
                throw std::runtime_error("weight_file_t: truncated header");
            std::size_t end = pos;
            while (end < m_blob.size() && m_blob[end] != '\n')
                ++end;
            std::string line(m_blob.data() + pos, (end < m_blob.size() ? end : m_blob.size()) - pos);
            pos = (end < m_blob.size()) ? end + 1 : end;
            return trim_cr(line);
        };

        const std::string magic = next_line();
        if (magic != "JASMINE_WEIGHTS_V1")
            throw std::runtime_error("weight_file_t: bad magic '" + magic + "'");

        const std::string dtype = next_line();
        if (dtype != "f32")
            throw std::runtime_error("weight_file_t: unsupported dtype '" + dtype + "'");

        const int count = std::stoi(next_line());
        if (count < 0)
            throw std::runtime_error("weight_file_t: negative tensor count");

        for (int i = 0; i < count; ++i)
        {
            std::istringstream ls(next_line());
            std::string name;
            entry_t e;
            ls >> name >> e.rows >> e.cols >> e.offset;
            if (!ls || name.empty())
                throw std::runtime_error("weight_file_t: malformed manifest entry " + std::to_string(i));
            if (e.rows <= 0 || e.cols <= 0)
                throw std::runtime_error("weight_file_t: non-positive shape for '" + name + "'");
            m_index[name] = e;
        }

        next_line();    // 空行：索引区与数据区的分界
        m_data_start = pos;
    }

public:
    void load(std::string const& path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("weight_file_t: cannot open " + path);
        m_blob.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        if (m_blob.empty())
            throw std::runtime_error("weight_file_t: empty file " + path);
        parse();
    }

    bool has(std::string const& name) const { return m_index.count(name) > 0; }
    std::size_t size() const { return m_index.size(); }

    /**
     * 释放底层数据区（保留索引）。大模型（如 TinyLlama 的 float32 权重约 4.4GB）在
     * read_into 全部读完之后没必要继续占内存 —— 加载完成后调用即可把这部分还给系统。
     * 释放后再调 read_into 会抛异常。
     */
    void release()
    {
        std::vector<char>().swap(m_blob);
        m_data_start = 0;
    }

    /** 数据区是否已被 release() 释放 */
    bool released() const { return m_blob.empty(); }

    entry_t const& entry(std::string const& name) const
    {
        auto it = m_index.find(name);
        if (it == m_index.end())
            throw std::runtime_error("weight_file_t: missing tensor '" + name + "'");
        return it->second;
    }

    /** 读入 float32 tensor 并按 dst 元素类型转换；要求 shape 完全一致 */
    template <typename val_type>
    void read_into(std::string const& name, mat_t<val_type>& dst) const
    {
        const auto& e = entry(name);
        if (dst.row_num() != e.rows || dst.col_num() != e.cols)
        {
            throw std::runtime_error(
                "weight_file_t: shape mismatch for '" + name + "': file " +
                std::to_string(e.rows) + "x" + std::to_string(e.cols) + ", target " +
                std::to_string(dst.row_num()) + "x" + std::to_string(dst.col_num()));
        }
        const std::size_t n = static_cast<std::size_t>(e.rows) * static_cast<std::size_t>(e.cols);
        const std::size_t bytes = n * sizeof(float);
        if (m_data_start + e.offset + bytes > m_blob.size())
            throw std::runtime_error("weight_file_t: data out of range for '" + name + "'");

        const char* base = m_blob.data() + m_data_start + e.offset;
        val_type* out = dst.data();
        for (std::size_t i = 0; i < n; ++i)
        {
            float v;
            // memcpy 而非 reinterpret_cast：避免未对齐访问的 UB
            std::memcpy(&v, base + i * sizeof(float), sizeof(float));
            out[i] = static_cast<val_type>(v);
        }
    }

    /** 读一个标量配置项（存为 1×1 tensor） */
    template <typename val_type>
    val_type read_scalar(std::string const& name) const
    {
        mat_t<val_type> tmp(1, 1);
        read_into(name, tmp);
        return tmp(0, 0);
    }
};

/**
 * 权重文件写出器：与 weight_file_t 对称。
 *
 * 用途：把 C++ 侧的中间结果（如 logits）按同一格式 dump，Python 侧即可复用同一个 reader
 * 解析，不必另立格式；方便与参考实现做数值比对（见 tools/verify_gpt2.py）。
 */
class weight_writer_t
{
public:
    struct record_t
    {
        std::string name;
        int rows = 0;
        int cols = 0;
        std::size_t offset = 0;     // 相对数据区起点，单位：字节
    };

    template <typename val_type>
    void add(std::string const& name, mat_t<val_type> const& m)
    {
        if (m.row_num() <= 0 || m.col_num() <= 0)
            throw std::runtime_error("weight_writer_t: non-positive shape for '" + name + "'");

        record_t rec;
        rec.name = name;
        rec.rows = m.row_num();
        rec.cols = m.col_num();
        rec.offset = m_data.size() * sizeof(float);     // 字节偏移
        m_records.push_back(rec);

        const std::size_t n = static_cast<std::size_t>(rec.rows) * static_cast<std::size_t>(rec.cols);
        m_data.reserve(m_data.size() + n);
        for (int i = 0; i < m.row_num(); ++i)
            for (int j = 0; j < m.col_num(); ++j)
                m_data.push_back(static_cast<float>(m(i, j)));
    }

    void write(std::string const& path) const
    {
        std::ofstream out(path, std::ios::binary);
        if (!out)
            throw std::runtime_error("weight_writer_t: cannot write " + path);

        std::ostringstream header;
        header << "JASMINE_WEIGHTS_V1\n"
               << "f32\n"
               << m_records.size() << "\n";
        for (const auto& r : m_records)
            header << r.name << " " << r.rows << " " << r.cols << " " << r.offset << "\n";
        header << "\n";

        const std::string head = header.str();
        out.write(head.data(), static_cast<std::streamsize>(head.size()));
        if (!m_data.empty())
            out.write(reinterpret_cast<const char*>(m_data.data()),
                      static_cast<std::streamsize>(m_data.size() * sizeof(float)));
        if (!out)
            throw std::runtime_error("weight_writer_t: write failed for " + path);
    }

    /** 存一个标量（1x1 tensor），用于把训练元信息（epoch/损失/精度）一起写进同一个文件 */
    template <typename val_type>
    void add_scalar(std::string const& name, val_type value)
    {
        mat_t<val_type> m(1, 1, {static_cast<val_type>(value)});
        add(name, m);
    }

    std::size_t size() const { return m_records.size(); }

private:
    std::vector<record_t> m_records;
    std::vector<float> m_data;
};

/**
 * 按 `<prefix>.weight` / `<prefix>.bias` 的命名存取「权重 + 偏置」型层的参数。
 *
 * conv2d_net_t / weight_net_t / output_proj_net_t 都是这个形状约定，所以这几个模板
 * 就是模型序列化的通用黏合层：训练完把每层按名字写进一个 JASMINE_WEIGHTS_V1 文件，
 * 之后用同名读回来即可（格式与 GPT-2 / LLaMA 权重完全相同，Python 侧也能直接解析）。
 */
template <typename layer_type>
void add_layer_params(weight_writer_t& w, std::string const& prefix, layer_type const& layer)
{
    w.add(prefix + ".weight", layer.weight());
    w.add(prefix + ".bias", layer.bias());
}

template <typename layer_type>
void read_layer_params(weight_file_t const& wf, std::string const& prefix, layer_type& layer)
{
    wf.read_into(prefix + ".weight", layer.weight());
    wf.read_into(prefix + ".bias", layer.bias());
}

/**
 * 按 GPT-2 张量命名把权重写入模型。
 *
 * 命名约定（均为 jasmine 原生布局，转换已在 export_gpt2.py 完成）：
 *   wte.weight                    [d_model, vocab]
 *   wpe.weight                    [d_model, n_pos]
 *   h.{i}.ln_1.weight / .bias     [d_model, 1]
 *   h.{i}.attn.q.weight / .bias   [d_model, d_model] / [d_model, 1]
 *   h.{i}.attn.k.weight / .bias
 *   h.{i}.attn.v.weight / .bias
 *   h.{i}.attn.out.weight / .bias [d_model, d_model] / [d_model, 1]
 *   h.{i}.ln_2.weight / .bias
 *   h.{i}.mlp.fc.weight / .bias   [d_ff, d_model] / [d_ff, 1]
 *   h.{i}.mlp.proj.weight / .bias [d_model, d_ff] / [d_model, 1]
 *   ln_f.weight / .bias           [d_model, 1]
 *
 * lm_head 与 wte 绑定（tie_word_embeddings），无需单独张量。
 */
template <typename model_type>
void load_gpt2(model_type& model, weight_file_t const& wf)
{
    wf.read_into("wte.weight", model.wte().weight());
    wf.read_into("wpe.weight", model.wpe().weight());

    for (int i = 0; i < model.n_layers(); ++i)
    {
        const std::string p = "h." + std::to_string(i) + ".";
        auto& attn = model.attn(i);

        wf.read_into(p + "ln_1.weight", model.ln_1(i).gama());
        wf.read_into(p + "ln_1.bias", model.ln_1(i).beta());

        wf.read_into(p + "attn.q.weight", attn.q_proj().weight());
        wf.read_into(p + "attn.q.bias", attn.q_proj().bias());
        wf.read_into(p + "attn.k.weight", attn.k_proj().weight());
        wf.read_into(p + "attn.k.bias", attn.k_proj().bias());
        wf.read_into(p + "attn.v.weight", attn.v_proj().weight());
        wf.read_into(p + "attn.v.bias", attn.v_proj().bias());
        wf.read_into(p + "attn.out.weight", attn.out_proj().weight());
        wf.read_into(p + "attn.out.bias", attn.out_proj().bias());

        wf.read_into(p + "ln_2.weight", model.ln_2(i).gama());
        wf.read_into(p + "ln_2.bias", model.ln_2(i).beta());

        wf.read_into(p + "mlp.fc.weight", model.mlp_fc(i).weight());
        wf.read_into(p + "mlp.fc.bias", model.mlp_fc(i).bias());
        wf.read_into(p + "mlp.proj.weight", model.mlp_proj(i).weight());
        wf.read_into(p + "mlp.proj.bias", model.mlp_proj(i).bias());
    }

    wf.read_into("ln_f.weight", model.ln_f().gama());
    wf.read_into("ln_f.bias", model.ln_f().beta());

    // GPT-2：lm_head.weight = wte.weight^T，且 lm_head 无 bias
    model.tie_word_embeddings();
}

template <typename model_type>
void load_gpt2(model_type& model, std::string const& path)
{
    weight_file_t wf;
    wf.load(path);
    load_gpt2(model, wf);
}

/** 从权重文件里的 cfg.* 标量读出结构参数 */
struct gpt2_config_t
{
    int n_layers = 1;
    int n_heads = 1;
    int d_model = 1;
    int d_ff = 0;
    int vocab = 1;
    int n_pos = 1;
};

inline gpt2_config_t read_gpt2_config(weight_file_t const& wf)
{
    gpt2_config_t cfg;
    cfg.n_layers = wf.read_scalar<double>("cfg.n_layers");
    cfg.n_heads = wf.read_scalar<double>("cfg.n_heads");
    cfg.d_model = wf.read_scalar<double>("cfg.d_model");
    cfg.d_ff = wf.read_scalar<double>("cfg.d_ff");
    cfg.vocab = wf.read_scalar<double>("cfg.vocab");
    cfg.n_pos = wf.read_scalar<double>("cfg.n_pos");
    return cfg;
}

/**
 * 按 LLaMA 张量命名把权重写入模型。
 *
 * 命名约定（均为 jasmine 原生布局；LLaMA 的 Linear 权重本就是 [out, in]，
 * 导出时**不需要转置**，这是与 GPT-2 的 Conv1D 最大的不同）：
 *   wte.weight                    [d_model, vocab]（embed_tokens 转置而来）
 *   h.{i}.ln_1.weight             [d_model, 1]  （input_layernorm，RMSNorm 无 bias）
 *   h.{i}.attn.q.weight           [d_model, d_model]
 *   h.{i}.attn.k.weight           [n_kv_heads*d_head, d_model]   ← GQA 天生窄
 *   h.{i}.attn.v.weight           [n_kv_heads*d_head, d_model]
 *   h.{i}.attn.out.weight         [d_model, d_model]
 *   h.{i}.ln_2.weight             [d_model, 1]  （post_attention_layernorm）
 *   h.{i}.mlp.gate.weight         [d_ff, d_model]
 *   h.{i}.mlp.up.weight           [d_ff, d_model]
 *   h.{i}.mlp.down.weight         [d_model, d_ff]
 *   ln_f.weight                   [d_model, 1]  （model.norm）
 *   lm_head.weight                [vocab, d_model]（仅在不绑定权重时存在）
 *
 * 没有任何 .bias 张量（LLaMA 系无线性层偏置）；模型侧对应 bias 由
 * llama_model_t::zero_all_biases() 置零。
 */
template <typename model_type>
void load_llama(model_type& model, weight_file_t const& wf)
{
    wf.read_into("wte.weight", model.wte().weight());

    for (int i = 0; i < model.n_layers(); ++i)
    {
        const std::string p = "h." + std::to_string(i) + ".";
        auto& attn = model.attn(i);

        wf.read_into(p + "ln_1.weight", model.ln_1(i).gama());
        wf.read_into(p + "attn.q.weight", attn.q_proj().weight());
        wf.read_into(p + "attn.k.weight", attn.k_proj().weight());
        wf.read_into(p + "attn.v.weight", attn.v_proj().weight());
        wf.read_into(p + "attn.out.weight", attn.out_proj().weight());

        wf.read_into(p + "ln_2.weight", model.ln_2(i).gama());
        wf.read_into(p + "mlp.gate.weight", model.mlp_gate(i).weight());
        wf.read_into(p + "mlp.up.weight", model.mlp_up(i).weight());
        wf.read_into(p + "mlp.down.weight", model.mlp_down(i).weight());
    }

    wf.read_into("ln_f.weight", model.ln_f().gama());

    // 绑定权重的变体不导出 lm_head.weight，导出侧已在文件中省略（见 export_llama.py）
    if (wf.has("lm_head.weight"))
        wf.read_into("lm_head.weight", model.lm_head().weight());
    else
        model.tie_word_embeddings();

    // LLaMA 没有线性层 bias；置零是加载流程的一部分，漏掉会让 logits 整体偏移
    model.finalize_after_load();
}

template <typename model_type>
void load_llama(model_type& model, std::string const& path)
{
    weight_file_t wf;
    wf.load(path);
    load_llama(model, wf);
}

/** 从权重文件里的 cfg.* 标量读出 LLaMA 结构参数 */
struct llama_config_t
{
    int n_layers = 1;
    int n_heads = 1;
    int n_kv_heads = 0;     // 0 => 等于 n_heads
    int d_model = 1;
    int d_ff = 0;
    int vocab = 1;
    int n_pos = 1;
    double rms_eps = 1e-5;
    double rope_theta = 10000.0;
    bool tied = false;
};

inline llama_config_t read_llama_config(weight_file_t const& wf)
{
    llama_config_t cfg;
    cfg.n_layers = wf.read_scalar<double>("cfg.n_layers");
    cfg.n_heads = wf.read_scalar<double>("cfg.n_heads");
    cfg.n_kv_heads = wf.has("cfg.n_kv_heads")
        ? static_cast<int>(wf.read_scalar<double>("cfg.n_kv_heads")) : 0;
    cfg.d_model = wf.read_scalar<double>("cfg.d_model");
    cfg.d_ff = wf.read_scalar<double>("cfg.d_ff");
    cfg.vocab = wf.read_scalar<double>("cfg.vocab");
    cfg.n_pos = wf.read_scalar<double>("cfg.n_pos");
    if (wf.has("cfg.rms_eps"))
        cfg.rms_eps = wf.read_scalar<double>("cfg.rms_eps");
    if (wf.has("cfg.rope_theta"))
        cfg.rope_theta = wf.read_scalar<double>("cfg.rope_theta");
    if (wf.has("cfg.tie_word_embeddings"))
        cfg.tied = wf.read_scalar<double>("cfg.tie_word_embeddings") > 0.5;
    return cfg;
}

} // namespace jasmine
#endif
