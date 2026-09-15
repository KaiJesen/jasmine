#ifndef __JAS_TRANSFORMER_CE_DEMO_HPP__
#define __JAS_TRANSFORMER_CE_DEMO_HPP__

#include <iomanip>
#include <iostream>
#include <vector>

#include "jas_mat_t.hpp"
#include "jas_mat_view_t.hpp"
#include "jas_mat_utility.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_embedding_t.hpp"
#include "jas_transformer_t.hpp"
#include "jas_net_t.hpp"
#include "jas_loss_t.hpp"

namespace jasmine {

/**
 * 离散 token + embedding + CE demo harness。
 * 任务：小词表上的 copy（src ids → tgt ids），与 MSE 连续向量 demo 并列。
 *
 * 特殊 id：PAD=0, SOS=1, EOS=2；其余为内容 token。
 * 链路：src/tgt embedding → transformer → output_proj → ce_loss（infer 跳过 CE）。
 */
class ce_transformer_demo_t
{
public:
    using val_type = double;
    template <typename T>
    using upr_t = cache_updator_t<T, nadam_t>;

    static constexpr int PAD = 0;
    static constexpr int SOS = 1;
    static constexpr int EOS = 2;

    static constexpr int en_layers = 2;
    static constexpr int de_layers = 2;
    static constexpr int head_num = 2;
    static constexpr int d_model = 16;
    static constexpr int vocab = 16; // 0..15，内容 token 从 3 起

    using dec_net_type = complex_net_builder_t<val_type>
        ::template push_back_updatable<transformer_base_t, base_upr_tpl>
        ::template push_back_updatable<output_proj_net_t, upr_t>
        ::template push_back_staticnet<ce_loss_t>
        ::type;

private:
    embedding_net_t<mat_t<val_type>, upr_t> m_src_emb;
    embedding_net_t<mat_t<val_type>, upr_t> m_tgt_emb;
    dec_net_type m_net;

    static void print_tui(int current, int total, double lr, double loss)
    {
        const int bar_width = 60;
        float progress = static_cast<float>(current) / total;
        int pos = static_cast<int>(bar_width * progress);
        std::cout << "\r\033[A";
        std::cout << "\rTraining: [";
        for (int i = 0; i < bar_width; ++i)
            std::cout << (i < pos ? "#" : (i == pos ? ">" : " "));
        std::cout << "] " << std::fixed << std::setw(2) << (progress * 100.0) << "%";
        std::cout << "\n          lr: " << std::scientific << lr
                  << "  loss: " << std::scientific << loss << "        ";
        std::cout.flush();
    }

    static int argmax_col(mat_t<val_type> const& logits, int t)
    {
        int best = 0;
        val_type v = logits(0, t);
        for (int i = 1; i < logits.row_num(); ++i)
        {
            if (logits(i, t) > v)
            {
                v = logits(i, t);
                best = i;
            }
        }
        return best;
    }

    auto& tf() { return m_net.template get<0>(); }
    auto& proj() { return m_net.template get<1>(); }
    auto& ce() { return m_net.template get<2>(); }

public:
    ce_transformer_demo_t()
        : m_src_emb(vocab, d_model), m_tgt_emb(vocab, d_model)
    {
    }

    void init(val_type lr)
    {
        m_src_emb.reinit({vocab, d_model});
        m_tgt_emb.reinit({vocab, d_model});
        tf().set_param(en_layers, de_layers, head_num, d_model, d_model * 4);
        proj().reinit({d_model, vocab});
        m_src_emb.init_weight<xavier_gaussian_t>();
        m_tgt_emb.init_weight<xavier_gaussian_t>();
        m_net.init_weight<xavier_gaussian_t>();
        m_src_emb.set_updator(lr);
        m_tgt_emb.set_updator(lr);
        m_net.set_updator(lr);
        ce().set_ignore_index(PAD);
    }

    void set_lr(val_type lr)
    {
        m_src_emb.set_lr(lr);
        m_tgt_emb.set_lr(lr);
        m_net.set_lr(lr);
    }

    /** teacher forcing：dec_in = [SOS, y...], labels = [y..., EOS] */
    static mat_t<val_type> make_decoder_input(mat_t<val_type> const& label_ids)
    {
        // label_ids: 1×T content tokens
        const int T = label_ids.col_num();
        mat_t<val_type> dec(1, T + 1);
        dec(0, 0) = static_cast<val_type>(SOS);
        for (int t = 0; t < T; ++t)
            dec(0, t + 1) = label_ids(0, t);
        return dec;
    }

    static mat_t<val_type> make_decoder_labels(mat_t<val_type> const& label_ids)
    {
        const int T = label_ids.col_num();
        mat_t<val_type> lab(1, T + 1);
        for (int t = 0; t < T; ++t)
            lab(0, t) = label_ids(0, t);
        lab(0, T) = static_cast<val_type>(EOS);
        return lab;
    }

    void train(mat_t<val_type> const& src_ids, mat_t<val_type> const& label_ids, int train_times)
    {
        auto dec_in = make_decoder_input(label_ids);
        auto labels = make_decoder_labels(label_ids);

        std::cout << "src ids: " << src_ids << "\n";
        std::cout << "decoder input ids: " << dec_in << "\n";
        std::cout << "labels: " << labels << "\n";
        std::cout << m_net.net_type() << "\n\n";

        cosine_annealing_decay lr_decay(train_times, 50, 3e-3, 1e-5, 0.2, 2.0, 0.8);
        int print_step = train_times < 100 ? 1 : train_times / 100;

        for (int i = 0; i < train_times; ++i)
        {
            set_lr(static_cast<val_type>(lr_decay.get_lr()));

            auto enc_h = m_src_emb.forward(src_ids);
            tf().encoder_forward(enc_h);

            auto dec_h = m_tgt_emb.forward(dec_in);
            m_net.forward(dec_h);
            auto d_dec = m_net.backward(labels);
            m_tgt_emb.backward(d_dec);
            m_src_emb.backward(tf().encoder_input_delta());

            m_net.step();
            m_tgt_emb.step();
            m_src_emb.step();
            lr_decay.step();

            if (i % print_step == 0)
                print_tui(i / print_step, 100, lr_decay.get_lr(), ce().loss(labels));
        }

        std::cout << "\nteacher-forcing loss: " << ce().loss(labels) << "\n";
        auto enc_h = m_src_emb.forward(src_ids);
        tf().encoder_forward(enc_h);
        auto dec_h = m_tgt_emb.forward(dec_in);
        auto logits = m_net.forward(dec_h);
        std::cout << "TF argmax: [";
        for (int t = 0; t < logits.col_num(); ++t)
        {
            if (t)
                std::cout << ", ";
            std::cout << argmax_col(logits, t);
        }
        std::cout << "]\n";
    }

    /** 自回归生成：返回内容 token（不含 SOS/EOS），1×L */
    mat_t<val_type> predict(mat_t<val_type> const& src_ids, int max_len = 8)
    {
        auto enc_h = m_src_emb.forward(src_ids);
        tf().encoder_forward(enc_h);
        m_net.infer_prepare(max_len + 2);

        mat_t<val_type> hist(1, max_len + 2);
        hist(0, 0) = static_cast<val_type>(SOS);
        int len = 1;
        std::vector<int> out_ids;

        while (len < max_len + 1)
        {
            mat_t<val_type> tok(1, 1);
            tok(0, 0) = hist(0, len - 1);
            auto h = m_tgt_emb.forward(tok);
            auto logits = m_net.infer(h); // V×1，跳过 CE
            int id = argmax_col(logits, 0);
            if (id == EOS)
                break;
            if (id != PAD && id != SOS)
                out_ids.push_back(id);
            hist(0, len) = static_cast<val_type>(id);
            ++len;
        }

        mat_t<val_type> ret(1, static_cast<int>(out_ids.size()));
        for (size_t i = 0; i < out_ids.size(); ++i)
            ret(0, static_cast<int>(i)) = static_cast<val_type>(out_ids[i]);
        std::cout << "predict ids: " << ret << "\n";
        return ret;
    }
};

} // namespace jasmine
#endif
