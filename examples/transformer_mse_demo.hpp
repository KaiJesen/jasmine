#ifndef __JAS_TRANSFORMER_MSE_DEMO_HPP__
#define __JAS_TRANSFORMER_MSE_DEMO_HPP__

#include "jas_mat_t.hpp"
#include "jas_mat_view_t.hpp"
#include "jas_transformer_t.hpp"
#include "jas_transformer_kernel_t.hpp"
#include "jas_net_t.hpp"
#include <iomanip>

namespace jasmine {

/*!SECTION
 * MSE 回归 demo harness（连续向量 + 特征维 SOS/EOS）。
 * 与 examples/transformer_ce_demo.hpp（离散 token + embedding + CE）并列。
 */
using test_val_type = double;

mat_t<test_val_type> sos(int const& input_row_num)
{
    mat_t<test_val_type> ret(input_row_num + 2, 1);
    ret(input_row_num, 0) = 1.0;
    return ret;
}

mat_t<test_val_type> add_sos(const mat_t<test_val_type>& input)
{
    int row_num = input.row_num();
    int col_num = input.col_num();
    mat_t<test_val_type> new_mat(row_num + 2, col_num + 1);
    mat_view_t<mat_t<test_val_type>> origin(new_mat, 0, 1, row_num, col_num);
    origin.assign(input);
    new_mat(row_num, 0) = 1.;     // SOS标签
    return new_mat;
}

mat_t<test_val_type> add_eos(const mat_t<test_val_type>& input)
{
    int row_num = input.row_num();
    int col_num = input.col_num();
    mat_t<test_val_type> new_mat(row_num + 2, col_num + 1);
    mat_view_t<mat_t<test_val_type>> origin(new_mat, 0, 0, row_num, col_num);       // 原来的值复制
    origin.assign(input);
    new_mat(row_num + 1, col_num) = 1.;     // EOS标签
    return new_mat;
}

// 编码器不需要标志位，但是为了保持d_model一致，所以需要加2行空的标志位
mat_t<test_val_type> expand_encoder_input(const mat_t<test_val_type>& input)
{
    int row_num = input.row_num();
    int col_num = input.col_num();
    mat_t<test_val_type> new_mat(row_num + 2, col_num);
    mat_view_t<mat_t<test_val_type>> origin(new_mat, 0, 0, row_num, col_num);
    origin.assign(input);
    return new_mat;
}

inline mat_view_t<mat_t<test_val_type>> back_col(mat_t<test_val_type>& mat)
{
    return mat.view(0, mat.col_num() - 1, mat.row_num(), 1);
}

inline bool is_eos(mat_t<test_val_type>& mat)
{
    return back_col(mat)(mat.row_num() - 1, 0) > 0.8; // 判断是否为EOS标签 
}

inline void print_tui_display(int current, int total, double lr, double loss)
{
    const int bar_width = 60;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);
    
    std::cout << "\r\033[A"; // 上移一行（如果支持 ANSI）
    
    // 第一行：进度条
    std::cout << "\rTraining: [";
    for (int i = 0; i < bar_width; ++i)
    {
        std::cout << (i < pos ? "#" : (i == pos ? ">" : " "));
    }
    std::cout << "] " << std::fixed << std::setw(2) << (progress * 100) << "%";
    
    // 第二行：学习率和 loss
    std::cout << "\n          lr: " << std::scientific << lr 
              << "  loss: " << std::scientific << loss << "        ";
    
    std::cout.flush();
}

class mse_transformer_demo_t
{

template<typename test_val_type>
using test_updator_t = cache_updator_t<test_val_type, nadam_t>;

using net_type = complex_net_builder_t<test_val_type>
    ::template push_back_updatable<transformer_base_t, base_upr_tpl>
    ::template push_back_updatable<weight_net_t, base_upr_tpl>
    ::template push_back_staticnet<sigmoid_net_t>
    ::template push_back_staticnet<mse_loss_t>
    ::type
    ;


    net_type m_net;
public:

    static constexpr int en_layers = 3;
    static constexpr int de_layers = 2;
    static constexpr int head_num = 2;          // d_head = d_model/head_num = 5
    static constexpr int d_model = 10;
    static constexpr int input_dim = d_model - 2;   // 需要预留2个标志位

    auto& tf_base()
    {
        return m_net.template get<0>();
    }

    auto& ffn()
    {
        return m_net.template get<1>();
    }

    void init(test_val_type lr)
    {
        tf_base().set_param(en_layers, de_layers, head_num, d_model, d_model * 4);
        ffn().reinit({d_model, d_model});
        m_net.init_weight<xavier_gaussian_t>();
        m_net.set_updator(lr);
    }

    void set_updator(test_val_type lr)
    {
        m_net.set_updator(lr);
    }

    void set_lr(test_val_type lr)
    {
        m_net.set_lr(lr);
    }


    void train(mat_t<test_val_type> const& en_input, mat_t<test_val_type> const& /*de_input*/, mat_t<test_val_type> const& label, int train_times)
    {
        auto en_input_expand = expand_encoder_input(en_input);
        std::cout << "expanded encoder input: \n" << en_input_expand << std::endl;
        // 与 predict 对齐：decoder teacher forcing 必须是 shifted label
        // input  = [SOS, y1, y2, y3]
        // target = [y1, y2, y3, EOS]
        mat_t<test_val_type> input_sos = add_sos(label);
        std::cout << "decoder teacher input (SOS+label): \n" << input_sos << std::endl;
        mat_t<test_val_type> label_eos = add_eos(label);
        std::cout << "label with flags: \n" << label_eos << std::endl;
        int epoch_max = train_times;
        int init_decay_steps = 100;
        double max_lr = 3e-4;
        double min_lr = 1e-6;
        double warmup_rate = 0.2;
        double T_multiplier = 2.0;
        double lr_decay_rate = 0.8;
        cosine_annealing_decay lr_decay(epoch_max
            , init_decay_steps
            , max_lr
            , min_lr
            , warmup_rate
            , T_multiplier
            , lr_decay_rate);
        
        int print_step = train_times < 100 ? 1 : train_times / 100;

        std::cout << m_net.net_type() << std::endl;

        // 前一段纯 TF 学拟合；后段 scheduled sampling 对齐 predict 自回归。
        constexpr bool use_scheduled_sampling = true;
        constexpr double ss_warmup_frac = 0.3;   // 前 30% 步纯 teacher forcing
        constexpr double final_teacher_forcing = 0.1;
        // EOS 维在 MSE 里只占 1/(d_model*T)，加重其梯度，否则 AR 时很难压过 0.8 阈值
        constexpr test_val_type eos_loss_weight = 8.0;
        int seq_len = label_eos.col_num();
        int const sos_row = input_dim;
        int const eos_row = input_dim + 1;
        
        std::cout << std::endl;
        for (int i = 0; i < train_times; i++)
        {
            // 只改学习率，不要 set_updator（会重置 Nadam 动量）
            m_net.set_lr(static_cast<test_val_type>(lr_decay.get_lr()));
            // 每步重算 encoder：权重在 backward 里会更新，若只 encode 一次，
            // train/TF 评估用的是旧 memory，predict 却用新权重重算 → 首 token 就会对不上
            tf_base().encoder_forward(en_input_expand);

            mat_t<test_val_type> cur_input = input_sos.clone();
            if (use_scheduled_sampling)
            {
                double progress = static_cast<double>(i) / train_times;
                double tf_ratio = 1.0;
                if (progress > ss_warmup_frac)
                {
                    double ss_progress = (progress - ss_warmup_frac) / (1.0 - ss_warmup_frac);
                    tf_ratio = 1.0 + (final_teacher_forcing - 1.0) * ss_progress;
                }
                for (int step = 1; step < seq_len; ++step)
                {
                    auto pred = m_net.forward(cur_input.view(0, 0, d_model, step));
                    double rand_val = static_cast<double>(rand()) / RAND_MAX;
                    if (rand_val > tf_ratio)
                    {
                        // 与 predict 一致：回灌上一拍整列，并清掉 SOS 泄漏
                        cur_input.col(step).assign(pred.back_col());
                        cur_input(sos_row, step) = 0;
                    }
                }
            }

            m_net.forward(cur_input);
            // 等价于对 EOS 行梯度乘 eos_loss_weight：fake = (1-w)*pred + w*label
            mat_t<test_val_type> backward_target = label_eos.clone();
            auto const& pred = m_net.template get<3>().m_input;
            for (int j = 0; j < backward_target.col_num(); ++j)
            {
                backward_target(eos_row, j) =
                    (1 - eos_loss_weight) * pred(eos_row, j) + eos_loss_weight * label_eos(eos_row, j);
            }
            m_net.backward(backward_target);
            m_net.step();
            
            lr_decay.step();
            if (i % print_step == 0)
            {
                print_tui_display(i/print_step, 100, lr_decay.get_lr(), m_net.template get<3>().loss(label_eos));
            }
        }
        std::cout << "\nlabel: \n" << label_eos << std::endl;
        // teacher forcing 评估：与 predict 一样先用当前权重重算 encoder
        tf_base().encoder_forward(en_input_expand);
        auto teacher_out = m_net.forward(input_sos);
        std::cout << "teacher-forcing output: \n" << teacher_out << std::endl;
        std::cout << "teacher-forcing loss: " << m_net.template get<3>().loss(label_eos) << std::endl;
        // 仅 SOS 前缀的首 token，应与 teacher_out 第 0 列、predict 第 1 列一致
        auto sos_only_out = m_net.forward(input_sos.view(0, 0, d_model, 1).clone());
        std::cout << "SOS-only first output (should match TF col0): \n" << sos_only_out << std::endl;
    }

    mat_t<test_val_type> remove_flags(const mat_t<test_val_type>& mat)
    {
        return mat.view(0, 0, mat.row_num() - 2, mat.col_num()).clone();
    }

    auto predict(mat_t<test_val_type>& en_input)
    {
        auto en_input_expand = expand_encoder_input(en_input);
        tf_base().encoder_forward(en_input_expand);
        m_net.infer_prepare(16);

        mat_t<test_val_type> pred_input(input_dim, 1023);
        auto pred_input_sos = add_sos(pred_input);
        int const sos_row = input_dim;
        int len = 1;
        while (true)
        {
            // 只喂最新一列；self-attn 经 KV cache 复用历史 K/V
            auto token = pred_input_sos.view(0, len - 1, d_model, 1).clone();
            auto output = m_net.infer(token);
            pred_input_sos.col(len).assign(output.back_col());
            pred_input_sos(sos_row, len) = 0; // 生成步不应带 SOS
            ++len;
            if (is_eos(output))break;
            if (len >= 10)break;
        }
        std::cout << "final output with flags: \n" << pred_input_sos.view(0, 0, d_model, len) << std::endl;
        return remove_flags(pred_input_sos).view(0, 1, input_dim, len - 1).clone();
    }

};



} // namespace jasmine
#endif
