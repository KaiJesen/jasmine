/**
 * MNIST 手写数字识别小玩具：用 jasmine 的静态层堆叠拼出三种结构，在同一份数据/超参/种子下对比。
 *
 *   cnn2（标准 CNN）
 *     conv 1→8 5x5 pad2 → ReLU → maxpool 2x2
 *     conv 8→16 5x5 pad2 → ReLU → maxpool 2x2
 *     flatten → fc 784→128 → ReLU → fc 128→10 → CE
 *
 *   cnn1（卷积 + MLP encoder）
 *     conv 1→8 5x5 pad2 → ReLU → maxpool 2x2
 *     flatten → fc 1568→128 → ReLU → fc 128→10 → CE
 *
 *   trf（卷积 stem + Transformer encoder，即 conv→relu→pool→展平成 token 序列→encoder→CE）
 *     conv 1→8 5x5 pad2 → ReLU → maxpool 2x2            → [8, 196]
 *     patch embedding: fc 8→d_model（逐位置投影，一个空间位置 = 一个 token）→ [d_model, 196]
 *     encoder_t：bidirectional Transformer encoder（MHA+LayerNorm+FFN+残差 × n_layers）
 *     mean_pool：T 个 token 取平均 → [d_model, 1]
 *     fc d_model→10 → CE
 *
 * 注意这里的 encoder 是**库里的 encoder_t**（Transformer encoder，双向自注意力），
 * 拼进链里时是 `push_back_impl<encoder_t<double, upr_tpl>>`：它本身就是由
 * residual / MHA / LayerNorm / FFN 堆出来的，只是入口形状要用 `set_param(层数, 头数, d_model, d_ff, seq_len)` 给。
 *
 * 三条链都用 complex_net_builder_t 静态堆叠，训练/评估/序列化代码完全共用：
 *
 *     const dmat logits = net.forward(x);      // 链式前向（末端 CE 透传并缓存 logits）
 *     const double l   = net.back().loss(y);   // 末端损失层
 *     net.backward(y);                         // 链式反向
 *     net.step();                              // 各层 updator（梯度累加器）落地
 *
 * 用法：
 *     ./build/examples/mnist_conv --data-dir build/mnist --arch both --epochs 3 \
 *         --train-limit 6000 --batch 16 --lr 2e-3 --save build/mnist/cmp
 *     ./build/examples/mnist_conv --synthetic --arch all --epochs 1     # 冒烟
 *     ./build/examples/mnist_conv --data-dir build/mnist --arch trf --load build/mnist/trf.jas --epochs 0
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "jas_conv_t.hpp"
#include "jas_loss_t.hpp"
#include "jas_net_t.hpp"
#include "jas_pool_t.hpp"
#include "jas_transformer_kernel_t.hpp"
#include "jas_weight_io.hpp"

using namespace jasmine;
using dmat = mat_t<double>;

/** 梯度累加器 + AdamW：batch 个样本的梯度平均后再更新一次（库里现成的 mini-batch 机制），
 *  AdamW 的解耦权重衰减用来做正则（见 jas_updator_t.hpp） */
template <typename val_type>
using upr_tpl = cache_updator_t<val_type, adamw_t>;

namespace
{

// ---------------------------------------------------------------- 数据

struct dataset_t
{
    std::vector<std::vector<double>> images;    // 每张 784 像素，归一化到 [0,1]
    std::vector<int> labels;
    std::size_t size() const { return labels.size(); }
};

std::uint32_t read_be_u32(std::istream& in)
{
    unsigned char b[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("mnist: failed to read 4-byte big-endian header");
    return (static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16)
         | (static_cast<std::uint32_t>(b[2]) << 8) | static_cast<std::uint32_t>(b[3]);
}

dataset_t load_mnist(std::string const& image_path, std::string const& label_path)
{
    std::ifstream imgs(image_path, std::ios::binary), lbls(label_path, std::ios::binary);
    if (!imgs || !lbls)
        throw std::runtime_error("mnist: cannot open " + image_path + " / " + label_path);

    const std::uint32_t img_magic = read_be_u32(imgs), n_images = read_be_u32(imgs);
    const std::uint32_t rows = read_be_u32(imgs), cols = read_be_u32(imgs);
    const std::uint32_t lbl_magic = read_be_u32(lbls), n_labels = read_be_u32(lbls);
    if (img_magic != 2051u || lbl_magic != 2049u)
        throw std::runtime_error("mnist: bad magic (expect 2051/2049) — 文件是否未解压？");
    if (n_images != n_labels) throw std::runtime_error("mnist: image/label count mismatch");
    if (rows != 28 || cols != 28) throw std::runtime_error("mnist: expected 28x28 images");

    dataset_t d;
    d.images.resize(n_images);
    d.labels.resize(n_labels);
    std::vector<unsigned char> buf(static_cast<std::size_t>(rows) * cols);
    for (std::uint32_t i = 0; i < n_images; ++i)
    {
        imgs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        d.images[i].resize(buf.size());
        for (std::size_t p = 0; p < buf.size(); ++p)
            d.images[i][p] = static_cast<double>(buf[p]) / 255.0;
        unsigned char label = 0;
        lbls.read(reinterpret_cast<char*>(&label), 1);
        d.labels[i] = static_cast<int>(label);
    }
    return d;
}

dataset_t make_synthetic(std::size_t count, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> noise(-0.08, 0.08);
    std::uniform_int_distribution<int> pick(0, 9);
    dataset_t d;
    d.images.resize(count);
    d.labels.resize(count);
    for (std::size_t s = 0; s < count; ++s)
    {
        const int digit = pick(rng);
        std::vector<double> img(784, 0.0);
        auto put = [&](int r, int c, double v) {
            if (r >= 0 && r < 28 && c >= 0 && c < 28) img[static_cast<std::size_t>(r) * 28 + c] = v;
        };
        const int kind = digit % 5;
        for (int i = 6; i < 22; ++i)
        {
            const int j = 6 + kind * 3;
            put(i, j, 1.0);
            if (digit >= 5) put(i, 27 - j, 1.0);
            if (kind == 2) put(j, i, 1.0);
            if (kind == 4) put(i, i, 1.0);
        }
        for (auto& v : img) v = std::max(0.0, v + noise(rng));
        d.images[s] = std::move(img);
        d.labels[s] = digit;
    }
    return d;
}

// ---------------------------------------------------------------- 三种结构（静态层堆叠）

enum class arch_t { cnn2, cnn1, trf };

/** 结构名 ↔ 枚举 */
std::string arch_name(arch_t a)
{
    switch (a)
    {
        case arch_t::cnn2: return "cnn2";
        case arch_t::cnn1: return "cnn1";
        default: return "trf";
    }
}

/** Transformer 变体的超参（写成常量便于对照） */
// Transformer 变体的规模：选成与 cnn2（约 10.5 万参数）相当
//   conv1+conv2 = 3424、proj 17d、每层 8d²+11d、分类头 11d → 3 层 d=64 时约 10.4 万
constexpr int kTrfDModel = 64;
constexpr int kTrfHeads = 4;      // d_head = 16（偶数 → RoPE 可用）
constexpr int kTrfLayers = 3;
constexpr int kTrfDff = 128;      // = 2 * d_model
constexpr int kTrfTokens = 7 * 7;                // 两次 2x2 池化后：每个空间位置一个 token → 49
constexpr int kTrfInChannels = 16;               // patch embedding 的输入通道数（第二次卷积的输出）
constexpr int kHidden = 128;                     // CNN 变体的 encoder 隐层

/** conv→relu→pool→[conv→relu→pool]→flatten→fc→relu→fc→ce */
template <bool two_conv>
using cnn_chain_t = std::conditional_t<
    two_conv,
    complex_net_builder_t<double>
        ::push_back_updatable<conv2d_net_t, upr_tpl>   // 0
        ::push_back_staticnet<relu_net_t>              // 1
        ::push_back_staticnet<pool2d_net_t>            // 2
        ::push_back_updatable<conv2d_net_t, upr_tpl>   // 3
        ::push_back_staticnet<relu_net_t>              // 4
        ::push_back_staticnet<pool2d_net_t>            // 5
        ::push_back_staticnet<flatten_net_t>           // 6
        ::push_back_updatable<weight_net_t, upr_tpl>   // 7
        ::push_back_staticnet<relu_net_t>              // 8
        ::push_back_staticnet<dropout_net_t>           // 9 dropout（正则）
        ::push_back_updatable<weight_net_t, upr_tpl>   // 10
        ::push_back_staticnet<ce_loss_t>               // 11
        ::type,
    complex_net_builder_t<double>
        ::push_back_updatable<conv2d_net_t, upr_tpl>   // 0
        ::push_back_staticnet<relu_net_t>              // 1
        ::push_back_staticnet<pool2d_net_t>            // 2
        ::push_back_staticnet<flatten_net_t>           // 3
        ::push_back_updatable<weight_net_t, upr_tpl>   // 4
        ::push_back_staticnet<relu_net_t>              // 5
        ::push_back_staticnet<dropout_net_t>           // 6 dropout（正则）
        ::push_back_updatable<weight_net_t, upr_tpl>   // 7
        ::push_back_staticnet<ce_loss_t>               // 8
        ::type>;

/** conv→relu→pool→(逐位置投影成 token)→Transformer encoder→mean pool→fc→ce */
using trf_chain_t = complex_net_builder_t<double>
    ::push_back_updatable<conv2d_net_t, upr_tpl>       // 0 conv 1→8
    ::push_back_staticnet<relu_net_t>                  // 1
    ::push_back_staticnet<pool2d_net_t>                // 2 28→14   → [8, 196]
    ::push_back_updatable<conv2d_net_t, upr_tpl>       // 3 conv 8→16
    ::push_back_staticnet<relu_net_t>                  // 4
    ::push_back_staticnet<pool2d_net_t>                // 5 14→7    → [16, 49]（49 个 token）
    ::push_back_updatable<weight_net_t, upr_tpl>       // 6 patch embedding: 16 → d_model，逐位置
    ::push_back_updatable<cls_token_net_t, upr_tpl>    // 7 拼一个可学习的 CLS 向量
    ::push_back_impl<encoder_t<double, upr_tpl>>       // 8 Transformer encoder（双向）
    ::push_back_staticnet<take_token_net_t>            // 9 取 CLS 那一列 → [d_model, 1]
    ::push_back_staticnet<dropout_net_t>               // 10 dropout（正则）
    ::push_back_updatable<weight_net_t, upr_tpl>       // 11 分类头
    ::push_back_staticnet<ce_loss_t>                   // 12
    ::type;

template <arch_t A>
using mnist_net_t = std::conditional_t<A == arch_t::cnn2, cnn_chain_t<true>,
                     std::conditional_t<A == arch_t::cnn1, cnn_chain_t<false>, trf_chain_t>>;

/** 链里各层的下标（三种结构不同，集中在这里） */
template <arch_t A>
struct idx_t
{
    static constexpr int conv1 = 0;
    static constexpr int pool1 = 2;
    static constexpr int conv2 = 3;                    // cnn2
    static constexpr int pool2 = 5;                    // cnn2
    static constexpr int flatten = (A == arch_t::cnn2) ? 6 : 3;
    static constexpr int proj = 6;                     // trf: patch embedding
    static constexpr int cls = 7;                      // trf: 可学习 CLS 向量
    static constexpr int encoder = 8;                  // trf
    static constexpr int taketoken = 9;                // trf
    static constexpr int fc1 = (A == arch_t::cnn2) ? 7 : 4;    // cnn 的 encoder 第一层
    static constexpr int dropout = (A == arch_t::cnn2) ? 9 : ((A == arch_t::cnn1) ? 6 : 10);
    static constexpr int fc2 = (A == arch_t::cnn2) ? 10 : ((A == arch_t::cnn1) ? 7 : 11);
    static constexpr int loss = (A == arch_t::cnn2) ? 11 : ((A == arch_t::cnn1) ? 8 : 12);
};

/** cnn 变体的 flatten 后特征数 */
template <arch_t A>
constexpr int features_of()
{
    if constexpr (A == arch_t::cnn2) return 16 * 7 * 7;
    else return 8 * 14 * 14;
}

template <arch_t A>
mnist_net_t<A> make_net(unsigned seed, double lr, int hidden, double dropout_p, double weight_decay)
{
    using idx = idx_t<A>;
    mnist_net_t<A> net;

    // 卷积 stem（三种结构共用）
    net.template get<idx::conv1>().set_param(1, 8, 28, 28, 5, 5, 1, 1, 2, 2);      // → [8, 784]
    net.template get<idx::pool1>().set_param(pool_mode::max, 28, 28, 2, 2, 2, 2);  // → [8, 196]

    if constexpr (A == arch_t::cnn2)
    {
        net.template get<idx::conv2>().set_param(8, 16, 14, 14, 5, 5, 1, 1, 2, 2);      // → [16, 196]
        net.template get<idx::pool2>().set_param(pool_mode::max, 14, 14, 2, 2, 2, 2);   // → [16, 49]
        net.template get<idx::flatten>().set_param(16, 49);
        net.reinit(std::vector<int>{features_of<A>(), hidden, 10});                    // 只作用于两个 fc
    }
    else if constexpr (A == arch_t::cnn1)
    {
        net.template get<idx::flatten>().set_param(8, 196);
        net.reinit(std::vector<int>{features_of<A>(), hidden, 10});
    }
    else
    {
        // 第二个卷积 + 第二次池化：28→14→7，token 数 196→49（自注意力开销降 16 倍）
        net.template get<idx::conv2>().set_param(8, kTrfInChannels, 14, 14, 5, 5, 1, 1, 2, 2);  // → [16, 196]
        net.template get<idx::pool2>().set_param(pool_mode::max, 14, 14, 2, 2, 2, 2);           // → [16, 49]
        // patch embedding：把 16 个通道的每个空间位置投影成 d_model 维 token
        net.template get<idx::proj>().reinit(std::vector<int>{kTrfInChannels, kTrfDModel});
        // CLS 向量：d_model 维，拼在原 49 个 token 之前
        net.template get<idx::cls>().set_param(kTrfDModel);
        // Transformer encoder：层数 / 头数 / d_model / d_ff / 序列长度（token 数 + 1 个 CLS）
        net.template get<idx::encoder>().set_param(kTrfLayers, kTrfHeads, kTrfDModel, kTrfDff, kTrfTokens + 1);
        net.template get<idx::taketoken>().set_param(0);      // 取 CLS 那一列
        // 逐层挂上 RoPE（mat_mha_t::bind_rope 从注册表按 d_head 取共享条目），
        // 让注意力知道 token 的顺序；不挂的话就是「一袋 patch」，位置信息只能靠卷积 stem。
        for (int i = 0; i < kTrfLayers; ++i)
            net.template get<idx::encoder>().get_mha(i).bind_rope(kTrfTokens + 1);
        // 分类头 d_model → 10
        net.template get<idx::fc2>().reinit(std::vector<int>{kTrfDModel, 10});
    }

    g_random_engine.seed(seed);
    net.template init_weight<he_gaussian_t>();
    net.template get<idx::conv1>().bias() = 0.0;
    if constexpr (A != arch_t::cnn1)
        net.template get<idx::conv2>().bias() = 0.0;
    net.template get<idx::fc2>().bias() = 0.0;
    if constexpr (A != arch_t::trf)
        net.template get<idx::fc1>().bias() = 0.0;

    net.template get<idx::dropout>().set_param(static_cast<typename mnist_net_t<A>::val_type>(dropout_p));
    net.set_updator(static_cast<typename mnist_net_t<A>::val_type>(lr),
                    typename mnist_net_t<A>::val_type(0.9), 0.999, 1e-8, weight_decay);
    return net;
}

/**
 * 「CNN 变体在给定隐层宽度下的参数量」的解析式（与 param_count 的口径一致）：
 * conv1 (+conv2) + fc1(features x h + h) + fc2(h x 10 + 10)。
 * match 模式用它反解出与 Transformer 变体参数量最接近的宽度。
 */
std::size_t cnn_params_for_hidden(int hidden, bool two_conv)
{
    const std::size_t conv = two_conv
        ? (1 * 8 * 5 * 5 + 8) + (8 * 16 * 5 * 5 + 16)
        : (1 * 8 * 5 * 5 + 8);
    const std::size_t features = two_conv ? 16 * 7 * 7 : 8 * 14 * 14;
    return conv + features * hidden + hidden + static_cast<std::size_t>(hidden) * 10 + 10;
}

/** 求解：让 cnn2 的参数量最接近参考值（= trf 的参数量）的隐层宽度 */
int hidden_matching(std::size_t target, bool two_conv)
{
    int best_h = 1;
    std::size_t best_gap = static_cast<std::size_t>(-1);
    for (int h = 1; h <= 4096; ++h)
    {
        const std::size_t p = cnn_params_for_hidden(h, two_conv);
        const std::size_t gap = p > target ? p - target : target - p;
        if (gap < best_gap) { best_gap = gap; best_h = h; }
        if (p > target) break;                 // 参数随宽度单调增，越过后就没必要继续
    }
    return best_h;
}

template <arch_t A>
std::size_t param_count(mnist_net_t<A> const& net)
{
    using idx = idx_t<A>;
    std::size_t n = 0;
    auto add = [&n](auto const& layer) {
        n += static_cast<std::size_t>(layer.weight().row_num()) * layer.weight().col_num();
        n += static_cast<std::size_t>(layer.bias().row_num()) * layer.bias().col_num();
    };
    add(net.template get<idx::conv1>());
    if constexpr (A != arch_t::cnn1) add(net.template get<idx::conv2>());
    add(net.template get<idx::fc2>());
    if constexpr (A != arch_t::trf)
    {
        add(net.template get<idx::fc1>());
    }
    else
    {
        add(net.template get<idx::proj>());
        // encoder 内部的参数（MHA 投影 + FFN + LayerNorm）也数进去
        const int d = kTrfDModel, ff = kTrfDff;
        const std::size_t per_layer = 4 * d * d + 4 * d + d * ff + ff + ff * d + d + 2 * 2 * d;
        n += static_cast<std::size_t>(kTrfLayers) * per_layer;
        n += static_cast<std::size_t>(d);        // CLS 向量
    }
    return n;
}

// ---------------------------------------------------------------- 序列化（三种结构共用）

template <arch_t A>
void save_net(mnist_net_t<A> const& net, std::string const& path, int epochs, double loss, double acc)
{
    using idx = idx_t<A>;
    weight_writer_t w;
    add_layer_params(w, "conv1", net.template get<idx::conv1>());
    if constexpr (A != arch_t::cnn1) add_layer_params(w, "conv2", net.template get<idx::conv2>());
    if constexpr (A == arch_t::trf)
    {
        add_layer_params(w, "proj", net.template get<idx::proj>());
        w.add("cls.token", net.template get<idx::cls>().token());     // CLS 向量单独存
    }
    if constexpr (A != arch_t::trf) add_layer_params(w, "fc1", net.template get<idx::fc1>());
    add_layer_params(w, "fc2", net.template get<idx::fc2>());
    w.add_scalar("meta.epochs", static_cast<float>(epochs));
    w.add_scalar("meta.loss", static_cast<float>(loss));
    w.add_scalar("meta.accuracy", static_cast<float>(acc));
    w.add_scalar("meta.arch", static_cast<float>(static_cast<int>(A)));
    w.write(path);
}

struct train_meta_t { int epochs = 0; double loss = 0.0; double accuracy = 0.0; };

template <arch_t A>
train_meta_t load_net(mnist_net_t<A>& net, std::string const& path)
{
    using idx = idx_t<A>;
    weight_file_t wf;
    wf.load(path);
    read_layer_params(wf, "conv1", net.template get<idx::conv1>());
    if constexpr (A != arch_t::cnn1) read_layer_params(wf, "conv2", net.template get<idx::conv2>());
    if constexpr (A == arch_t::trf)
    {
        read_layer_params(wf, "proj", net.template get<idx::proj>());
        wf.read_into("cls.token", net.template get<idx::cls>().token());
    }
    if constexpr (A != arch_t::trf) read_layer_params(wf, "fc1", net.template get<idx::fc1>());
    read_layer_params(wf, "fc2", net.template get<idx::fc2>());
    train_meta_t meta;
    meta.epochs = static_cast<int>(wf.read_scalar<float>("meta.epochs"));
    meta.loss = wf.read_scalar<float>("meta.loss");
    meta.accuracy = wf.read_scalar<float>("meta.accuracy");
    return meta;
}

// ---------------------------------------------------------------- 训练 / 评估

struct args_t
{
    std::string data_dir = "build/mnist";
    std::string save_path, load_path;
    std::string arch = "both";                 // cnn2 | cnn1 | trf | both | all
    int epochs = 3, batch = 16;
    int hidden = 128;                          // CNN 变体 encoder 的隐层宽度（--hidden，match 模式会自动求解）
    double lr = 2e-3;
    double dropout = 0.2;                      // 分类头前的 dropout 概率（0 = 关闭）
    double weight_decay = 0.01;                // AdamW 的解耦权重衰减
    std::string scheduler = "cosine";          // cosine（余弦退火+热重启）| fixed（固定 lr）
    long train_limit = 6000, test_limit = 10000;
    unsigned seed = 1234;
    bool synthetic = false;
};

struct result_t
{
    std::string name;
    std::size_t params = 0;
    int epochs = 0;
    double loss = 0.0, train_acc = 0.0, test_acc = 0.0, seconds = 0.0;
    bool ok = true;
};

int argmax_col(dmat const& m)
{
    int best = 0;
    for (int i = 1; i < m.row_num(); ++i)
        if (m(i, 0) > m(best, 0)) best = i;
    return best;
}

template <arch_t A>
double evaluate(mnist_net_t<A>& net, dataset_t const& d, std::size_t limit)
{
    net.template get<idx_t<A>::dropout>().set_enabled(false);   // 评估必须关掉 dropout
    const std::size_t n = std::min(limit, d.size());
    std::size_t correct = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        dmat x(1, 784);
        for (int p = 0; p < 784; ++p) x(0, p) = d.images[i][static_cast<std::size_t>(p)];
        const dmat logits = net.forward(x);        // 只前向；末端 CE 是透传层，不参与梯度
        if (argmax_col(logits) == d.labels[i]) ++correct;
    }
    net.template get<idx_t<A>::dropout>().set_enabled(true);    // 训练继续用 dropout
    return n == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(n);
}

template <arch_t A>
result_t run_arch(dataset_t const& train, dataset_t const& test, args_t const& a,
                  std::string const& load_path, std::string const& save_path)
{
    result_t r;
    r.name = arch_name(A);
    const auto t0 = std::chrono::steady_clock::now();

    auto net = make_net<A>(a.seed, a.lr, a.hidden, a.dropout, a.weight_decay);
    r.params = param_count<A>(net);

    if (!load_path.empty())
    {
        const train_meta_t meta = load_net<A>(net, load_path);
        std::cout << "[" << r.name << "] 载入 " << load_path << "：epochs=" << meta.epochs
                  << " loss=" << meta.loss << " acc=" << meta.accuracy << "\n";
    }

    const std::size_t test_n = std::min(static_cast<std::size_t>(a.test_limit), test.size());
    r.test_acc = evaluate<A>(net, test, test_n);
    std::cout << "[" << r.name << "] 训练前 test_acc=" << r.test_acc << "\n";

    const std::size_t n_train = std::min(static_cast<std::size_t>(a.train_limit), train.size());
    std::vector<std::size_t> order(train.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::mt19937 rng(a.seed);

    // 学习率调度：余弦退火 + 热重启（库自带的 cosine_annealing_decay），按「mini-batch 步」推进。
    // init_decay_steps 取总步数的一半 → 训练中途恰好经历一次热重启（之后周期 ×2）。
    const int steps_per_epoch = static_cast<int>((n_train + static_cast<std::size_t>(a.batch) - 1)
                                                / static_cast<std::size_t>(a.batch));
    const int total_steps = std::max(1, steps_per_epoch * std::max(0, a.epochs));
    cosine_annealing_decay sched(/*epoch_max=*/std::max(total_steps, 1),
                                 /*init_decay_steps=*/std::max(1, total_steps / 2),
                                 /*max_lr=*/a.lr,
                                 /*min_lr=*/a.lr * 0.05,
                                 /*warmup_rate=*/0.1,
                                 /*T_multiplier=*/2.0);
    if (a.scheduler != "cosine" && a.scheduler != "fixed")
        throw std::runtime_error("unknown --scheduler '" + a.scheduler + "' (expect cosine | fixed)");
    double lr_now = a.lr;

    for (int epoch = 0; epoch < a.epochs; ++epoch)
    {
        std::shuffle(order.begin(), order.end(), rng);
        double epoch_loss = 0.0;
        std::size_t seen = 0, correct = 0;
        int in_batch = 0;

        for (std::size_t k = 0; k < n_train; ++k)
        {
            const std::size_t idx = order[k];
            dmat x(1, 784);
            for (int p = 0; p < 784; ++p) x(0, p) = train.images[idx][static_cast<std::size_t>(p)];
            dmat label(1, 1, {static_cast<double>(train.labels[idx])});

            const dmat logits = net.forward(x);
            epoch_loss += net.back().loss(label);
            if (argmax_col(logits) == train.labels[idx]) ++correct;

            net.backward(label);
            ++seen;
            ++in_batch;
            if (in_batch >= a.batch || k + 1 == n_train)
            {
                if (a.scheduler == "cosine")
                {
                    sched.step();                        // 先推进再取：跳过预热里 lr=0 的第 0 步
                    lr_now = sched.get_lr();
                    net.set_lr(static_cast<typename mnist_net_t<A>::val_type>(lr_now));
                }
                net.step();
                in_batch = 0;
            }
        }

        r.epochs = epoch + 1;
        r.loss = seen ? epoch_loss / static_cast<double>(seen) : 0.0;
        r.train_acc = seen ? static_cast<double>(correct) / static_cast<double>(seen) : 0.0;
        r.test_acc = evaluate<A>(net, test, test_n);
        std::cout << "[" << r.name << "][epoch " << r.epochs << "] train_loss=" << r.loss
                  << " train_acc=" << r.train_acc << " test_acc=" << r.test_acc
                  << " lr=" << lr_now << " cycle=" << sched.get_current_cycle() << std::endl;
    }

    if (!save_path.empty())
    {
        save_net<A>(net, save_path, r.epochs, r.loss, r.test_acc);
        mnist_net_t<A> reloaded = make_net<A>(a.seed + 999, a.lr, a.hidden, a.dropout, a.weight_decay);
        load_net<A>(reloaded, save_path);
        const double acc = evaluate<A>(reloaded, test, test_n);
        std::cout << "[" << r.name << "] [check] 重新载入 test_acc=" << acc;
        if (std::abs(acc - r.test_acc) > 1e-12) { std::cout << "  FAILED（与保存前不一致）\n"; r.ok = false; }
        else std::cout << "  OK\n";
    }

    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

void print_comparison(std::vector<result_t> const& rs)
{
    std::cout << "\n================ 对比（同一份数据 / 超参 / 随机种子）================\n";
    std::cout << std::left << std::setw(8) << "arch" << std::right << std::setw(10) << "params"
              << std::setw(8) << "epochs" << std::setw(12) << "train_loss"
              << std::setw(12) << "train_acc" << std::setw(12) << "test_acc"
              << std::setw(10) << "seconds" << "\n";
    for (const auto& r : rs)
        std::cout << std::left << std::setw(8) << r.name << std::right << std::setw(10) << r.params
                  << std::setw(8) << r.epochs << std::setw(12) << r.loss
                  << std::setw(12) << r.train_acc << std::setw(12) << r.test_acc
                  << std::setw(10) << r.seconds << (r.ok ? "" : "   <-- 往返自检失败") << "\n";
    std::cout << "====================================================================\n";
    if (rs.size() == 2)
        std::cout << "test_acc 差值（" << rs[1].name << " - " << rs[0].name << "）= "
                  << std::showpos << (rs[1].test_acc - rs[0].test_acc) << std::noshowpos << "\n";
}

std::string sub_path(std::string const& base, std::string const& name, bool per_arch)
{
    if (base.empty() || !per_arch) return base;
    return base + "." + name + ".jas";
}

} // namespace

int main(int argc, char** argv)
{
    args_t a;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (arg == "--data-dir") a.data_dir = next();
        else if (arg == "--save") a.save_path = next();
        else if (arg == "--load") a.load_path = next();
        else if (arg == "--arch") a.arch = next();
        else if (arg == "--epochs") a.epochs = std::stoi(next());
        else if (arg == "--batch") a.batch = std::stoi(next());
        else if (arg == "--lr") a.lr = std::stod(next());
        else if (arg == "--hidden") a.hidden = std::stoi(next());
        else if (arg == "--dropout") a.dropout = std::stod(next());
        else if (arg == "--weight-decay") a.weight_decay = std::stod(next());
        else if (arg == "--scheduler") a.scheduler = next();
        else if (arg == "--train-limit") a.train_limit = std::stol(next());
        else if (arg == "--test-limit") a.test_limit = std::stol(next());
        else if (arg == "--seed") a.seed = static_cast<unsigned>(std::stoul(next()));
        else if (arg == "--synthetic") a.synthetic = true;
        else
        {
            std::cout << "usage: mnist_conv [--arch cnn2|cnn1|trf|both|all|match] [--data-dir DIR] [--epochs N]\n"
                         "                  [--batch N] [--lr LR] [--hidden N] [--dropout P] [--train-limit N]\n"
                         "                  [--scheduler cosine|fixed] [--weight-decay WD] [--test-limit N]\n"
                         "                  [--save FILE] [--load FILE] [--synthetic] [--seed N]\n"
                         "  cnn2  = conv->relu->pool->conv->relu->pool->flatten->fc->relu->fc->ce（标准 CNN）\n"
                         "  cnn1  = conv->relu->pool->flatten->fc->relu->fc->ce\n"
                         "  trf   = conv->relu->pool->(patch embedding)->Transformer encoder->mean pool->fc->ce\n"
                         "  both  = cnn2 vs trf（默认）    all = 三种都跑\n"
                         "  match = 把 cnn2 的隐层宽度自动裁到与 trf 参数量相当，再做等容量对比\n"
                         "          （--hidden N 可手动指定 CNN 宽度）\n";
            return (arg == "--help") ? 0 : 1;
        }
    }

    // 要跑哪些结构
    std::vector<arch_t> archs;
    if (a.arch == "cnn2") archs = {arch_t::cnn2};
    else if (a.arch == "cnn1") archs = {arch_t::cnn1};
    else if (a.arch == "trf") archs = {arch_t::trf};
    else if (a.arch == "both") archs = {arch_t::cnn2, arch_t::trf};
    else if (a.arch == "all") archs = {arch_t::cnn2, arch_t::cnn1, arch_t::trf};
    else if (a.arch == "match") archs = {arch_t::cnn2, arch_t::trf};
    else { std::cerr << "unknown --arch '" << a.arch << "'\n"; return 1; }
    const bool per_arch_files = archs.size() > 1;

    dataset_t train, test;
    if (a.synthetic)
    {
        train = make_synthetic(static_cast<std::size_t>(std::max(1l, std::min(a.train_limit, 1000l))), a.seed);
        test = make_synthetic(static_cast<std::size_t>(std::max(1l, std::min(a.test_limit, 500l))), a.seed + 1);
        std::cout << "[data] 合成数据集（--synthetic）：train=" << train.size() << " test=" << test.size() << "\n";
    }
    else
    {
        try
        {
            train = load_mnist(a.data_dir + "/train-images-idx3-ubyte", a.data_dir + "/train-labels-idx1-ubyte");
            test = load_mnist(a.data_dir + "/t10k-images-idx3-ubyte", a.data_dir + "/t10k-labels-idx1-ubyte");
            std::cout << "[data] MNIST from " << a.data_dir << "：train=" << train.size()
                      << " test=" << test.size() << "（官方划分，两集不重叠）\n";
        }
        catch (std::exception const& e)
        {
            std::cout << "[data] " << e.what() << "\n[data] 退化为合成数据集\n";
            train = make_synthetic(static_cast<std::size_t>(std::max(1l, std::min(a.train_limit, 1000l))), a.seed);
            test = make_synthetic(static_cast<std::size_t>(std::max(1l, std::min(a.test_limit, 500l))), a.seed + 1);
        }
    }

    if (a.arch == "match")
    {
        // 等容量对比：先算出 Transformer 变体的参数量，再反解 CNN 的隐层宽度
        const std::size_t trf_params = param_count<arch_t::trf>(make_net<arch_t::trf>(a.seed, a.lr, a.hidden, a.dropout, a.weight_decay));
        a.hidden = hidden_matching(trf_params, /*two_conv=*/true);
        std::cout << "[match] trf 参数量 = " << trf_params << "；把 cnn2 隐层裁到 " << a.hidden
                  << "（参数量 " << cnn_params_for_hidden(a.hidden, true) << "）做等容量对比\n";
    }

    std::cout << "[train] epochs=" << a.epochs << " batch=" << a.batch << " lr=" << a.lr
              << " train_limit=" << std::min(static_cast<std::size_t>(a.train_limit), train.size())
              << " test_limit=" << std::min(static_cast<std::size_t>(a.test_limit), test.size())
              << " seed=" << a.seed << "\n\n";

    std::vector<result_t> results;
    bool ok = true;
    for (arch_t arch : archs)
    {
        const std::string lp = sub_path(a.load_path, arch_name(arch), per_arch_files);
        const std::string sp = sub_path(a.save_path, arch_name(arch), per_arch_files);
        if (!lp.empty() && !std::ifstream(lp).good())
        {
            std::cerr << "[" << arch_name(arch) << "] --load 文件不存在：" << lp << "\n";
            return 1;
        }
        result_t r;
        if (arch == arch_t::cnn2) r = run_arch<arch_t::cnn2>(train, test, a, lp, sp);
        else if (arch == arch_t::cnn1) r = run_arch<arch_t::cnn1>(train, test, a, lp, sp);
        else r = run_arch<arch_t::trf>(train, test, a, lp, sp);
        ok = ok && r.ok;
        results.push_back(r);
    }

    print_comparison(results);
    return ok ? 0 : 2;
}
