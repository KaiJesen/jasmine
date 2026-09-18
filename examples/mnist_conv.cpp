/**
 * MNIST 手写数字识别小玩具：**用 jasmine 的静态层堆叠（complex_net_builder_t）**拼出
 * 两种结构，在同一份数据/超参下训练并对比。
 *
 * 两条链（都由 complex_net_t 持有，forward / backward / step / init_weight 全部走链）：
 *
 *   conv2（"标准 CNN"，两次下采样）
 *     conv 1→8 5x5 pad2 → ReLU → maxpool 2x2
 *     conv 8→16 5x5 pad2 → ReLU → maxpool 2x2
 *     flatten → fc 784→128 → ReLU → fc 128→10 → CE
 *
 *   conv1（conv + relu + pool + flatten + encoder + ce，一次下采样、特征更多）
 *     conv 1→8 5x5 pad2 → ReLU → maxpool 2x2
 *     flatten → fc 8*14*14=1568→128 → ReLU → fc 128→10 → CE
 *
 * 两者除了前几层的形状配置不同，训练/评估/序列化用的是**同一份代码**：
 *
 *     const dmat logits = net.forward(x);      // 链式前向（末端 CE 透传并缓存 logits）
 *     const double l   = net.back().loss(y);   // 取末端损失层的 loss
 *     net.backward(y);                         // 链式反向（net_backward 逆序回传）
 *     net.step();                              // 各层 updator（梯度累加器）落地
 *
 * 数据：`--data-dir` 下的官方 MNIST IDX 文件（train 60000 / t10k 10000，两集不重叠）；
 *       找不到就退化成内置合成图案，保证无网络时也能跑通。
 *
 * 序列化：每层参数按名字 + 元信息写进一个 JASMINE_WEIGHTS_V1 文件（见 jas_weight_io.hpp），
 *         可 `--load` 回来继续训练或直接评估，并用往返自检确认一致。
 *
 * 用法：
 *     # 两条结构各训 3 个 epoch，在同一测试集上对比
 *     ./build/examples/mnist_conv --data-dir build/mnist --epochs 3 --train-limit 6000 \
 *         --batch 16 --lr 2e-3 --arch both --save build/mnist/toy
 *
 *     # 只跑标准 CNN，并把结果存成一个文件
 *     ./build/examples/mnist_conv --data-dir build/mnist --arch conv2 --save build/mnist/cnn.jas
 *
 *     # 载入已保存的模型（不训练）评估
 *     ./build/examples/mnist_conv --data-dir build/mnist --arch conv2 --load build/mnist/cnn.jas --epochs 0
 *
 *     # 无数据/无网络时的冒烟测试
 *     ./build/examples/mnist_conv --synthetic --arch both --epochs 1
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "jas_conv_t.hpp"
#include "jas_loss_t.hpp"
#include "jas_net_t.hpp"
#include "jas_pool_t.hpp"
#include "jas_weight_io.hpp"

using namespace jasmine;
using dmat = mat_t<double>;

/** 梯度累加器 + Adam：batch 个样本的梯度平均后再更新一次（库里现成的 mini-batch 机制） */
template <typename val_type>
using upr_tpl = cache_updator_t<val_type, adam_t>;

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
    if (!in)
        throw std::runtime_error("mnist: failed to read 4-byte big-endian header");
    return (static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16)
         | (static_cast<std::uint32_t>(b[2]) << 8) | static_cast<std::uint32_t>(b[3]);
}

dataset_t load_mnist(std::string const& image_path, std::string const& label_path)
{
    std::ifstream imgs(image_path, std::ios::binary);
    std::ifstream lbls(label_path, std::ios::binary);
    if (!imgs || !lbls)
        throw std::runtime_error("mnist: cannot open " + image_path + " / " + label_path);

    const std::uint32_t img_magic = read_be_u32(imgs);
    const std::uint32_t n_images = read_be_u32(imgs);
    const std::uint32_t rows = read_be_u32(imgs);
    const std::uint32_t cols = read_be_u32(imgs);
    const std::uint32_t lbl_magic = read_be_u32(lbls);
    const std::uint32_t n_labels = read_be_u32(lbls);

    if (img_magic != 2051u || lbl_magic != 2049u)
        throw std::runtime_error("mnist: bad magic (expect 2051/2049) — 文件是否未解压？");
    if (n_images != n_labels)
        throw std::runtime_error("mnist: image/label count mismatch");
    if (rows != 28 || cols != 28)
        throw std::runtime_error("mnist: expected 28x28 images");

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

/** 合成数据集：画 0..9 的简化笔画图案，用于无数据时的冒烟测试 */
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
            if (r >= 0 && r < 28 && c >= 0 && c < 28)
                img[static_cast<std::size_t>(r) * 28 + c] = v;
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

// ---------------------------------------------------------------- 两种结构的静态层堆叠

/**
 * 把 conv / relu / pool / flatten / encoder / ce 按顺序压进 complex_net_builder_t。
 * 注意：`push_back_updatable` 用于带权重+updator 的层，`push_back_staticnet` 用于无参静态层，
 * 两种层在链里的层号（get<N>()）就按下面这个顺序数。
 */
template <bool two_conv>
using mnist_net_t = std::conditional_t<
    two_conv,
    // conv → relu → pool → conv → relu → pool → flatten → fc → relu → fc → ce
    complex_net_builder_t<double>
        ::push_back_updatable<conv2d_net_t, upr_tpl>       // 0
        ::push_back_staticnet<relu_net_t>                  // 1
        ::push_back_staticnet<pool2d_net_t>                // 2
        ::push_back_updatable<conv2d_net_t, upr_tpl>       // 3
        ::push_back_staticnet<relu_net_t>                  // 4
        ::push_back_staticnet<pool2d_net_t>                // 5
        ::push_back_staticnet<flatten_net_t>               // 6
        ::push_back_updatable<weight_net_t, upr_tpl>       // 7  (encoder 第一层)
        ::push_back_staticnet<relu_net_t>                  // 8
        ::push_back_updatable<weight_net_t, upr_tpl>       // 9  (encoder 输出层)
        ::push_back_staticnet<ce_loss_t>                   // 10 (loss)
        ::type,
    // conv → relu → pool → flatten → fc → relu → fc → ce
    complex_net_builder_t<double>
        ::push_back_updatable<conv2d_net_t, upr_tpl>       // 0
        ::push_back_staticnet<relu_net_t>                  // 1
        ::push_back_staticnet<pool2d_net_t>                // 2
        ::push_back_staticnet<flatten_net_t>               // 3
        ::push_back_updatable<weight_net_t, upr_tpl>       // 4
        ::push_back_staticnet<relu_net_t>                  // 5
        ::push_back_updatable<weight_net_t, upr_tpl>       // 6
        ::push_back_staticnet<ce_loss_t>                   // 7
        ::type>;

/** 链里各层的下标（两种结构不同，集中在这里） */
template <bool two_conv>
struct idx_t
{
    static constexpr int conv1 = 0;
    static constexpr int pool1 = 2;
    static constexpr int conv2 = 3;            // 仅 two_conv
    static constexpr int pool2 = 5;            // 仅 two_conv
    static constexpr int flatten = two_conv ? 6 : 3;
    static constexpr int fc1 = two_conv ? 7 : 4;
    static constexpr int fc2 = two_conv ? 9 : 6;
    static constexpr int loss = two_conv ? 10 : 7;
};

/** 特征数：conv2 → 16*7*7；conv1 → 8*14*14 */
template <bool two_conv>
constexpr int features_of()
{
    if constexpr (two_conv) return 16 * 7 * 7;
    else return 8 * 14 * 14;
}

template <bool two_conv>
mnist_net_t<two_conv> make_net(unsigned seed, double lr)
{
    using idx = idx_t<two_conv>;
    mnist_net_t<two_conv> net;

    // 形状配置：只有前几层不同，其余（flatten / encoder / loss）完全一致
    net.template get<idx::conv1>().set_param(1, 8, 28, 28, 5, 5, 1, 1, 2, 2);   // → [8, 784]
    net.template get<idx::pool1>().set_param(pool_mode::max, 28, 28, 2, 2, 2, 2);  // → [8, 196]
    if constexpr (two_conv)
    {
        net.template get<idx::conv2>().set_param(8, 16, 14, 14, 5, 5, 1, 1, 2, 2);  // → [16, 196]
        net.template get<idx::pool2>().set_param(pool_mode::max, 14, 14, 2, 2, 2, 2);  // → [16, 49]
        net.template get<idx::flatten>().set_param(16, 49);
    }
    else
    {
        net.template get<idx::flatten>().set_param(8, 196);
    }

    // 全连接（encoder）：reinit 只作用于链里的 weight_net_t，容器给 {in, hidden, out}
    net.reinit(std::vector<int>{features_of<two_conv>(), 128, 10});

    g_random_engine.seed(seed);
    net.template init_weight<he_gaussian_t>();
    net.template get<idx::conv1>().bias() = 0.0;      // 偏置从 0 起，训练更稳
    if constexpr (two_conv)
        net.template get<idx::conv2>().bias() = 0.0;
    net.template get<idx::fc1>().bias() = 0.0;
    net.template get<idx::fc2>().bias() = 0.0;

    net.set_updator(lr);
    return net;
}

template <bool two_conv>
std::size_t param_count(mnist_net_t<two_conv> const& net)
{
    using idx = idx_t<two_conv>;
    std::size_t n = 0;
    auto add = [&n](auto const& layer) {
        n += static_cast<std::size_t>(layer.weight().row_num()) * layer.weight().col_num();
        n += static_cast<std::size_t>(layer.bias().row_num()) * layer.bias().col_num();
    };
    add(net.template get<idx::conv1>());
    if constexpr (two_conv) add(net.template get<idx::conv2>());
    add(net.template get<idx::fc1>());
    add(net.template get<idx::fc2>());
    return n;
}

// ---------------------------------------------------------------- 序列化（两种结构共用）

template <bool two_conv>
void save_net(mnist_net_t<two_conv> const& net, std::string const& path,
              int epochs, double loss, double accuracy)
{
    using idx = idx_t<two_conv>;
    weight_writer_t w;
    add_layer_params(w, "conv1", net.template get<idx::conv1>());
    if constexpr (two_conv)
        add_layer_params(w, "conv2", net.template get<idx::conv2>());
    add_layer_params(w, "fc1", net.template get<idx::fc1>());
    add_layer_params(w, "fc2", net.template get<idx::fc2>());
    w.add_scalar("meta.epochs", static_cast<float>(epochs));
    w.add_scalar("meta.loss", static_cast<float>(loss));
    w.add_scalar("meta.accuracy", static_cast<float>(accuracy));
    w.add_scalar("meta.features", static_cast<float>(features_of<two_conv>()));
    w.write(path);
}

struct train_meta_t
{
    int epochs = 0;
    double loss = 0.0;
    double accuracy = 0.0;
};

template <bool two_conv>
train_meta_t load_net(mnist_net_t<two_conv>& net, std::string const& path)
{
    using idx = idx_t<two_conv>;
    weight_file_t wf;
    wf.load(path);
    read_layer_params(wf, "conv1", net.template get<idx::conv1>());
    if constexpr (two_conv)
        read_layer_params(wf, "conv2", net.template get<idx::conv2>());
    read_layer_params(wf, "fc1", net.template get<idx::fc1>());
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
    std::string arch = "both";              // conv2 | conv1 | both
    int epochs = 3, batch = 16;
    double lr = 2e-3;
    long train_limit = 6000, test_limit = 10000;
    unsigned seed = 1234;
    bool synthetic = false;
};

struct result_t
{
    std::string name;
    std::size_t params = 0;
    int epochs = 0;
    double loss = 0.0;
    double train_acc = 0.0;
    double test_acc = 0.0;
    double seconds = 0.0;
    bool ok = true;
};

int argmax_col(dmat const& m)
{
    int best = 0;
    for (int i = 1; i < m.row_num(); ++i)
        if (m(i, 0) > m(best, 0)) best = i;
    return best;
}

template <bool two_conv>
double evaluate(mnist_net_t<two_conv>& net, dataset_t const& d, std::size_t limit)
{
    const std::size_t n = std::min(limit, d.size());
    std::size_t correct = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        dmat x(1, 784);
        for (int p = 0; p < 784; ++p) x(0, p) = d.images[i][static_cast<std::size_t>(p)];
        // 评估只走前向：infer 会跳过链末的 loss 层（skip_on_infer），不会污染缓存
        const dmat logits = net.infer(x);
        if (argmax_col(logits) == d.labels[i]) ++correct;
    }
    return n == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(n);
}

template <bool two_conv>
result_t run_arch(std::string const& arch_name, dataset_t const& train, dataset_t const& test,
                  args_t const& a, std::string const& load_path, std::string const& save_path)
{
    result_t r;
    r.name = arch_name;
    const auto t0 = std::chrono::steady_clock::now();

    auto net = make_net<two_conv>(a.seed, a.lr);
    r.params = param_count<two_conv>(net);

    if (!load_path.empty())
    {
        const train_meta_t meta = load_net<two_conv>(net, load_path);
        std::cout << "[" << arch_name << "] 载入 " << load_path << "：epochs=" << meta.epochs
                  << " loss=" << meta.loss << " acc=" << meta.accuracy << "\n";
    }

    const std::size_t test_n = std::min(static_cast<std::size_t>(a.test_limit), test.size());
    r.test_acc = evaluate<two_conv>(net, test, test_n);
    std::cout << "[" << arch_name << "] 训练前 test_acc=" << r.test_acc << "\n";

    const std::size_t n_train = std::min(static_cast<std::size_t>(a.train_limit), train.size());
    std::vector<std::size_t> order(train.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::mt19937 rng(a.seed);

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

            const dmat logits = net.forward(x);       // 链式前向（末端 CE 透传 + 缓存 logits）
            epoch_loss += net.back().loss(label);
            if (argmax_col(logits) == train.labels[idx]) ++correct;

            net.backward(label);                      // 链式反向（CE → encoder → conv）
            ++seen;
            ++in_batch;
            if (in_batch >= a.batch || k + 1 == n_train)
            {
                net.step();                           // 梯度累加器落地（平均后更新）
                in_batch = 0;
            }
        }

        r.epochs = epoch + 1;
        r.loss = seen ? epoch_loss / static_cast<double>(seen) : 0.0;
        r.train_acc = seen ? static_cast<double>(correct) / static_cast<double>(seen) : 0.0;
        r.test_acc = evaluate<two_conv>(net, test, test_n);
        std::cout << "[" << arch_name << "][epoch " << r.epochs << "] train_loss=" << r.loss
                  << " train_acc=" << r.train_acc << " test_acc=" << r.test_acc << std::endl;
    }

    if (!save_path.empty())
    {
        save_net<two_conv>(net, save_path, r.epochs, r.loss, r.test_acc);
        // 往返自检：重新载入后必须得到同样的测试精度
        mnist_net_t<two_conv> reloaded = make_net<two_conv>(a.seed + 999, a.lr);
        load_net<two_conv>(reloaded, save_path);
        const double acc = evaluate<two_conv>(reloaded, test, test_n);
        std::cout << "[" << arch_name << "] [check] 重新载入 test_acc=" << acc;
        if (std::abs(acc - r.test_acc) > 1e-12)
        {
            std::cout << "  FAILED（与保存前不一致）\n";
            r.ok = false;
        }
        else
        {
            std::cout << "  OK\n";
        }
    }

    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

void print_comparison(std::vector<result_t> const& rs)
{
    std::cout << "\n================ 对比（同一份数据 / 超参 / 随机种子）================\n";
    std::cout << std::left << std::setw(10) << "arch" << std::right << std::setw(10) << "params"
              << std::setw(8) << "epochs" << std::setw(12) << "train_loss"
              << std::setw(12) << "train_acc" << std::setw(12) << "test_acc"
              << std::setw(10) << "seconds" << "\n";
    for (const auto& r : rs)
    {
        std::cout << std::left << std::setw(10) << r.name << std::right << std::setw(10) << r.params
                  << std::setw(8) << r.epochs << std::setw(12) << r.loss
                  << std::setw(12) << r.train_acc << std::setw(12) << r.test_acc
                  << std::setw(10) << r.seconds
                  << (r.ok ? "" : "   <-- 往返自检失败") << "\n";
    }
    std::cout << "====================================================================\n";
    if (rs.size() == 2)
    {
        const double d = rs[1].test_acc - rs[0].test_acc;
        std::cout << "test_acc 差值（" << rs[1].name << " - " << rs[0].name << "）= "
                  << std::showpos << d << std::noshowpos << "\n";
    }
}

/** 结构名 → 该结构的结果文件名（both 模式下给每个结构一个后缀） */
std::string arch_save_path(std::string const& base, std::string const& arch, bool compare_mode)
{
    if (base.empty()) return base;
    if (!compare_mode) return base;
    return base + "." + arch + ".jas";
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
        else if (arg == "--train-limit") a.train_limit = std::stol(next());
        else if (arg == "--test-limit") a.test_limit = std::stol(next());
        else if (arg == "--seed") a.seed = static_cast<unsigned>(std::stoul(next()));
        else if (arg == "--synthetic") a.synthetic = true;
        else
        {
            std::cout << "usage: mnist_conv [--arch conv2|conv1|both] [--data-dir DIR] [--epochs N]\n"
                         "                  [--batch N] [--lr LR] [--train-limit N] [--test-limit N]\n"
                         "                  [--save FILE] [--load FILE] [--synthetic] [--seed N]\n"
                         "  conv2 = conv->relu->pool->conv->relu->pool->flatten->encoder->ce（标准 CNN）\n"
                         "  conv1 = conv->relu->pool->flatten->encoder->ce\n"
                         "  both  = 两条结构在同一份数据/超参下训练并对比（默认）\n";
            return (arg == "--help") ? 0 : 1;
        }
    }
    const bool compare_mode = (a.arch == "both");
    if (a.arch != "both" && a.arch != "conv2" && a.arch != "conv1")
    {
        std::cerr << "unknown --arch '" << a.arch << "' (expect conv2 | conv1 | both)\n";
        return 1;
    }

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
            std::cout << "[data] " << e.what() << "\n"
                      << "[data] 退化为合成数据集（用 --data-dir 指向 MNIST IDX 文件可训练真实数据）\n";
            train = make_synthetic(static_cast<std::size_t>(std::max(1l, std::min(a.train_limit, 1000l))), a.seed);
            test = make_synthetic(static_cast<std::size_t>(std::max(1l, std::min(a.test_limit, 500l))), a.seed + 1);
        }
    }

    std::cout << "[train] epochs=" << a.epochs << " batch=" << a.batch << " lr=" << a.lr
              << " train_limit=" << std::min(static_cast<std::size_t>(a.train_limit), train.size())
              << " test_limit=" << std::min(static_cast<std::size_t>(a.test_limit), test.size())
              << " seed=" << a.seed << "\n\n";

    std::vector<result_t> results;
    bool ok = true;

    if (a.arch == "conv2" || compare_mode)
    {
        const std::string lp = compare_mode
            ? (a.load_path.empty() ? std::string() : arch_save_path(a.load_path, "conv2", true))
            : a.load_path;
        const std::string sp = arch_save_path(a.save_path, "conv2", compare_mode);
        if (!lp.empty() && !std::ifstream(lp).good())
        {
            std::cerr << "[conv2] --load 文件不存在：" << lp << "\n";
            return 1;
        }
        result_t r = run_arch<true>("conv2", train, test, a, lp, sp);
        ok = ok && r.ok;
        results.push_back(r);
    }
    if (a.arch == "conv1" || compare_mode)
    {
        const std::string lp = compare_mode
            ? (a.load_path.empty() ? std::string() : arch_save_path(a.load_path, "conv1", true))
            : a.load_path;
        const std::string sp = arch_save_path(a.save_path, "conv1", compare_mode);
        if (!lp.empty() && !std::ifstream(lp).good())
        {
            std::cerr << "[conv1] --load 文件不存在：" << lp << "\n";
            return 1;
        }
        result_t r = run_arch<false>("conv1", train, test, a, lp, sp);
        ok = ok && r.ok;
        results.push_back(r);
    }

    print_comparison(results);
    return ok ? 0 : 2;
}
