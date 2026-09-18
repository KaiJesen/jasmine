/**
 * MNIST 手写数字识别小玩具：卷积 + 编码器（全连接）的端到端训练 / 评估 / 序列化。
 *
 * 网络（全部由本库的层拼出来，反向传播也是本库的 backward）：
 *
 *     输入 [1, 28*28]
 *       conv1 1→8  5x5 pad2  → [8, 784]     → ReLU → maxpool 2x2  → [8, 196]
 *       conv2 8→16 5x5 pad2  → [16, 196]    → ReLU → maxpool 2x2  → [16, 49]
 *       flatten（reshape_view，零拷贝）      → [784, 1]
 *       fc1  784→128 → ReLU
 *       fc2  128→10  → softmax+CE（ce_loss_t）
 *
 * 训练：按样本前向/反向，用 cache_updator_t 把 batch 个样本的梯度平均后再 step()
 * （即库里现成的「梯度累加 = mini-batch」机制）。
 *
 * 数据：优先读 `--data-dir` 下的 MNIST IDX 文件（未压缩的 train/t10k 四个文件）；
 *       找不到就退化成内置的合成数字图案，保证这个 demo 在无网络/无数据时也能跑通。
 *
 * 序列化：训练结果（各层权重/偏置 + epoch/损失/精度元信息）写进一个
 *         JASMINE_WEIGHTS_V1 文件（见 jas_weight_io.hpp），可以再次 `--load` 载入继续用，
 *         也可以用 tools/ 下的 Python reader 直接解析。
 *
 * 用法：
 *     ./build/examples/mnist_conv --data-dir build/mnist --epochs 3 --save build/mnist/model.jas
 *     ./build/examples/mnist_conv --data-dir build/mnist --load build/mnist/model.jas
 *     ./build/examples/mnist_conv --synthetic          # 无数据时的冒烟测试
 */

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "jas_conv_t.hpp"
#include "jas_loss_t.hpp"
#include "jas_net_t.hpp"
#include "jas_pool_t.hpp"
#include "jas_weight_io.hpp"

using namespace jasmine;
using dmat = mat_t<double>;

/** 梯度累加器 + Adam：batch 个样本的梯度平均后再更新一次 */
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

/** 读 MNIST IDX 文件（未压缩）；图像归一化到 [0,1] */
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

/** 合成数据集：画 0..9 的简化笔画图案，用来在没有 MNIST 文件时跑通整条链路 */
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
        const int kind = digit % 5;                  // 每个数字一个粗糙图案，类别可分即可
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

// ---------------------------------------------------------------- 模型

struct mnist_cnn_t
{
    conv2d_net_t<dmat, upr_tpl> conv1{1, 8, 28, 28, 5, 5, 1, 1, 2, 2};
    relu_net_t<dmat> relu1;
    pool2d_net_t<dmat> pool1{pool_mode::max, 28, 28, 2, 2, 2, 2};
    conv2d_net_t<dmat, upr_tpl> conv2{8, 16, 14, 14, 5, 5, 1, 1, 2, 2};
    relu_net_t<dmat> relu2;
    pool2d_net_t<dmat> pool2{pool_mode::max, 14, 14, 2, 2, 2, 2};
    weight_net_t<dmat, upr_tpl> fc1{784, 128};
    relu_net_t<dmat> relu3;
    weight_net_t<dmat, upr_tpl> fc2{128, 10};
    ce_loss_t<dmat> loss;

    void init(unsigned seed)
    {
        g_random_engine.seed(seed);
        conv1.init_weight<he_gaussian_t>();
        conv2.init_weight<he_gaussian_t>();
        fc1.init_weight<he_gaussian_t>();
        fc2.init_weight<he_gaussian_t>();
        conv1.bias() = 0.0;   // 偏置从 0 起，训练更稳
        conv2.bias() = 0.0;
        fc1.bias() = 0.0;
        fc2.bias() = 0.0;
    }

    void set_lr(double lr)
    {
        conv1.set_updator(lr);
        conv2.set_updator(lr);
        fc1.set_updator(lr);
        fc2.set_updator(lr);
    }

    /** 一个样本的前向：返回 logits（[10, 1]） */
    dmat forward_one(std::vector<double> const& image)
    {
        dmat x(1, 784);
        for (int i = 0; i < 784; ++i) x(0, i) = image[static_cast<std::size_t>(i)];

        dmat h = relu1.forward(conv1.forward(x));       // [8, 784]
        h = pool1.forward(h);                           // [8, 196]
        h = relu2.forward(conv2.forward(h));            // [16, 196]
        h = pool2.forward(h);                           // [16, 49]

        auto flat = h.reshape_view(784, 1);             // 零拷贝展平
        dmat f = relu3.forward(fc1.forward(flat));      // [128, 1]
        dmat logits = fc2.forward(f);                   // [10, 1]
        loss.forward(logits);                           // 缓存 logits 供 backward/loss
        return logits;
    }

    /** 一个样本的反向（梯度累加到 updator 的 cache 里） */
    void backward_one(dmat const& label)
    {
        dmat d = loss.backward(label);                  // [10, 1]
        d = relu3.backward(fc2.backward(d));            // [128, 1]
        dmat dfc1 = fc1.backward(d);                    // [784, 1]
        auto dpool = dfc1.reshape_view(16, 49);         // 零拷贝还原成特征图
        dmat dc2 = conv2.backward(relu2.backward(pool2.backward(dpool)));
        conv1.backward(relu1.backward(pool1.backward(dc2)));
    }

    void step()
    {
        conv1.step();
        conv2.step();
        fc1.step();
        fc2.step();
    }

    int predict(std::vector<double> const& image)
    {
        const dmat logits = forward_one(image);
        int best = 0;
        for (int i = 1; i < logits.row_num(); ++i)
            if (logits(i, 0) > logits(best, 0)) best = i;
        return best;
    }
};

// ---------------------------------------------------------------- 序列化

void save_model(mnist_cnn_t const& net, std::string const& path,
                int epochs, double loss, double accuracy)
{
    weight_writer_t w;
    add_layer_params(w, "conv1", net.conv1);
    add_layer_params(w, "conv2", net.conv2);
    add_layer_params(w, "fc1", net.fc1);
    add_layer_params(w, "fc2", net.fc2);
    w.add_scalar("meta.epochs", static_cast<float>(epochs));
    w.add_scalar("meta.loss", static_cast<float>(loss));
    w.add_scalar("meta.accuracy", static_cast<float>(accuracy));
    w.add_scalar("arch.conv1.shape", static_cast<float>(28));   // 结构指纹，载入时可校验
    w.write(path);
    std::cout << "[save] " << path << "  (" << w.size() << " tensors: 4 层参数 + 元信息)\n";
}

struct train_meta_t
{
    int epochs = 0;
    double loss = 0.0;
    double accuracy = 0.0;
};

train_meta_t load_model(mnist_cnn_t& net, std::string const& path)
{
    weight_file_t wf;
    wf.load(path);
    read_layer_params(wf, "conv1", net.conv1);
    read_layer_params(wf, "conv2", net.conv2);
    read_layer_params(wf, "fc1", net.fc1);
    read_layer_params(wf, "fc2", net.fc2);

    train_meta_t meta;
    meta.epochs = static_cast<int>(wf.read_scalar<float>("meta.epochs"));
    meta.loss = wf.read_scalar<float>("meta.loss");
    meta.accuracy = wf.read_scalar<float>("meta.accuracy");
    return meta;
}

double evaluate(mnist_cnn_t& net, dataset_t const& d, std::size_t limit)
{
    const std::size_t n = std::min(limit, d.size());
    std::size_t correct = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (net.predict(d.images[i]) == d.labels[i]) ++correct;
    return n == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(n);
}

} // namespace

int main(int argc, char** argv)
{
    std::string data_dir = "build/mnist";
    std::string save_path, load_path;
    int epochs = 3, batch = 8;
    double lr = 1e-3;
    long train_limit = 6000, test_limit = 2000;
    unsigned seed = 1234;
    bool synthetic = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--data-dir") data_dir = next();
        else if (a == "--save") save_path = next();
        else if (a == "--load") load_path = next();
        else if (a == "--epochs") epochs = std::stoi(next());
        else if (a == "--batch") batch = std::stoi(next());
        else if (a == "--lr") lr = std::stod(next());
        else if (a == "--train-limit") train_limit = std::stol(next());
        else if (a == "--test-limit") test_limit = std::stol(next());
        else if (a == "--seed") seed = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--synthetic") synthetic = true;
        else
        {
            std::cout << "usage: mnist_conv [--data-dir DIR] [--epochs N] [--batch N] [--lr LR]\n"
                         "                  [--train-limit N] [--test-limit N] [--save FILE] [--load FILE]\n"
                         "                  [--synthetic] [--seed N]\n";
            return a == "--help" ? 0 : 1;
        }
    }

    dataset_t train, test;
    if (synthetic)
    {
        train = make_synthetic(static_cast<std::size_t>(std::max(1l, train_limit)), seed);
        test = make_synthetic(static_cast<std::size_t>(std::max(1l, test_limit)), seed + 1);
        std::cout << "[data] 合成数据集（--synthetic）：train=" << train.size()
                  << " test=" << test.size() << "\n";
    }
    else
    {
        try
        {
            train = load_mnist(data_dir + "/train-images-idx3-ubyte", data_dir + "/train-labels-idx1-ubyte");
            test = load_mnist(data_dir + "/t10k-images-idx3-ubyte", data_dir + "/t10k-labels-idx1-ubyte");
            std::cout << "[data] MNIST from " << data_dir << "：train=" << train.size()
                      << " test=" << test.size() << "\n";
        }
        catch (std::exception const& e)
        {
            std::cout << "[data] " << e.what() << "\n"
                      << "[data] 退化为合成数据集（用 --data-dir 指向 MNIST IDX 文件可训练真实数据）\n";
            train = make_synthetic(static_cast<std::size_t>(std::max(1l, train_limit)), seed);
            test = make_synthetic(static_cast<std::size_t>(std::max(1l, test_limit)), seed + 1);
        }
    }

    mnist_cnn_t net;
    net.init(seed);
    net.set_lr(lr);

    if (!load_path.empty())
    {
        const train_meta_t meta = load_model(net, load_path);
        std::cout << "[load] " << load_path << "：epochs=" << meta.epochs
                  << " loss=" << meta.loss << " acc=" << meta.accuracy << "\n";
    }

    std::cout << "\n[model]\n" << net.conv1.net_type(2) << "\n" << net.pool1.net_type(2) << "\n"
              << net.conv2.net_type(2) << "\n" << net.pool2.net_type(2) << "\n"
              << net.fc1.net_type(2) << "\n" << net.fc2.net_type(2) << "\n";

    const double acc_before = evaluate(net, test, static_cast<std::size_t>(test_limit));
    std::cout << "[eval] 训练前 accuracy = " << acc_before << "\n";

    std::vector<std::size_t> order(train.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::mt19937 rng(seed);

    const std::size_t n_train = std::min(static_cast<std::size_t>(train_limit), train.size());
    double last_loss = 0.0, last_acc = acc_before;
    int done_epochs = 0;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(order.begin(), order.end(), rng);
        double epoch_loss = 0.0;
        std::size_t seen = 0, correct = 0;
        int in_batch = 0;

        for (std::size_t k = 0; k < n_train; ++k)
        {
            const std::size_t idx = order[k];
            const dmat logits = net.forward_one(train.images[idx]);
            dmat label(1, 1, {static_cast<double>(train.labels[idx])});
            epoch_loss += net.loss.loss(label);
            // 直接用在手的 logits 判对错，避免多跑一次前向（每样本一次前向的开销翻倍）
            int pred = 0;
            for (int c = 1; c < logits.row_num(); ++c)
                if (logits(c, 0) > logits(pred, 0)) pred = c;
            if (pred == train.labels[idx]) ++correct;
            net.backward_one(label);
            ++seen;
            ++in_batch;
            if (in_batch >= batch)
            {
                net.step();
                in_batch = 0;
            }
            if (k + 1 == n_train && in_batch > 0) net.step();   // 收尾的半个 batch
        }

        last_loss = seen ? epoch_loss / static_cast<double>(seen) : 0.0;
        last_acc = evaluate(net, test, static_cast<std::size_t>(test_limit));
        done_epochs = epoch + 1;
        std::cout << "[epoch " << done_epochs << "] train_loss=" << last_loss
                  << " train_acc=" << (seen ? static_cast<double>(correct) / seen : 0.0)
                  << " test_acc=" << last_acc << std::endl;
    }

    if (!save_path.empty())
        save_model(net, save_path, done_epochs, last_loss, last_acc);

    // 序列化自检：存下来的权重重新载入后，预测必须完全一致
    if (!save_path.empty())
    {
        mnist_cnn_t reloaded;
        reloaded.init(seed + 999);                 // 先用不同权重，确保下面比较的是"载入"结果
        const train_meta_t meta = load_model(reloaded, save_path);
        const double acc_reloaded = evaluate(reloaded, test, static_cast<std::size_t>(test_limit));
        std::cout << "[check] 重新载入后 test_acc=" << acc_reloaded
                  << "（文件里记录 " << meta.accuracy << "）\n";
        if (std::abs(acc_reloaded - last_acc) > 1e-12)
        {
            std::cerr << "[check] FAILED: 载入后的精度与保存前不一致\n";
            return 2;
        }
        std::cout << "[check] OK: 保存/载入往返一致\n";
    }
    return 0;
}
