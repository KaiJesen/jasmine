/**
 * MNIST + DBN（Deep Belief Network）小 demo：**用静态层堆叠**把 RBM 串成 DBN。
 *
 *   RBM 0：784 → 256（无监督 CD-1 预训练）
 *   RBM 1：256 → 128（吃第 0 层的隐层概率，继续无监督预训练）
 *   分类头：128 → 10（监督微调时和上面的 RBM 一起反向传播）
 *
 * 这条链就是 `dbn_net_t<2, upr_tpl>`（`jas_rbm_t.hpp`）：2 个 RBM + weight_net + CE 全部由
 * `complex_net_builder_t` 静态堆叠而成，因此：
 *
 *     dbn.reinit({784, 256, 128, 10});       // 容器协议：每个 RBM / 分类头各消费一对数
 *     dbn_pretrain<2>(dbn, data, cd_k, epochs);   // 逐层贪心预训练（RBM 的 CD-k）
 *     dbn.forward(x) / dbn.backward(label) / dbn.step();   // 监督微调走整链反向
 *
 * 用法：
 *     ./build/examples/mnist_dbn --data-dir build/mnist --pretrain 3 --finetune 5 \
 *         --train-limit 2000 --save build/mnist/dbn.jas
 *     ./build/examples/mnist_dbn --data-dir build/mnist --load build/mnist/dbn.jas --finetune 0
 */

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "jas_rbm_t.hpp"
#include "jas_weight_io.hpp"

using namespace jasmine;
using dmat = mat_t<double>;

template <typename val_type>
using upr_tpl = cache_updator_t<val_type, adamw_t>;

namespace
{
constexpr int kVisible = 28 * 28;
constexpr int kHidden1 = 256;
constexpr int kHidden2 = 128;
constexpr int kClasses = 10;

struct dataset_t
{
    std::vector<std::vector<double>> images;   // 每张 784 个 [0,1] 像素（按 0.5 二值化后使用）
    std::vector<int> labels;
    std::size_t size() const { return labels.size(); }
};

std::uint32_t read_be_u32(std::istream& in)
{
    unsigned char b[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("mnist: truncated header");
    return (static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16)
         | (static_cast<std::uint32_t>(b[2]) << 8) | static_cast<std::uint32_t>(b[3]);
}

dataset_t load_mnist(std::string const& image_path, std::string const& label_path)
{
    std::ifstream imgs(image_path, std::ios::binary), lbls(label_path, std::ios::binary);
    if (!imgs || !lbls) throw std::runtime_error("cannot open " + image_path + " / " + label_path);
    const std::uint32_t img_magic = read_be_u32(imgs), n = read_be_u32(imgs);
    const std::uint32_t rows = read_be_u32(imgs), cols = read_be_u32(imgs);
    const std::uint32_t lbl_magic = read_be_u32(lbls), m = read_be_u32(lbls);
    if (img_magic != 2051u || lbl_magic != 2049u || n != m || rows * cols != kVisible)
        throw std::runtime_error("bad IDX header (未解压？)");

    dataset_t d;
    d.images.resize(n);
    d.labels.resize(n);
    std::vector<unsigned char> buf(kVisible);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        imgs.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
        d.images[i].resize(kVisible);
        for (std::size_t p = 0; p < kVisible; ++p)
            d.images[i][p] = buf[p] > 127 ? 1.0 : 0.0;      // RBM 是 Bernoulli 可见单元 → 二值化
        unsigned char label = 0;
        lbls.read(reinterpret_cast<char*>(&label), 1);
        d.labels[i] = static_cast<int>(label);
    }
    return d;
}

/** 把一批样本组装成 [784, B] 的矩阵（列 = 样本） */
dmat batch_matrix(dataset_t const& d, std::vector<std::size_t> const& idx, std::size_t from,
                  std::size_t count)
{
    const std::size_t n = std::min(count, idx.size() - from);
    dmat m(kVisible, n);
    for (std::size_t t = 0; t < n; ++t)
        for (int i = 0; i < kVisible; ++i)
            m(i, t) = d.images[idx[from + t]][static_cast<std::size_t>(i)];
    return m;
}

int argmax_col(dmat const& m)
{
    int best = 0;
    for (int i = 1; i < m.row_num(); ++i)
        if (m(i, 0) > m(best, 0)) best = i;
    return best;
}

template <typename net_type>
double evaluate(net_type& dbn, dataset_t const& d, std::size_t limit)
{
    const std::size_t n = std::min(limit, d.size());
    std::size_t correct = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        dmat x(kVisible, 1);
        for (int p = 0; p < kVisible; ++p) x(p, 0) = d.images[i][static_cast<std::size_t>(p)];
        const dmat logits = dbn.forward(x);
        if (argmax_col(logits) == d.labels[i]) ++correct;
    }
    return n ? static_cast<double>(correct) / static_cast<double>(n) : 0.0;
}

/** DBN 的序列化：每个 RBM 的 W/b/c + 分类头 + 元信息 */
void save_dbn(dbn_net_t<2, upr_tpl> const& dbn, std::string const& path, int epochs, double acc)
{
    weight_writer_t w;
    for (int layer = 0; layer < 2; ++layer)
    {
        const std::string p = "rbm" + std::to_string(layer) + ".";
        if (layer == 0)
        {
            w.add(p + "weight", dbn.template get<0>().weight());
            w.add(p + "visible_bias", dbn.template get<0>().visible_bias());
            w.add(p + "hidden_bias", dbn.template get<0>().hidden_bias());
        }
        else
        {
            w.add(p + "weight", dbn.template get<1>().weight());
            w.add(p + "visible_bias", dbn.template get<1>().visible_bias());
            w.add(p + "hidden_bias", dbn.template get<1>().hidden_bias());
        }
    }
    add_layer_params(w, "head", dbn.template get<2>());
    w.add_scalar("meta.epochs", epochs);
    w.add_scalar("meta.accuracy", acc);
    w.write(path);
}

void load_dbn(dbn_net_t<2, upr_tpl>& dbn, std::string const& path, int& epochs, double& acc)
{
    weight_file_t wf;
    wf.load(path);
    wf.read_into("rbm0.weight", dbn.template get<0>().weight());
    wf.read_into("rbm0.visible_bias", dbn.template get<0>().visible_bias());
    wf.read_into("rbm0.hidden_bias", dbn.template get<0>().hidden_bias());
    wf.read_into("rbm1.weight", dbn.template get<1>().weight());
    wf.read_into("rbm1.visible_bias", dbn.template get<1>().visible_bias());
    wf.read_into("rbm1.hidden_bias", dbn.template get<1>().hidden_bias());
    read_layer_params(wf, "head", dbn.template get<2>());
    epochs = static_cast<int>(wf.read_scalar<float>("meta.epochs"));
    acc = wf.read_scalar<float>("meta.accuracy");
}

} // namespace

int main(int argc, char** argv)
{
    std::string data_dir = "build/mnist", save_path, load_path;
    int pretrain_epochs = 3, finetune_epochs = 5, cd_k = 1, batch = 16;
    double lr = 1e-3, ft_lr = 1e-3;
    long train_limit = 2000, test_limit = 1000;
    unsigned seed = 1234;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--data-dir") data_dir = next();
        else if (a == "--save") save_path = next();
        else if (a == "--load") load_path = next();
        else if (a == "--pretrain") pretrain_epochs = std::stoi(next());
        else if (a == "--finetune") finetune_epochs = std::stoi(next());
        else if (a == "--cd") cd_k = std::stoi(next());
        else if (a == "--batch") batch = std::stoi(next());
        else if (a == "--lr") lr = std::stod(next());
        else if (a == "--ft-lr") ft_lr = std::stod(next());
        else if (a == "--train-limit") train_limit = std::stol(next());
        else if (a == "--test-limit") test_limit = std::stol(next());
        else if (a == "--seed") seed = static_cast<unsigned>(std::stoul(next()));
        else
        {
            std::cout << "usage: mnist_dbn [--data-dir DIR] [--pretrain N] [--finetune N] [--cd K]\\n"
                         "                 [--batch N] [--lr LR] [--ft-lr LR] [--train-limit N]\\n"
                         "                 [--test-limit N] [--save FILE] [--load FILE] [--seed N]\\n";
            return (a == "--help") ? 0 : 1;
        }
    }

    dataset_t train, test;
    try
    {
        train = load_mnist(data_dir + "/train-images-idx3-ubyte", data_dir + "/train-labels-idx1-ubyte");
        test = load_mnist(data_dir + "/t10k-images-idx3-ubyte", data_dir + "/t10k-labels-idx1-ubyte");
        std::cout << "[data] MNIST train=" << train.size() << " test=" << test.size() << "\\n";
    }
    catch (std::exception const& e)
    {
        std::cerr << "[data] " << e.what() << "\\n（先在 build/mnist 放好解压后的 IDX 文件）\\n";
        return 1;
    }

    // ---- 静态层堆叠出 DBN：RBM(784→256) → RBM(256→128) → 分类头(128→10) → CE ----
    dbn_net_t<2, upr_tpl> dbn;
    dbn.reinit(std::vector<int>{kVisible, kHidden1, kHidden2, kClasses});

    g_random_engine.seed(seed);
    dbn.template get<0>().init_weight<xavier_uniform_t>();
    dbn.template get<1>().init_weight<xavier_uniform_t>();
    dbn.set_updator(static_cast<double>(lr));
    std::cout << "[model]\\n" << dbn.net_type(2) << "\\n";

    if (!load_path.empty())
    {
        int epochs = 0;
        double acc = 0.0;
        load_dbn(dbn, load_path, epochs, acc);
        std::cout << "[load] " << load_path << "（记录 epochs=" << epochs << " acc=" << acc << "）\\n";
    }

    std::vector<std::size_t> order(train.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    const std::size_t n_train = std::min(static_cast<std::size_t>(train_limit), train.size());

    // ---- 1) 逐层贪心无监督预训练（CD-k）----
    if (pretrain_epochs > 0)
    {
        std::cout << "\\n[pretrain] 逐层 CD-" << cd_k << "，每层 " << pretrain_epochs << " 个 epoch\\n";
        for (int epoch = 0; epoch < pretrain_epochs; ++epoch)
        {
            double recon = 0.0;
            for (std::size_t b = 0; b < n_train; b += static_cast<std::size_t>(batch))
            {
                const dmat chunk = batch_matrix(train, order, b, static_cast<std::size_t>(batch));
                recon = dbn_pretrain<2>(dbn, chunk, cd_k, 1);
            }
            std::cout << "  epoch " << (epoch + 1) << " 重建误差≈" << recon << std::endl;
        }
    }

    // ---- 2) 监督微调（整链反向）----
    if (finetune_epochs > 0)
    {
        dbn.set_updator(static_cast<double>(ft_lr));
        std::cout << "\\n[finetune] " << finetune_epochs << " 个 epoch（mini-batch = " << batch << "）\\n";
        for (int epoch = 0; epoch < finetune_epochs; ++epoch)
        {
            double loss_sum = 0.0;
            std::size_t seen = 0;
            for (std::size_t b = 0; b < n_train; b += static_cast<std::size_t>(batch))
            {
                const std::size_t cnt = std::min(static_cast<std::size_t>(batch), n_train - b);
                dmat x(kVisible, cnt), label(1, cnt);
                for (std::size_t t = 0; t < cnt; ++t)
                {
                    const std::size_t sample = order[b + t];
                    for (int p = 0; p < kVisible; ++p)
                        x(p, t) = train.images[sample][static_cast<std::size_t>(p)];
                    label(0, t) = static_cast<double>(train.labels[sample]);
                }
                dbn.forward(x);
                loss_sum += dbn.back().loss(label);
                dbn.backward(label);
                dbn.step();
                seen += cnt;
            }
            const double test_acc = evaluate(dbn, test, static_cast<std::size_t>(test_limit));
            std::cout << "  epoch " << (epoch + 1) << " loss=" << (loss_sum / static_cast<double>(seen))
                      << " test_acc=" << test_acc << std::endl;
        }
    }

    const double acc = evaluate(dbn, test, static_cast<std::size_t>(test_limit));
    std::cout << "\\n[result] test_acc=" << acc << "（" << std::min(static_cast<std::size_t>(test_limit), test.size())
              << " 张测试图）\\n";

    if (!save_path.empty())
    {
        save_dbn(dbn, save_path, pretrain_epochs + finetune_epochs, acc);
        // 往返自检
        dbn_net_t<2, upr_tpl> reloaded;
        reloaded.reinit(std::vector<int>{kVisible, kHidden1, kHidden2, kClasses});
        int e = 0;
        double a = 0.0;
        load_dbn(reloaded, save_path, e, a);
        const double acc2 = evaluate(reloaded, test, static_cast<std::size_t>(test_limit));
        std::cout << "[save] " << save_path << "；重新载入 test_acc=" << acc2
                  << (std::abs(acc2 - acc) < 1e-12 ? "  OK" : "  FAILED") << "\\n";
    }
    return 0;
}
