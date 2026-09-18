/**
 * A small MNIST digit-recognition toy: three architectures assembled with jasmine's static layer
 * stacking and compared on the same data, hyper-parameters and seed.
 *
 *   cnn2 (standard CNN)
 *     conv 1→8 5x5 pad2 → ReLU → maxpool 2x2
 *     conv 8→16 5x5 pad2 → ReLU → maxpool 2x2
 *     flatten → fc 784→128 → ReLU → fc 128→10 → CE
 *
 *   cnn1 (convolution + MLP encoder)
 *     conv 1→8 5x5 pad2 → ReLU → maxpool 2x2
 *     flatten → fc 1568→128 → ReLU → fc 128→10 → CE
 *
 *   trf (conv stem + Transformer encoder, i.e. conv->relu->pool->flatten into tokens->encoder->CE)
 *     conv 1→8 5x5 pad2 → ReLU → maxpool 2x2            → [8, 196]
 *     patch embedding: fc 8→d_model (per-position projection, one spatial position = one token)→ [d_model, 196]
 *     encoder_t: bidirectional Transformer encoder (MHA + LayerNorm + FFN + residual x n_layers)
 *     mean_pool: average the T tokens -> [d_model, 1]
 *     fc d_model→10 → CE
 *
 * Note that the encoder here is the **library's encoder_t** (a Transformer encoder with
 * bidirectional self-attention);
 * it joins the chain as `push_back_impl<encoder_t<double, mnist_conv_upr_tpl>>`, being assembled
 * from residual / MHA / LayerNorm / FFN itself; only its entry shape has to be given via
 * `set_param(layers, heads, d_model, d_ff, seq_len)`.
 *
 * All three chains use complex_net_builder_t static stacking and share the training / evaluation /
 * serialization code:
 *
 *     const dmat logits = net.forward(x);      // chained forward (the trailing CE passes through and caches the logits)
 *     const double l   = net.back().loss(y);   // the trailing loss layer
 *     net.backward(y);                         // chained backward
 *     net.step();                              // every layer's updator (gradient accumulator) is stepped
 *
 * Usage:
 *     ./build/examples/mnist_conv --data-dir build/mnist --arch both --epochs 3 \
 *         --train-limit 6000 --batch 16 --lr 2e-3 --save build/mnist/cmp
 *     ./build/examples/mnist_conv --synthetic --arch all --epochs 1     # smoke test
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

/** Gradient accumulator + AdamW: average the gradients of a batch and update once (the library's
 *  existing mini-batch mechanism);
 *  AdamW's decoupled weight decay serves as the regulariser (see jas_updator_t.hpp) */
template <typename val_type>
using mnist_conv_upr_tpl = cache_updator_t<val_type, adamw_t>;

namespace
{

// ---------------------------------------------------------------- data

struct dataset_t
{
    std::vector<std::vector<double>> images;    // 784 pixels per image, normalised to [0,1]
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
        throw std::runtime_error("mnist: bad magic (expect 2051/2049) -- is the file still gzipped?");
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

// ---------------------------------------------------------------- the three architectures (static stacking)

enum class arch_t { cnn2, cnn1, trf };

/** Architecture name <-> enum */
std::string arch_name(arch_t a)
{
    switch (a)
    {
        case arch_t::cnn2: return "cnn2";
        case arch_t::cnn1: return "cnn1";
        default: return "trf";
    }
}

/** Transformer variant hyper-parameters (constants, to make the comparison readable) */
// Size of the Transformer variant: chosen to match cnn2 (about 105k parameters)
//   conv1+conv2 = 3424, proj 17d, per layer 8d^2+11d, head 11d -> 3 layers at d=64 gives ~104k
constexpr int kTrfDModel = 64;
constexpr int kTrfHeads = 4;      // d_head = 16 (even -> RoPE can be used)
constexpr int kTrfLayers = 3;
constexpr int kTrfDff = 128;      // = 2 * d_model
constexpr int kTrfTokens = 7 * 7;                // after two 2x2 pools: one token per spatial position -> 49
constexpr int kTrfInChannels = 16;               // input channels of the patch embedding (the second convolution's output)
constexpr int kHidden = 128;                     // hidden width of the CNN variant's encoder

/** conv→relu→pool→[conv→relu→pool]→flatten→fc→relu→fc→ce */
template <bool two_conv>
using cnn_chain_t = std::conditional_t<
    two_conv,
    complex_net_builder_t<double>
        ::push_back_updatable<conv2d_net_t, mnist_conv_upr_tpl>   // 0
        ::push_back_staticnet<relu_net_t>              // 1
        ::push_back_staticnet<pool2d_net_t>            // 2
        ::push_back_updatable<conv2d_net_t, mnist_conv_upr_tpl>   // 3
        ::push_back_staticnet<relu_net_t>              // 4
        ::push_back_staticnet<pool2d_net_t>            // 5
        ::push_back_staticnet<flatten_net_t>           // 6
        ::push_back_updatable<weight_net_t, mnist_conv_upr_tpl>   // 7
        ::push_back_staticnet<relu_net_t>              // 8
        ::push_back_staticnet<dropout_net_t>           // 9 dropout (regularisation)
        ::push_back_updatable<weight_net_t, mnist_conv_upr_tpl>   // 10
        ::push_back_staticnet<ce_loss_t>               // 11
        ::type,
    complex_net_builder_t<double>
        ::push_back_updatable<conv2d_net_t, mnist_conv_upr_tpl>   // 0
        ::push_back_staticnet<relu_net_t>              // 1
        ::push_back_staticnet<pool2d_net_t>            // 2
        ::push_back_staticnet<flatten_net_t>           // 3
        ::push_back_updatable<weight_net_t, mnist_conv_upr_tpl>   // 4
        ::push_back_staticnet<relu_net_t>              // 5
        ::push_back_staticnet<dropout_net_t>           // 6 dropout (regularisation)
        ::push_back_updatable<weight_net_t, mnist_conv_upr_tpl>   // 7
        ::push_back_staticnet<ce_loss_t>               // 8
        ::type>;

/** conv->relu->pool->(per-position projection into tokens)->Transformer encoder->mean pool->fc->ce */
using trf_chain_t = complex_net_builder_t<double>
    ::push_back_updatable<conv2d_net_t, mnist_conv_upr_tpl>       // 0 conv 1→8
    ::push_back_staticnet<relu_net_t>                  // 1
    ::push_back_staticnet<pool2d_net_t>                // 2 28→14   → [8, 196]
    ::push_back_updatable<conv2d_net_t, mnist_conv_upr_tpl>       // 3 conv 8→16
    ::push_back_staticnet<relu_net_t>                  // 4
    ::push_back_staticnet<pool2d_net_t>                // 5 14->7    -> [16, 49] (49 tokens)
    ::push_back_updatable<weight_net_t, mnist_conv_upr_tpl>       // 6 patch embedding: 16 -> d_model, per position
    ::push_back_updatable<cls_token_net_t, mnist_conv_upr_tpl>    // 7 prepend a learnable CLS vector
    ::push_back_impl<encoder_t<double, mnist_conv_upr_tpl>>       // 8 Transformer encoder (bidirectional)
    ::push_back_staticnet<take_token_net_t>            // 9 take the CLS column -> [d_model, 1]
    ::push_back_staticnet<dropout_net_t>               // 10 dropout (regularisation)
    ::push_back_updatable<weight_net_t, mnist_conv_upr_tpl>       // 11 classifier head
    ::push_back_staticnet<ce_loss_t>                   // 12
    ::type;

template <arch_t A>
using mnist_net_t = std::conditional_t<A == arch_t::cnn2, cnn_chain_t<true>,
                     std::conditional_t<A == arch_t::cnn1, cnn_chain_t<false>, trf_chain_t>>;

/** Layer indices inside the chain (they differ per architecture; kept in one place) */
template <arch_t A>
struct idx_t
{
    static constexpr int conv1 = 0;
    static constexpr int pool1 = 2;
    static constexpr int conv2 = 3;                    // cnn2
    static constexpr int pool2 = 5;                    // cnn2
    static constexpr int flatten = (A == arch_t::cnn2) ? 6 : 3;
    static constexpr int proj = 6;                     // trf: patch embedding
    static constexpr int cls = 7;                      // trf: the learnable CLS vector
    static constexpr int encoder = 8;                  // trf
    static constexpr int taketoken = 9;                // trf
    static constexpr int fc1 = (A == arch_t::cnn2) ? 7 : 4;    // the CNN encoder's first layer
    static constexpr int dropout = (A == arch_t::cnn2) ? 9 : ((A == arch_t::cnn1) ? 6 : 10);
    static constexpr int fc2 = (A == arch_t::cnn2) ? 10 : ((A == arch_t::cnn1) ? 7 : 11);
    static constexpr int loss = (A == arch_t::cnn2) ? 11 : ((A == arch_t::cnn1) ? 8 : 12);
};

/** Feature count after flattening, for the CNN variants */
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

    // convolution stem (shared by all three architectures)
    net.template get<idx::conv1>().set_param(1, 8, 28, 28, 5, 5, 1, 1, 2, 2);      // → [8, 784]
    net.template get<idx::pool1>().set_param(pool_mode::max, 28, 28, 2, 2, 2, 2);  // → [8, 196]

    if constexpr (A == arch_t::cnn2)
    {
        net.template get<idx::conv2>().set_param(8, 16, 14, 14, 5, 5, 1, 1, 2, 2);      // → [16, 196]
        net.template get<idx::pool2>().set_param(pool_mode::max, 14, 14, 2, 2, 2, 2);   // → [16, 49]
        net.template get<idx::flatten>().set_param(16, 49);
        net.reinit(std::vector<int>{features_of<A>(), hidden, 10});                    // only affects the two fc layers
    }
    else if constexpr (A == arch_t::cnn1)
    {
        net.template get<idx::flatten>().set_param(8, 196);
        net.reinit(std::vector<int>{features_of<A>(), hidden, 10});
    }
    else
    {
        // second convolution + second pool: 28->14->7, token count 196->49 (16x less self-attention work)
        net.template get<idx::conv2>().set_param(8, kTrfInChannels, 14, 14, 5, 5, 1, 1, 2, 2);  // → [16, 196]
        net.template get<idx::pool2>().set_param(pool_mode::max, 14, 14, 2, 2, 2, 2);           // → [16, 49]
        // patch embedding: project every spatial position of the 16 channels into a d_model-dim token
        net.template get<idx::proj>().reinit(std::vector<int>{kTrfInChannels, kTrfDModel});
        // CLS vector: d_model dims, prepended to the original 49 tokens
        net.template get<idx::cls>().set_param(kTrfDModel);
        // Transformer encoder: layers / heads / d_model / d_ff / sequence length (tokens + 1 CLS)
        net.template get<idx::encoder>().set_param(kTrfLayers, kTrfHeads, kTrfDModel, kTrfDff, kTrfTokens + 1);
        net.template get<idx::taketoken>().set_param(0);      // take the CLS column
        // Attach RoPE to every layer (mat_mha_t::bind_rope fetches the shared entry by d_head from the
        // registry) so attention knows the token order; without it the encoder sees a bag of patches and
        // position information can only come from the conv stem.
        for (int i = 0; i < kTrfLayers; ++i)
            net.template get<idx::encoder>().get_mha(i).bind_rope(kTrfTokens + 1);
        // classifier head d_model -> 10
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
 * Closed form for "the CNN variant's parameter count at a given hidden width" (same accounting as
 * param_count):
 * conv1 (+conv2) + fc1(features x h + h) + fc2(h x 10 + 10)。
 * match mode uses it to solve for the width whose parameter count is closest to the Transformer's.
 */
std::size_t cnn_params_for_hidden(int hidden, bool two_conv)
{
    const std::size_t conv = two_conv
        ? (1 * 8 * 5 * 5 + 8) + (8 * 16 * 5 * 5 + 16)
        : (1 * 8 * 5 * 5 + 8);
    const std::size_t features = two_conv ? 16 * 7 * 7 : 8 * 14 * 14;
    return conv + features * hidden + hidden + static_cast<std::size_t>(hidden) * 10 + 10;
}

/** Solve for the hidden width whose cnn2 parameter count is closest to the reference (trf's) */
int hidden_matching(std::size_t target, bool two_conv)
{
    int best_h = 1;
    std::size_t best_gap = static_cast<std::size_t>(-1);
    for (int h = 1; h <= 4096; ++h)
    {
        const std::size_t p = cnn_params_for_hidden(h, two_conv);
        const std::size_t gap = p > target ? p - target : target - p;
        if (gap < best_gap) { best_gap = gap; best_h = h; }
        if (p > target) break;                 // the count grows monotonically with the width, so stop once it passes the target
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
        // count the encoder's internal parameters too (MHA projections + FFN + LayerNorm)
        const int d = kTrfDModel, ff = kTrfDff;
        const std::size_t per_layer = 4 * d * d + 4 * d + d * ff + ff + ff * d + d + 2 * 2 * d;
        n += static_cast<std::size_t>(kTrfLayers) * per_layer;
        n += static_cast<std::size_t>(d);        // the CLS vector
    }
    return n;
}

// ---------------------------------------------------------------- serialization (shared by all three architectures)

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
        w.add("cls.token", net.template get<idx::cls>().token());     // the CLS vector is stored separately
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

// ---------------------------------------------------------------- training / evaluation

struct args_t
{
    std::string data_dir = "build/mnist";
    std::string save_path, load_path;
    std::string arch = "both";                 // cnn2 | cnn1 | trf | both | all
    int epochs = 3, batch = 16;
    int hidden = 128;                          // hidden width of the CNN encoder (--hidden; match mode solves for it)
    double lr = 2e-3;
    double dropout = 0.2;                      // dropout probability before the classifier head (0 = off)
    double weight_decay = 0.01;                // AdamW's decoupled weight decay
    std::string scheduler = "cosine";          // cosine (annealing + warm restarts) | fixed
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
    net.template get<idx_t<A>::dropout>().set_enabled(false);   // dropout must be off for evaluation
    const std::size_t n = std::min(limit, d.size());
    std::size_t correct = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        dmat x(1, 784);
        for (int p = 0; p < 784; ++p) x(0, p) = d.images[i][static_cast<std::size_t>(p)];
        const dmat logits = net.forward(x);        // forward only; the trailing CE is a pass-through layer and computes no gradient
        if (argmax_col(logits) == d.labels[i]) ++correct;
    }
    net.template get<idx_t<A>::dropout>().set_enabled(true);    // training keeps dropout on
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
        std::cout << "[" << r.name << "] loaded " << load_path << " epochs=" << meta.epochs
                  << " loss=" << meta.loss << " acc=" << meta.accuracy << "\n";
    }

    const std::size_t test_n = std::min(static_cast<std::size_t>(a.test_limit), test.size());
    r.test_acc = evaluate<A>(net, test, test_n);
    std::cout << "[" << r.name << "] test_acc before training = " << r.test_acc << "\n";

    const std::size_t n_train = std::min(static_cast<std::size_t>(a.train_limit), train.size());
    std::vector<std::size_t> order(train.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::mt19937 rng(a.seed);

    // LR schedule: cosine annealing + warm restarts (the library's cosine_annealing_decay), stepped per
    // mini-batch.
    // init_decay_steps is half the total steps, so the run contains exactly one warm restart (later
    // cycles double).
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
                    sched.step();                        // step first, then read: skips the lr=0 step 0 of the warmup
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
        std::cout << "[" << r.name << "] [check] reloaded test_acc=" << acc;
        if (std::abs(acc - r.test_acc) > 1e-12) { std::cout << "  FAILED (differs from the saved run)\n"; r.ok = false; }
        else std::cout << "  OK\n";
    }

    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

void print_comparison(std::vector<result_t> const& rs)
{
    std::cout << "\n================ comparison (same data / hyper-parameters / seed) ================\n";
    std::cout << std::left << std::setw(8) << "arch" << std::right << std::setw(10) << "params"
              << std::setw(8) << "epochs" << std::setw(12) << "train_loss"
              << std::setw(12) << "train_acc" << std::setw(12) << "test_acc"
              << std::setw(10) << "seconds" << "\n";
    for (const auto& r : rs)
        std::cout << std::left << std::setw(8) << r.name << std::right << std::setw(10) << r.params
                  << std::setw(8) << r.epochs << std::setw(12) << r.loss
                  << std::setw(12) << r.train_acc << std::setw(12) << r.test_acc
                  << std::setw(10) << r.seconds << (r.ok ? "" : "   <-- round-trip self-check failed") << "\n";
    std::cout << "====================================================================\n";
    if (rs.size() == 2)
        std::cout << "test_acc difference (" << rs[1].name << " - " << rs[0].name << "）= "
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
                         "  cnn2  = conv->relu->pool->conv->relu->pool->flatten->fc->relu->fc->ce (standard CNN)\n"
                         "  cnn1  = conv->relu->pool->flatten->fc->relu->fc->ce\n"
                         "  trf   = conv->relu->pool->(patch embedding)->Transformer encoder->mean pool->fc->ce\n"
                         "  both  = cnn2 vs trf (default)    all = all three\n"
                         "  match = shrink cnn2's hidden width to match trf's parameter count, then compare at equal capacity\n"
                         "          (--hidden N sets the CNN width manually)\n";
            return (arg == "--help") ? 0 : 1;
        }
    }

    // which architectures to run
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
        std::cout << "[data] synthetic dataset (--synthetic): train=" << train.size() << " test=" << test.size() << "\n";
    }
    else
    {
        try
        {
            train = load_mnist(a.data_dir + "/train-images-idx3-ubyte", a.data_dir + "/train-labels-idx1-ubyte");
            test = load_mnist(a.data_dir + "/t10k-images-idx3-ubyte", a.data_dir + "/t10k-labels-idx1-ubyte");
            std::cout << "[data] MNIST from " << a.data_dir << "：train=" << train.size()
                      << " test=" << test.size() << " (official split, the two sets do not overlap)\n";
        }
        catch (std::exception const& e)
        {
            std::cout << "[data] " << e.what() << "\n[data] falling back to the synthetic dataset\n";
            train = make_synthetic(static_cast<std::size_t>(std::max(1l, std::min(a.train_limit, 1000l))), a.seed);
            test = make_synthetic(static_cast<std::size_t>(std::max(1l, std::min(a.test_limit, 500l))), a.seed + 1);
        }
    }

    if (a.arch == "match")
    {
        // equal-capacity comparison: compute the Transformer variant's parameter count, then solve for the
        // CNN hidden width
        const std::size_t trf_params = param_count<arch_t::trf>(make_net<arch_t::trf>(a.seed, a.lr, a.hidden, a.dropout, a.weight_decay));
        a.hidden = hidden_matching(trf_params, /*two_conv=*/true);
        std::cout << "[match] trf parameters = " << trf_params << "; shrinking cnn2 hidden width to " << a.hidden
                  << " (parameters " << cnn_params_for_hidden(a.hidden, true) << ") for an equal-capacity comparison\n";
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
            std::cerr << "[" << arch_name(arch) << "] --load file does not exist: " << lp << "\n";
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
