/**
 * Serialization of training results: a weights + metadata round-trip (JASMINE_WEIGHTS_V1).
 *
 * Three things are covered:
 *   1. add_layer_params / read_layer_params work for both conv2d_net_t and weight_net_t;
 *   2. after the round-trip the **forward output matches** (the file is float32, hence a 1e-6
 *      tolerance instead of bit-exactness);
 *   3. the metadata (epoch / loss / accuracy) is stored as 1x1 tensors in the same file and read back
 *      unchanged; a missing tensor throws instead of silently yielding 0.
 */

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_conv_t.hpp"
#include "jas_net_t.hpp"
#include "jas_weight_io.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{

using dmat = mat_t<double>;

dmat make_mat(int rows, int cols, double scale, unsigned seed)
{
    dmat m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-scale, scale);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m(i, j) = dist(rng);
    return m;
}

/** The model under test: a small convolution plus a small fully-connected layer */
struct toy_model_t
{
    conv2d_net_t<dmat, sgd_t> conv{1, 2, 4, 4, 3, 3, 1, 1, 1, 1};   // → [2, 16]
    weight_net_t<dmat, sgd_t> fc{2, 3};                              // 2 -> 3 (conv's output rows = channels)

    dmat forward(dmat const& x)
    {
        return fc.forward(conv.forward(x));
    }
};

std::string temp_path(std::string const& name)
{
    return std::string("./") + name;
}

} // namespace

TEST(ModelSerialization, RoundTripPreservesForwardOutput)
{
    const std::string path = temp_path("test_serialization_roundtrip.jas");

    toy_model_t saved;
    saved.conv.weight() = make_mat(2, 9, 0.5, 1);
    saved.conv.bias() = make_mat(2, 1, 0.3, 2);
    saved.fc.weight() = make_mat(3, 2, 0.5, 3);
    saved.fc.bias() = make_mat(3, 1, 0.3, 4);

    const dmat x = make_mat(1, 16, 0.5, 5);
    const dmat y_ref = saved.forward(x);

    {
        weight_writer_t w;
        add_layer_params(w, "conv", saved.conv);
        add_layer_params(w, "fc", saved.fc);
        w.add_scalar("meta.epochs", 7);
        w.add_scalar("meta.loss", 0.125);
        w.add_scalar("meta.accuracy", 0.9375);
        w.write(path);
        EXPECT_EQ(w.size(), 4u + 3u);      // 4 parameter tensors + 3 metadata entries
    }

    // the header must be the format the documentation describes (format stability)
    {
        std::ifstream in(path, std::ios::binary);
        ASSERT_TRUE(in.good());
        std::string magic, dtype;
        int count = 0;
        std::getline(in, magic);
        std::getline(in, dtype);
        in >> count;
        EXPECT_EQ(magic, "JASMINE_WEIGHTS_V1");
        EXPECT_EQ(dtype, "f32");
        EXPECT_EQ(count, 7);
    }

    // load into a model holding completely different weights and confirm the output is restored
    toy_model_t loaded;
    loaded.conv.weight() = make_mat(2, 9, 0.5, 91);
    loaded.conv.bias() = make_mat(2, 1, 0.3, 92);
    loaded.fc.weight() = make_mat(3, 2, 0.5, 93);
    loaded.fc.bias() = make_mat(3, 1, 0.3, 94);

    {
        weight_file_t wf;
        wf.load(path);
        ASSERT_TRUE(wf.has("conv.weight"));
        ASSERT_TRUE(wf.has("fc.bias"));
        read_layer_params(wf, "conv", loaded.conv);
        read_layer_params(wf, "fc", loaded.fc);

        EXPECT_EQ(static_cast<int>(wf.read_scalar<float>("meta.epochs")), 7);
        EXPECT_NEAR(wf.read_scalar<float>("meta.loss"), 0.125f, 1e-7f);
        EXPECT_NEAR(wf.read_scalar<float>("meta.accuracy"), 0.9375f, 1e-7f);

        // a missing tensor must throw (no silent zeros)
        EXPECT_THROW(wf.read_scalar<float>("meta.missing"), std::runtime_error);
    }

    // float32 storage read back into double: tolerance 1e-6
    ExpectNearMat(loaded.forward(x), y_ref, 1e-6);
    std::remove(path.c_str());
}

TEST(ModelSerialization, ShapeMismatchIsRejected)
{
    const std::string path = temp_path("test_serialization_shape.jas");
    {
        weight_writer_t w;
        add_layer_params(w, "conv", toy_model_t{}.conv);
        w.write(path);
    }

    toy_model_t other;
    other.conv.set_param(1, 3, 4, 4, 3, 3, 1, 1, 1, 1);   // different output channels -> different weight shape
    weight_file_t wf;
    wf.load(path);
    EXPECT_THROW(read_layer_params(wf, "conv", other.conv), std::runtime_error);
    std::remove(path.c_str());
}
