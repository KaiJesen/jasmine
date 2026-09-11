#include <gtest/gtest.h>
#include "mat_mha_t.hpp"
#include "test_helpers.hpp"


using namespace jasmine;
TEST(MhaTools, VConcatVSplitRoundTrip)
{
    std::vector<mat_t<double>> inputs;
    for (double i = 0; i < 4; ++i)
    {
        inputs.emplace_back(mat_t<double>(2, 2, {i, i + 1, i + 2, i + 3}));
    }

    mat_t<double> output = vconcat(inputs);
    ExpectShape(output, 8, 2);

    auto views = vsplit(output, 4);
    ASSERT_EQ(views.size(), 4u);
    for (int k = 0; k < 4; ++k)
    {
        ExpectNearMat(views[k], inputs[k], 1e-12);
    }
}
