/**
 * dropout_net_t 单测（inverted dropout）。
 *
 * 关键性质：
 *   - 关闭 / p=0 时是恒等（评估路径）；
 *   - 训练时保留的元素乘以 1/(1-p)，所以输出的**期望**与输入相同（inverted 的含义）；
 *   - backward 用的是 forward 那一份 mask：被丢掉的位置梯度为 0，其余按同一个比例缩放；
 *   - 无参数静态层，不占 reinit 槽位。
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "jas_mat_concepts.hpp"
#include "jas_mat_init_t.hpp"
#include "jas_net_t.hpp"
#include "jas_updator_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

namespace
{
using dmat = mat_t<double>;
using drop_t = dropout_net_t<dmat>;
} // namespace

TEST(Dropout, DisabledOrZeroProbabilityIsIdentity)
{
    const dmat x(2, 3, {1, -2, 3, 4, -5, 6});

    drop_t off;
    off.set_param(0.5);
    off.set_enabled(false);
    ExpectNearMat(off.forward(x), x, 0.0);
    ExpectNearMat(off.backward(x), x, 0.0);          // 恒等前向 → 梯度原样回传

    drop_t p0;
    p0.set_param(0.0);
    ExpectNearMat(p0.forward(x), x, 0.0);
    ExpectNearMat(p0.backward(x), x, 0.0);
}

TEST(Dropout, KeptElementsAreScaledByOneOverKeep)
{
    g_random_engine.seed(1234);
    drop_t drop;
    drop.set_param(0.5);
    dmat in(20, 20);
    in = 1.0;                                         // 全 1 输入：输出只能是 0 或 2
    const dmat y = drop.forward(in);
    int kept = 0, dropped = 0;
    for (int i = 0; i < y.row_num(); ++i)
        for (int j = 0; j < y.col_num(); ++j)
        {
            if (y(i, j) == 0.0) ++dropped;
            else { EXPECT_DOUBLE_EQ(y(i, j), 2.0); ++kept; }   // 1/(1-0.5) = 2
        }
    EXPECT_GT(kept, 0);
    EXPECT_GT(dropped, 0);
    // 大数定律：丢弃比例接近 0.5（400 个样本，5% 容差足够宽松）
    const double drop_rate = static_cast<double>(dropped) / 400.0;
    EXPECT_NEAR(drop_rate, 0.5, 0.05) << "drop_rate=" << drop_rate;
}

TEST(Dropout, ExpectedValueIsPreserved)
{
    g_random_engine.seed(7);
    drop_t drop;
    drop.set_param(0.5);
    dmat in(50, 50);
    in = 3.0;
    const dmat y = drop.forward(in);
    double sum = 0.0;
    for (int i = 0; i < y.row_num(); ++i)
        for (int j = 0; j < y.col_num(); ++j) sum += y(i, j);
    const double mean = sum / 2500.0;
    EXPECT_NEAR(mean, 3.0, 0.3) << "inverted dropout 的均值应仍接近输入均值";
}

TEST(Dropout, BackwardUsesTheForwardMask)
{
    g_random_engine.seed(99);
    drop_t drop;
    drop.set_param(0.5);
    dmat in(16, 16);
    in = 2.0;                                    // 非零输入，便于从输出反推 mask
    const dmat y = drop.forward(in);
    dmat d(16, 16);
    d = 1.0;
    const dmat dx = drop.backward(d);

    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
        {
            const double mask = y(i, j) / 2.0;    // forward 输出 / 输入 = mask（0 或 2）
            if (mask == 0.0) EXPECT_DOUBLE_EQ(dx(i, j), 0.0);
            else EXPECT_DOUBLE_EQ(dx(i, j), 2.0); // delta(=1) * mask(=2)
        }
}

TEST(Dropout, IsStaticLayerWithoutReinit)
{
    static_assert(!is_updatable_net<drop_t>);
    static_assert(!is_reinitable_net<drop_t>);
    SUCCEED();
}

TEST(Dropout, RejectsInvalidProbabilityAndShape)
{
    drop_t drop;
    drop.set_param(1.0);
    EXPECT_THROW(drop.forward(dmat(2, 2)), std::invalid_argument);

    drop_t ok;
    ok.set_param(0.5);
    ok.forward(dmat(2, 3));
    EXPECT_THROW(ok.backward(dmat(2, 2)), std::runtime_error);
    EXPECT_THROW(ok.backward(dmat(3, 3)), std::runtime_error);
}

TEST(Dropout, NetTypeReportsMode)
{
    drop_t drop;
    drop.set_param(0.25);
    EXPECT_NE(drop.net_type().find("p:0.25"), std::string::npos);
    EXPECT_NE(drop.net_type().find("train"), std::string::npos);
    drop.set_enabled(false);
    EXPECT_NE(drop.net_type().find("eval"), std::string::npos);
}
