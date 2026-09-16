#include <array>
#include <atomic>
#include <cmath>
#include <string>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "jas_RoPE_t.hpp"
#include "test_helpers.hpp"

using namespace jasmine;

/**
 * RoPE 旋转矩阵缓存的「填充不变量」回归测试。
 *
 * 缓存按 (d, layout) 挂在进程级单例 rope_registry_t 上，MHA 的所有头在
 * `#pragma omp parallel for` 里共享同一份 —— 「哪些格子已经算过」这件事既要不重复
 * 计算（否则每加一个位置就重算整张表），又必须在并发下成立。
 *
 * 不变量只有一条：**缓存自称已填充的矩形里，不允许存在某个从未被写过的格子**。
 * 曾经这里破过两次，都是同一个原因 —— 用「两个轴的 max」记账，而不是真写过的范围：
 *.  1) init 只在请求到的矩形里逐格跳过「已填」格子，却把两轴上限各自取 max：
 *     先 init(0, d, 0, 2)（高而窄）再 init(0, 2, 0, 8)（矮而宽），[2,d)×[2,8) 从没被
 *     遍历到却被记成已填 —— 那块的 cos/sin 一律是 0（mat_t 分配时 memset 过），
 *     于是对应的特征对被整体清零，误差沿层数放大（对齐测试里就是逐层漂移）。
 *.  2) 补齐过程没有同步：多个线程同时要求扩容时，一个线程可能读到另一个线程
 *     正在写、甚至还没写的格子（扩容还会顺手换掉底层存储）。
 *
 * 现在 fill_from_origin() 把「补齐」定义成精确的 L 形带（新行 × 全部列 + 老行 × 新列），
 * 整个「扩容 + 补齐」在 m_fill_mutex 里串行化，已填充上限用 release 发布、读者 acquire，
 * 命中时走完全无锁的只读快路径。下面四个测试分别钉住这些面：
 *
 *  - FilledRectangleHasNoUnwrittenCell：确定性回归上面第 1 条（与线程无关，失败必然重现）。
 *  - ConcurrentExtentsMatchAnalyticValues：并发回归上面第 2 条；配合 TSan 曾经能看到
 *    m_enable_rows/m_enable_cols（今 m_filled_*）上的数据竞争，现在是干净的。
 *  - ConcurrentFillMatchesSerialFill：整块矩形「并发 vs 串行」对拍，覆盖全量块而不只是抽样。
 *  - ReservedCacheIsSafeToReadConcurrently：static_fixed 只读共享的对照组。
 */

namespace
{

/** 解析旋转块：pair i、位置 m 的 θ = m / 10000^(2i/d)，块为 [[c,-s],[s,c]] */
template <typename View>
void ExpectUniteAnalytic(const View& u, int pair, int pos, int d, const char* tag)
{
    const double theta = static_cast<double>(pos) /
                         std::pow(10000.0, (2.0 * pair) / static_cast<double>(d));
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const std::string where = std::string(tag) + " pair=" + std::to_string(pair) +
                              " pos=" + std::to_string(pos);
    EXPECT_NEAR(u(0, 0), c, 1e-12) << where;
    EXPECT_NEAR(u(0, 1), -s, 1e-12) << where;
    EXPECT_NEAR(u(1, 0), s, 1e-12) << where;
    EXPECT_NEAR(u(1, 1), c, 1e-12) << where;
}

} // namespace

/**
 * 确定性回归：先铺「高而窄」的一条，再请求「矮而宽」的一块。
 *
 * init(0, d, 0, 2)   → 写了 [0,d) × [0,2)（位置 0 的全部特征对）
 * init(0, 2, 0, 8)   → 只写了 [0,2) × [2,8)（位置 1..3 的前两个特征行）
 *
 * 旧实现把「已填充」记成两个轴的 max：(d, 8)，于是 [2,d) × [2,8) 这块既没有在第二次
 * init 里被遍历到（它的行区间是 [0,2)），又被记成已填充，range() 认为无需补齐 ——
 * 读到的是全 0 的旋转块，对应特征对被静默清零。这里的断言（pair 1..3 = 行 2..8，
 * pos 1..3 = 列 2..8）正好覆盖那块洞。
 */
TEST(RoPeCacheThreading, FilledRectangleHasNoUnwrittenCell)
{
    const int d = 8; // 4 个特征对
    mat_RoPE_t<double> rope(d);

    rope.init(0, d, 0, 2);
    rope.init(0, 2, 0, 8);

    for (int pair = 0; pair < d / 2; ++pair)
        for (int pos = 1; pos <= 3; ++pos)
            ExpectUniteAnalytic(rope.forward_unite(pair, pos), pair, pos, d, "serial");
}

/**
 * 并发面：8 个线程各自请求不同矩形。
 *
 * 一半线程从位置 0 往前扫，一半从末尾往后退；每个线程内部还会轮转特征对，
 * 使同一时刻不同线程请求的「行区间 / 列区间」互不相同 —— 只按一个轴的
 * max 记账，遇上这种交错就会把别人还没写的格子记成已填充。
 *
 * 判定必须在**线程内部**做完：视图拿在手里，出了线程就可能被扩容换掉；
 * 更重要的是，事后再读会再走一次 range()，而补齐逻辑本身可能顺手把洞填上，
 * 把 bug 掩盖掉 —— 洞一旦被写上的值正确，事后检查就再也看不见它了。
 */
TEST(RoPeCacheThreading, ConcurrentExtentsMatchAnalyticValues)
{
    const int d = 64; // 32 个特征对
    const int pairs = d / 2;
    const int positions = 24;
    const int nthreads = 8;

    mat_RoPE_t<double> rope(d);

    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t)
    {
        workers.emplace_back([&, t]
        {
            for (int round = 0; round < positions; ++round)
            {
                const int pos = (t % 2 == 0) ? round : (positions - 1 - round);
                for (int k = 0; k < pairs; ++k)
                {
                    const int pair = (k + t) % pairs;
                    ExpectUniteAnalytic(rope.forward_unite(pair, pos), pair, pos, d, "concurrent");
                }
            }
        });
    }
    for (auto& w : workers)
        w.join();
}

/**
 * 并发 vs 串行对拍：同样的请求集合，一个实例串行补齐，另一个实例多线程乱序补齐，
 * 事后把整块矩形逐格比较。
 *
 * 与 ConcurrentExtentsMatchAnalyticValues 的区别在覆盖面：那条按交错抽样、逐个块
 * 对解析值；这条是把 pairs × positions 全部块（= 整块已填充矩形）都读出来比一遍，
 * 且比对基准是另一个实例的串行结果。值本身由解析值钉死，这里额外钉住的是：
 * 乱序请求 + 扩容换存储的过程中，没有任何一次读会拿到半成品或被换掉的旧存储。
 */
TEST(RoPeCacheThreading, ConcurrentFillMatchesSerialFill)
{
    const int d = 64;
    const int pairs = d / 2;
    const int positions = 24;
    const int nthreads = 8;
    const std::size_t n = static_cast<std::size_t>(pairs) * static_cast<std::size_t>(positions);

    // 串行参考：按 (pair, pos) 字典序逐个补齐并读出
    mat_RoPE_t<double> serial(d);
    std::vector<std::array<double, 4>> ref(n);
    for (int pair = 0; pair < pairs; ++pair)
    {
        for (int pos = 0; pos < positions; ++pos)
        {
            const auto u = serial.forward_unite(pair, pos);
            ref[static_cast<std::size_t>(pair) * positions + pos] =
                {u(0, 0), u(0, 1), u(1, 0), u(1, 1)};
        }
    }

    // 并发：同一集合按线程跨步取（顺序被打散），各自读回自己的块
    mat_RoPE_t<double> concurrent(d);
    std::vector<std::array<double, 4>> got(n);
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t)
    {
        workers.emplace_back([&, t]
        {
            for (std::size_t i = static_cast<std::size_t>(t); i < n;
                 i += static_cast<std::size_t>(nthreads))
            {
                const int pair = static_cast<int>(i) / positions;
                const int pos = static_cast<int>(i) % positions;
                const auto u = concurrent.forward_unite(pair, pos);
                got[i] = {u(0, 0), u(0, 1), u(1, 0), u(1, 1)};
            }
        });
    }
    for (auto& w : workers)
        w.join();

    for (std::size_t i = 0; i < n; ++i)
    {
        for (int c = 0; c < 4; ++c)
        {
            EXPECT_DOUBLE_EQ(got[i][c], ref[i][c])
                << "block " << i << " (pair " << i / positions << ", pos " << i % positions
                << ") cell " << c;
        }
    }
}

/**
 * 只读共享的正确姿势：reserve 之后运行期不再改底层存储（static_fixed），
 * 多线程分别去读，任何一次读都必须等于解析值。这条是并发测试的对照组：
 * 它本来就该过，用来证明测试本身（解析值、并发驱动方式）没问题，
 * 而不是像 ConcurrentExtentsMatchAnalyticValues 那样去戳动态扩容。
 */
TEST(RoPeCacheThreading, ReservedCacheIsSafeToReadConcurrently)
{
    const int d = 64;
    const int pairs = d / 2;
    const int positions = 24;
    const int nthreads = 8;

    mat_RoPE_t<double> rope(d);
    rope.set_cache_mode(rope_cache_mode::static_fixed);
    rope.reserve(positions);

    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t)
    {
        workers.emplace_back([&, t]
        {
            for (int round = 0; round < positions; ++round)
            {
                const int pos = (t + round) % positions;
                for (int k = 0; k < pairs; ++k)
                {
                    const int pair = (k + t) % pairs;
                    ExpectUniteAnalytic(rope.forward_unite(pair, pos), pair, pos, d, "reserved");
                }
            }
        });
    }
    for (auto& w : workers)
        w.join();
}
