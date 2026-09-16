#include <cmath>
#include <gtest/gtest.h>
#include <utility>
#include <type_traits>

#include "jas_mat_t.hpp"
#include "jas_mat_view_t.hpp"
#include "jas_mat_express_t.hpp"
#include "test_helpers.hpp"

/**
 * 表达式模板的生命周期回归测试。
 *
 * 背景：表达式节点原先一律用 `T const&` 持有操作数。这对「左值操作数」没问题（借引用、零拷贝），
 * 但对「右值操作数」是悬垂引用 —— 右值往往就是临时的子表达式或临时矩阵：
 *
 *     auto tree = (a + b) * c;   // (a+b) 这个临时 mat_add_t 在语句结束就销毁了
 *     mat_t<double> r = tree;    // tree 里的引用已悬垂，ASan 报 stack-use-after-scope
 *
 * 现在改为按【值类别】分派：左值借引用（零拷贝），右值按值拥有。
 * 下面的用例锁定这一契约，任何一条失败都说明表达式树重新变成了「不能安全存储的值」。
 */

using namespace jasmine;

namespace
{

/** 按值接收表达式树：模拟「把树交给别的函数 / 别的执行后端」。
 *  能编译通过本身就说明树是可拷贝的值类型，而不是一堆悬垂引用。 */
template <typename tree_type>
mat_t<typename tree_type::ele_type> materialize_by_value(tree_type tree)
{
    return tree.clone();
}

/** 按值返回一个矩阵，用来制造「右值矩阵操作数」 */
mat_t<double> make_filled(int rows, int cols, double v)
{
    mat_t<double> m(rows, cols);
    m = v;
    return m;
}

mat_t<double> make_2x2(double a, double b, double c, double d)
{
    return mat_t<double>(2, 2, {a, b, c, d});
}

} // namespace

// ---------------------------------------------------------------------------
// 存储策略：编译期锁定「左值借引用、右值按值拥有」
// ---------------------------------------------------------------------------

TEST(ExpressionLifetime, LvalueOperandsAreBorrowedNotCopied)
{
    // 左值操作数必须仍然借引用 —— 这是表达式模板避免临时量的根本，
    // 若改成按值会退化成「每构造一个表达式就深拷贝整块矩阵」。
    using tree_t = decltype(std::declval<mat_t<double>&>() + std::declval<mat_t<double>&>());
    static_assert(std::is_reference_v<typename tree_t::lval_storage_type>,
                  "左值操作数必须按引用持有（零拷贝）");
    static_assert(std::is_reference_v<typename tree_t::rval_storage_type>,
                  "左值操作数必须按引用持有（零拷贝）");
}

TEST(ExpressionLifetime, RvalueOperandsAreOwnedByValue)
{
    // 右值操作数必须按值拥有，否则必然悬垂
    using tree_t = decltype(std::declval<mat_t<double>>() + std::declval<mat_t<double>&>());
    static_assert(!std::is_reference_v<typename tree_t::lval_storage_type>,
                  "右值操作数必须按值拥有，否则表达式树一被存下来就悬垂");
}

TEST(ExpressionLifetime, NestedExpressionNodeIsOwnedByValue)
{
    // (a+b)*c 里的左操作数是临时表达式节点，同样必须按值拥有
    using inner_t = decltype(std::declval<mat_t<double>&>() + std::declval<mat_t<double>&>());
    using tree_t = decltype(std::declval<inner_t>() * std::declval<mat_t<double>&>());
    static_assert(!std::is_reference_v<typename tree_t::lval_storage_type>,
                  "嵌套表达式节点必须按值拥有");
}

TEST(ExpressionLifetime, ScalarOperandIsStoredByValue)
{
    using tree_t = decltype(std::declval<mat_t<double>&>() / std::declval<double&>());
    static_assert(!std::is_reference_v<typename tree_t::rval_storage_type>,
                  "标量操作数必须按值存（左值标量也不例外）");
}

// ---------------------------------------------------------------------------
// 回归：把树存下来之后再物化（原先的 UB 路径）
// ---------------------------------------------------------------------------

TEST(ExpressionLifetime, NestedExpressionSurvivesItsTemporaries)
{
    mat_t<double> a(2, 2, {1, 2, 3, 4});
    mat_t<double> b(2, 2, {10, 20, 30, 40});
    mat_t<double> c(2, 2, {1, 1, 1, 1});

    // 参照：同一个完整表达式内物化
    mat_t<double> direct = (a + b) * c;

    // 回归路径：先把树存进变量，(a+b) 这个临时量在此销毁
    auto tree = (a + b) * c;
    mat_t<double> later = tree;

    ExpectShape(later, 2, 2);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(later(i, j), direct(i, j), 1e-12);

    EXPECT_NEAR(later(0, 0), 11.0, 1e-12);
    EXPECT_NEAR(later(1, 1), 44.0, 1e-12);
}

TEST(ExpressionLifetime, DeeplyNestedTreeSurvivesBeingStored)
{
    mat_t<double> a(2, 2, {1, 2, 3, 4});
    mat_t<double> b(2, 2, {10, 20, 30, 40});
    mat_t<double> c(2, 2, {1, 1, 1, 1});

    mat_t<double> direct = ((a + b) - a) * c / (a + c);

    auto tree = ((a + b) - a) * c / (a + c);
    mat_t<double> later = tree;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(later(i, j), direct(i, j), 1e-12);
}

TEST(ExpressionLifetime, TreeIsSafeToCopyAndPassByValue)
{
    mat_t<double> a(2, 2, {1, 2, 3, 4});
    mat_t<double> b(2, 2, {10, 20, 30, 40});
    mat_t<double> c(2, 2, {1, 1, 1, 1});

    auto tree = (a + b) * c;
    auto copy = tree;                              // 拷贝构造
    mat_t<double> from_original = tree;
    mat_t<double> from_copy = copy;
    mat_t<double> by_value = materialize_by_value(tree);   // 按值传参

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(from_original(i, j), from_copy(i, j), 1e-12);
            EXPECT_NEAR(from_original(i, j), by_value(i, j), 1e-12);
        }
}

TEST(ExpressionLifetime, TreeSurvivesMovingIntoAContainer)
{
    mat_t<double> a(2, 2, {1, 2, 3, 4});
    mat_t<double> b(2, 2, {10, 20, 30, 40});

    std::vector<decltype((a + b) * a)> trees;
    trees.push_back((a + b) * a);       // 移进容器后，临时量全部销毁
    auto tree = std::move(trees.back());
    trees.clear();

    mat_t<double> r = tree;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(r(i, j), (a + b)(i, j) * a(i, j), 1e-12);
}

// ---------------------------------------------------------------------------
// 回归：右值「叶子」操作数（临时矩阵 / 临时视图）
// ---------------------------------------------------------------------------

TEST(ExpressionLifetime, RvalueMatrixOperandIsOwnedByTree)
{
    mat_t<double> c(2, 2, {1, 1, 1, 1});

    // make_filled(...) 返回临时矩阵，必须在树里被按值拥有
    auto tree = (make_filled(2, 2, 3.0) + make_filled(2, 2, 4.0)) * c;
    mat_t<double> r = tree;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(r(i, j), 7.0, 1e-12);
}

TEST(ExpressionLifetime, TemporaryViewOperandIsOwnedByTree)
{
    mat_t<double> m(2, 2, {1, 2, 3, 4});
    mat_t<double> n(2, 2, {10, 20, 30, 40});

    // m.t() 是临时视图对象；视图本身被按值拥有（其引用的矩阵 m 由调用方持有）
    auto tree = m.t() * n;
    mat_t<double> r = tree;

    // 注意：本库的 `*` 是逐元素乘，矩阵积是 .dot()。
    // m.t() = [[1,3],[2,4]]，与 n 逐元素相乘
    EXPECT_NEAR(r(0, 0), 1.0 * 10.0, 1e-12);
    EXPECT_NEAR(r(0, 1), 3.0 * 20.0, 1e-12);
    EXPECT_NEAR(r(1, 0), 2.0 * 30.0, 1e-12);
    EXPECT_NEAR(r(1, 1), 4.0 * 40.0, 1e-12);
}

TEST(ExpressionLifetime, DotNodeOwnsTemporaryViewOperand)
{
    mat_t<double> a(2, 2, {1, 2, 3, 4});
    mat_t<double> identity(2, 2, {1, 0, 0, 1});

    // mat_dot_t 原先把引用写死在成员上，是 a.t().dot(b) 这类写法悬垂的来源
    auto tree = a.t().dot(identity);
    mat_t<double> r = tree;

    EXPECT_NEAR(r(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(r(0, 1), 3.0, 1e-12);
    EXPECT_NEAR(r(1, 0), 2.0, 1e-12);
    EXPECT_NEAR(r(1, 1), 4.0, 1e-12);
}

TEST(ExpressionLifetime, DotChainOnTemporaryNodesIsOwned)
{
    mat_t<double> a(2, 2, {1, 2, 3, 4});

    // 每个中间 mat_dot_t 都是临时量，必须逐层按值拥有
    auto tree = (a.t().dot(a)).dot(a);
    mat_t<double> r = tree;

    // a.t() = [[1,3],[2,4]]；a.t()*a = [[1*1+3*3, 1*2+3*4],[2*1+4*3, 2*2+4*4]] = [[10,14],[14,20]]
    mat_t<double> expected(2, 2, {10, 14, 14, 20});
    mat_t<double> expected2 = expected.dot(a);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(r(i, j), expected2(i, j), 1e-12);
}

TEST(ExpressionLifetime, TemporaryExpressionReceiverIsOwnedByDot)
{
    mat_t<double> a(2, 2, {1, 2, 3, 4});
    mat_t<double> b(2, 2, {10, 20, 30, 40});
    mat_t<double> identity(2, 2, {1, 0, 0, 1});

    // (a+b) 是临时表达式节点，作为 .dot() 的【接收者】时必须被按值拥有。
    // 这是最容易漏的一类：成员函数里的 *this 恒为左值。
    auto tree = (a + b).dot(identity);
    mat_t<double> r = tree;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(r(i, j), a(i, j) + b(i, j), 1e-12);
}

TEST(ExpressionLifetime, TemporaryMatrixReceiverIsOwnedByDot)
{
    mat_t<double> identity(2, 2, {1, 0, 0, 1});
    mat_t<double> a(2, 2, {1, 2, 3, 4});

    // 临时矩阵直接 .dot()
    auto tree = make_2x2(1, 2, 3, 4).dot(identity);
    mat_t<double> r = tree;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(r(i, j), a(i, j), 1e-12);
}

TEST(ExpressionLifetime, TemporaryArgumentOfDotIsOwned)
{
    mat_t<double> a(2, 2, {1, 2, 3, 4});
    mat_t<double> identity(2, 2, {1, 0, 0, 1});

    // 右操作数同样是临时量
    auto tree = a.dot(make_2x2(1, 0, 0, 1));
    mat_t<double> r = tree;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(r(i, j), a(i, j), 1e-12);
}

// ---------------------------------------------------------------------------
// 左值操作数的借用语义（有意保留的契约）
// ---------------------------------------------------------------------------

TEST(ExpressionLifetime, LvalueOperandsAreBorrowedSoMutationIsVisible)
{
    // 表达式是惰性的：左值操作数借引用，所以构造之后、物化之前修改原矩阵，
    // 结果会跟着变。这条锁定了「零拷贝」是有意为之，而不是漏拷贝的 bug。
    mat_t<double> a(1, 1, {1.0});
    mat_t<double> b(1, 1, {2.0});

    auto tree = a + b;
    a(0, 0) = 10.0;

    mat_t<double> r = tree;
    EXPECT_NEAR(r(0, 0), 12.0, 1e-12);
}

TEST(ExpressionLifetime, UnaryNodesAlsoOwnRvalueOperands)
{
    // 一元节点走的是另一套基类（mat_express_1_param_stable_t），同样要按值拥有右值
    using tree_t = decltype(exp(mat_t<double>(2, 2)));
    static_assert(!std::is_reference_v<typename tree_t::val_storage_type>,
                  "一元表达式节点必须按值拥有右值操作数");

    auto tree = exp(make_2x2(0.0, 1.0, 2.0, 3.0));
    mat_t<double> r = tree;
    EXPECT_NEAR(r(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(r(1, 1), std::exp(3.0), 1e-12);
}

TEST(ExpressionLifetime, SoftmaxOfRvalueOwnsItsOperand)
{
    auto tree = softmax(make_2x2(0.0, 0.0, 0.0, 0.0));
    mat_t<double> r = tree;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(r(i, j), 0.25, 1e-12);
}
