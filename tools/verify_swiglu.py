"""一次性交叉验证：用 PyTorch 计算完整 SwiGLU FFN，生成等价的 C++ 程序并比对。

用法：
    python3 tools/verify_swiglu.py
生成 /tmp/swiglu_cross.cpp 并用 g++ 编译运行；退出码非 0 表示与 PyTorch 不一致。
比对的是 `down(silu(gate(x)) * up(x))`，其中 FFN 由 jasmine 的 gated_net_t 拼出。
"""
import subprocess
import sys

import torch

torch.manual_seed(7)
# d_ff = 8/3 * d_model 是 LLaMA 的取值约定
D_MODEL, D_FF, T = 6, 16, 4

lin_gate = torch.nn.Linear(D_MODEL, D_FF, bias=True).double()
lin_up = torch.nn.Linear(D_MODEL, D_FF, bias=True).double()
lin_down = torch.nn.Linear(D_FF, D_MODEL, bias=True).double()
act = torch.nn.SiLU()

# jasmine 的 mat_t 是 d_model×T（每列一个 token），这里逐列处理以对齐
x = torch.randn(D_MODEL, T, dtype=torch.float64)
with torch.no_grad():
    y = torch.stack(
        [lin_down(act(lin_gate(x[:, t])) * lin_up(x[:, t])) for t in range(T)], dim=1
    )  # d_model×T


def carr(name, tensor):
    """把张量摊平成一个 C 数组字面量。"""
    vals = ", ".join(repr(float(v)) for v in tensor.detach().reshape(-1))
    return f"const double {name}[{tensor.numel()}] = {{{vals}}};"


def fill(mat_name, arr_name, rows, cols):
    """生成按行主序从 C 数组填入 mat_t 的循环。"""
    return (
        f"    dmat {mat_name}({rows}, {cols});\n"
        f"    {{ int k = 0; for (int i = 0; i < {rows}; ++i)"
        f" for (int j = 0; j < {cols}; ++j) {mat_name}(i, j) = {arr_name}[k++]; }}"
    )


body = [
    "#include <cmath>",
    "#include <cstdio>",
    '#include "jas_net_t.hpp"',
    '#include "jas_silu_t.hpp"',
    "using namespace jasmine;",
    "using dmat = mat_t<double>;",
    "template<typename T> using upr = sgd_t<T>;",
    "int main() {",
    f"    const int d_model = {D_MODEL}, d_ff = {D_FF}, T = {T};",
    carr("gWg", lin_gate.weight),
    carr("gBg", lin_gate.bias),
    carr("gWu", lin_up.weight),
    carr("gBu", lin_up.bias),
    carr("gWd", lin_down.weight),
    carr("gBd", lin_down.bias),
    carr("gX", x),
    carr("gY", y),
    # ---- 构建 SwiGLU FFN：gated(gate=Linear→SiLU, up=Linear) → down ----
    "    gated_ffn_branches_t<double, upr, silu_net_t> gated;",
    "    weight_net_t<dmat, upr> down;",
    "    gated.reinit(std::vector<int>{d_model, d_ff});",
    "    down.reinit(std::vector<int>{d_ff, d_model});",
    fill("Wg", "gWg", D_FF, D_MODEL),
    fill("Bg", "gBg", D_FF, 1),
    fill("Wu", "gWu", D_FF, D_MODEL),
    fill("Bu", "gBu", D_FF, 1),
    fill("Wd", "gWd", D_MODEL, D_FF),
    fill("Bd", "gBd", D_MODEL, 1),
    "    gated.gate_branch().template get<0>().weight() = Wg;",
    "    gated.gate_branch().template get<0>().bias()   = Bg;",
    "    gated.up_branch().weight() = Wu;",
    "    gated.up_branch().bias()   = Bu;",
    "    down.weight() = Wd;",
    "    down.bias()   = Bd;",
    fill("x", "gX", D_MODEL, T),
    # ---- 前向并与 PyTorch 结果比对 ----
    "    dmat inter = gated.forward(x);",
    "    dmat out = down.forward(inter);",
    "    double maxd = 0.0;",
    "    for (int r = 0; r < d_model; ++r)",
    "        for (int t = 0; t < T; ++t) {",
    f"            const double expect = gY[r * T + t];",
    "            const double diff = std::fabs(out(r, t) - expect);",
    "            if (diff > maxd) maxd = diff;",
    "        }",
    '    std::printf("SwiGLU FFN vs PyTorch: d_model=%d d_ff=%d T=%d  max_abs_diff=%.3e  %s\\n",'
    ' d_model, d_ff, T, maxd, maxd < 1e-12 ? "PASS" : "FAIL");',
    "    return maxd < 1e-12 ? 0 : 1;",
    "}",
]

src_path = "/tmp/swiglu_cross.cpp"
bin_path = "/tmp/swiglu_cross"
with open(src_path, "w") as f:
    f.write("\n".join(body))

print(f"generated {src_path}")
subprocess.run(["g++", "-std=c++20", "-O2", "-I.", src_path, "-o", bin_path], check=True)
sys.exit(subprocess.run([bin_path]).returncode)
