# Jasmine `cuda-backend` A800 验收包

> 包修订 `r5`：修正压力 GEMM 的测试输入。原输入存在严重正负抵消，host BLAS 与 cuBLAS 的正常 FP32 求和顺序差异会超过 1e-3，并非 TF32 导致。
>
> 包修订 `r4`：FP32 GEMM 显式使用 `CUBLAS_COMPUTE_32F_PEDANTIC` 作为精度硬化；后续复测表明它不是本次压力用例失败的根因。
>
> 包修订 `r3`：CUDA 测试改为按设备算力自适应；A800 不再执行 P4 温控打印，并在 sm_80+ 上默认运行压力用例。
>
> 包修订 `r2`：加入 `docker/Dockerfile.cuda128` 及镜像使用说明，适配宿主机已有的 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`。
>
> 包修订 `r1`：修复 RHEL 8 / glibc 2.28 环境中 `ldd --version | head -n1` 在 `set -Eeuo pipefail` 下可能静默退出的问题；现在脚本会继续执行工具链检查，并在失败时输出源码行号与失败命令。

本包是源码验收包。**不要仅凭包内预编译文件判断目标环境一定可用**，因为目标机是否可构建/运行取决于：

- NVIDIA 驱动版本，以及它支持的最高 CUDA runtime；
- 目标机安装的 `nvcc` / CUDA Toolkit 版本；
- 目标机 `g++` / Clang、CMake、glibc、libstdc++；
- 运行预编译文件时能否找到 `libcudart.so.12`、`libcublas.so.12`、`libcublasLt.so.12`。

因此验收以 `scripts/a800_acceptance.sh` 在目标机本地重建为准；`prebuilt/` 只作为版本完全匹配时的冒烟兜底。

## 0. 版本基线

本包由以下已知环境交叉构建并生成 `sm_80` 代码：

- branch: `cuda-backend`
- commit: `3bfd77d394c3`
- CUDA Toolkit: 12.4.131
- target architecture: `sm_80`
- CMake: 3.28.3
- host compiler: GCC 13.3.0
- glibc: 2.39

如果目标机不是同一套 runtime/编译器，必须从源码重建；源码包已经内置 GoogleTest v1.14.0，构建不需要联网。

## 1. 上传并解包

```bash
tar -xzf jasmine-cuda-backend-a800-3bfd77d394c3.tar.gz
cd jasmine-cuda-backend-a800-3bfd77d394c3
```

## 2. 先做环境检查

```bash
./scripts/a800_acceptance.sh doctor
```

至少记录并确认：

```bash
nvidia-smi
nvcc --version
cmake --version
g++ --version
uname -m
ldd --version | head -1
```

通过条件：

1. `nvidia-smi` 能看到 A800，且驱动状态正常；
2. `nvidia-smi` 显示的 driver CUDA version 与将要使用的 CUDA Toolkit runtime 兼容；
3. `nvcc --list-gpu-arch` 包含 `compute_80`；
4. C++ 编译器满足本分支要求（GCC 13+ / Clang 16+）；
5. CMake >= 3.16。

若 `nvcc` 不在 PATH 中：

```bash
export PATH=/path/to/cuda/bin:$PATH
# 或显式指定
export NVCC=/path/to/cuda/bin/nvcc
```

## 3. 在目标机重建

```bash
./scripts/a800_acceptance.sh build
```

该命令会用本机 `nvcc`、`-DJASMINE_CUDA_ARCHITECTURES=80`、内置 GoogleTest 离线构建：

- `build-a800/tests/unit_tests`：主机端 178 个用例；
- `build-a800/tests/cuda_tests`：CUDA 端 174 个用例。

若目标机有多个 CUDA Toolkit，应显式指定实际要验收的版本，例如：

```bash
NVCC=/usr/local/cuda-12.4/bin/nvcc \
BUILD_DIR=$PWD/build-a800 \
JASMINE_CUDA_ARCHITECTURES=80 \
./scripts/a800_acceptance.sh build
```

## 4. 执行验收

```bash
./scripts/a800_acceptance.sh test
```

脚本会依次运行：

```bash
./build-a800/tests/unit_tests --gtest_brief=1
./build-a800/tests/cuda_tests --gtest_brief=1
JASMINE_CUDA_STRESS=1 ./build-a800/tests/cuda_tests \
  --gtest_brief=1 \
  --gtest_filter='CudaDeviceTest.LargeMatrixFusionMatchesHostWhenStressEnabled:CudaDeviceTest.LargeGemmMatchesHostWhenStressEnabled'
```

验收通过的判定：

- 主机测试无失败；没有上传大权重时，GPT-2/LLaMA alignment 用例可以 `SKIP`；
- CUDA fast suite 无失败；
- 两个 CUDA stress 用例无失败；
- 全程没有 `cudaErrorNoDevice`、驱动拒绝、非法指令、找不到 `libcudart.so.12` / `libcublas.so.12` 等环境错误。

一条命令完成 doctor + build + test：

```bash
./scripts/a800_acceptance.sh accept
```

## 5. 可选：预编译冒烟

```bash
./scripts/a800_acceptance.sh prebuilt-smoke
```

该模式只检查包内使用 CUDA 12.4、GCC 13.3、glibc 2.39 构建的 `sm_80` 二进制。若出现问题，不能据此判定源码有缺陷，应回到第 3 节在本机重建。

## 6. 验收结果回传

建议把以下输出保存回传：

```bash
./scripts/a800_acceptance.sh doctor > target-doctor.txt 2>&1
./scripts/a800_acceptance.sh test > target-test.txt 2>&1
```

并附上：

```bash
sha256sum -c MANIFEST.sha256
```

如果任何一步失败，需要回传完整的 `target-doctor.txt`、`target-test.txt` 和失败前约 50 行日志，尤其要保留驱动版本、`nvcc --version`、构建命令和首个 CUDA API 错误。
