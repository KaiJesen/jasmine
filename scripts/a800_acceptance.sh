#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-accept}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-a800}"
JOBS="${JOBS:-$(nproc)}"
ARCH="${JASMINE_CUDA_ARCHITECTURES:-80}"
NVCC="${NVCC:-}"

info() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

on_error() {
    local status=$?
    local source="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    local line="${BASH_LINENO[0]:-?}"
    printf 'ERROR: acceptance script failed at %s:%s (exit %s): %s\n' \
        "$source" "$line" "$status" "$BASH_COMMAND" >&2
    exit "$status"
}
trap on_error ERR

usage() {
    cat <<'USAGE'
Usage: scripts/a800_acceptance.sh [doctor|build|test|accept|prebuilt-smoke]

Modes:
  doctor          Inspect GPU, driver, nvcc, CMake and C++ compiler; verify sm_80 support.
  build           Build host + CUDA tests locally for the target architecture.
  test            Run host tests, CUDA tests and the two stress cases.
  accept          doctor + build + test (default).
  prebuilt-smoke  Diagnostic only: try the bundled CUDA 12.4/sm_80 binary.
                  This is not authoritative acceptance evidence if the target
                  glibc/libstdc++/CUDA runtime versions differ from the build host.

Useful overrides:
  BUILD_DIR=/path/to/build
  JOBS=16
  NVCC=/usr/local/cuda/bin/nvcc
  JASMINE_CUDA_ARCHITECTURES=80
USAGE
}

find_nvcc() {
    if [[ -n "$NVCC" ]]; then
        [[ -x "$NVCC" ]] || die "NVCC is set but not executable: $NVCC"
        return
    fi
    if command -v nvcc >/dev/null 2>&1; then
        NVCC="$(command -v nvcc)"
        return
    fi
    if [[ -x /usr/local/cuda/bin/nvcc ]]; then
        NVCC=/usr/local/cuda/bin/nvcc
        return
    fi
    die "nvcc not found. Export PATH=/path/to/cuda/bin:\$PATH or set NVCC=/path/to/nvcc"
}

check_cmake_and_compiler() {
    command -v cmake >/dev/null 2>&1 || die "cmake not found (need >= 3.16)"
    local cmake_ver
    cmake_ver="$(cmake --version)" || die "cmake --version failed"
    cmake_ver="${cmake_ver%%$'\n'*}"
    cmake_ver="${cmake_ver##* }"
    if [[ "$(printf '%s\n' 3.16 "$cmake_ver" | sort -V | sed -n '1p')" != "3.16" ]]; then
        die "CMake $cmake_ver is too old; need >= 3.16"
    fi

    local cxx="${CXX:-c++}"
    command -v "$cxx" >/dev/null 2>&1 || die "C++ compiler not found: $cxx"
    local cxx_major cxx_banner
    cxx_major="$("$cxx" -dumpversion | cut -d. -f1)"
    cxx_banner="$("$cxx" --version)" || die "$cxx --version failed"
    cxx_banner="${cxx_banner%%$'\n'*}"
    if (( cxx_major < 13 )); then
        die "C++ compiler $cxx is GCC/Clang major $cxx_major; this branch requires GCC 13+ / Clang 16+ C++20 support"
    fi
    printf 'CMake: %s\nC++ compiler: %s (%s)\n' "$cmake_ver" "$cxx" "$cxx_banner"
}

doctor() {
    info "Host and toolchain"
    uname -a
    local libc_version
    libc_version="$(ldd --version 2>&1)" || die "ldd --version failed"
    printf '%s\n' "${libc_version%%$'\n'*}"
    check_cmake_and_compiler
    find_nvcc
    "$NVCC" --version

    info "CUDA architecture support"
    if ! "$NVCC" --list-gpu-arch 2>/dev/null | grep 'compute_80' >/dev/null; then
        die "The selected nvcc does not list compute_80; it cannot build this A800 bundle"
    fi
    printf 'nvcc supports compute_80: yes\n'
    printf 'Requested target: sm_%s\n' "$ARCH"

    info "GPU and driver"
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
    nvidia-smi || die "nvidia-smi cannot communicate with the NVIDIA driver"
    local gpu_names caps
    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
    caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '. ' | paste -sd, - || true)"
    printf 'GPU(s): %s\n' "$gpu_names"
    [[ -n "$caps" ]] && printf 'Compute capability: %s\n' "$caps"
    if [[ "$gpu_names" != *A800* ]]; then
        warn "No A800 was found in nvidia-smi output; this script is still using ARCH=$ARCH"
    fi
    local non_sm80
    non_sm80="$(printf '%s\n' "$caps" | tr ',' '\n' | grep -vx '80' || true)"
    if [[ -n "$non_sm80" ]]; then
        warn "Detected non-sm_80 compute capability: $non_sm80. Override JASMINE_CUDA_ARCHITECTURES if needed."
    fi
    printf 'Important: nvidia-smi shows the maximum CUDA runtime understood by the driver. It must be compatible with the selected nvcc/toolkit runtime.\n'
}

build_tests() {
    doctor
    info "Configure and build (offline, bundled GoogleTest)"
    cmake -S "$ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$NVCC" \
        -DJASMINE_USE_CUDA=ON \
        -DJASMINE_CUDA_ARCHITECTURES="$ARCH" \
        -DJASMINE_BUILD_BENCH=OFF \
        -DJASMINE_BUILD_EXAMPLES=OFF \
        -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$ROOT/third_party/googletest"
    cmake --build "$BUILD_DIR" -j "$JOBS" --target unit_tests cuda_tests
}

test_all() {
    [[ -x "$BUILD_DIR/tests/unit_tests" ]] || die "unit_tests not built; run: $0 build"
    [[ -x "$BUILD_DIR/tests/cuda_tests" ]] || die "cuda_tests not built; run: $0 build"

    info "Host regression (178 cases; weight-dependent alignment tests may skip)"
    "$BUILD_DIR/tests/unit_tests" --gtest_brief=1

    info "CUDA fast suite (174 cases; two stress cases skipped by default)"
    JASMINE_CUDA_STRESS=0 "$BUILD_DIR/tests/cuda_tests" --gtest_brief=1

    info "CUDA stress cases"
    JASMINE_CUDA_STRESS=1 "$BUILD_DIR/tests/cuda_tests" \
        --gtest_brief=1 \
        --gtest_filter='CudaDeviceTest.LargeMatrixFusionMatchesHostWhenStressEnabled:CudaDeviceTest.LargeGemmMatchesHostWhenStressEnabled'

    info "ACCEPTANCE PASSED"
}

prebuilt_smoke() {
    doctor
    local prebuilt="$ROOT/prebuilt/cuda-12.4-sm80/tests"
    local exe="$prebuilt/cuda_tests"
    [[ -x "$exe" ]] || die "bundled cuda_tests not found: $exe"
    info "Dynamic dependency check"
    ldd "$exe"
    if ldd "$exe" | grep 'not found' >/dev/null; then
        die "Bundled binary has unresolved shared libraries; build from source instead"
    fi
    info "Bundled binary smoke test (not a substitute for source-build acceptance)"
    "$exe" --gtest_brief=1 --gtest_filter='CudaEnvironment.DeviceIsVisible'
}

case "$MODE" in
    doctor) doctor ;;
    build) build_tests ;;
    test) test_all ;;
    accept) doctor; build_tests; test_all ;;
    prebuilt-smoke) prebuilt_smoke ;;
    -h|--help|help) usage ;;
    *) usage; die "unknown mode: $MODE" ;;
esac
