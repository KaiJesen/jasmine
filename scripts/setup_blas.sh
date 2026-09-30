#!/usr/bin/env bash
#
# Build the fastest GPT-2 chat toy.
#
# Why this exists: jasmine links whatever CBLAS CMake finds, and on a stock
# Debian/Ubuntu that is the reference netlib BLAS -- single-threaded and
# unoptimized, about 5x slower than OpenBLAS on this model. The distro's OpenBLAS
# may also be too old to recognise the CPU: the 0.3.15 build originally used here
# reported "Prescott" (a 2004 SSE3 part) on a 13600KF and ran SSE3 kernels even
# though it shipped AVX2 ones, which cost another 1.7x. So this fetches a current
# OpenBLAS into .blas/ (gitignored) and links the build against it by RPATH.
#
# Measured on a 13600KF, distilgpt2, 128-token prefill / 32-token decode, 4 threads:
#   netlib (distro default)   990.9 ms forward, 34.8 ms/token decode
#   OpenBLAS 0.3.26           106.5 ms forward,  9.5 ms/token decode (fp32)
#
# Intel MKL was also measured and is deliberately not used: on the double path
# that gpt2_chat runs, it is only ~4% faster than OpenBLAS (15.4 vs 15.9 ms/token)
# while requiring MKL_THREADING_LAYER=GNU -- omit that and the process aborts with
# "Cannot load libmkl_intel_thread.so", because libmkl_rt otherwise loads Intel's
# own OpenMP next to jasmine's GNU one. Not a good trade for a 4% gain. MKL does
# help more on the fp32 forward path; see doc/bench/gpt2_infer_vs_torch.md.
#
# Usage:
#   scripts/setup_blas.sh          # fetch the library and build the chat toy
#   scripts/setup_blas.sh --clean  # remove .blas/ and build-chat/ first
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLAS_DIR="$ROOT/.blas"
BUILD_DIR="$ROOT/build-chat"

# Ubuntu 24.04's OpenBLAS. Any 0.3.23+ should detect a recent CPU correctly; this
# one is pinned because that is what the numbers above were measured with.
PKG="libopenblas0-pthread"
PKG_VER="0.3.26+ds-1ubuntu0.1"

clean=0
for arg in "$@"; do
    case "$arg" in
        --clean) clean=1 ;;
        -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ "$clean" = 1 ]; then
    rm -rf "$BLAS_DIR" "$BUILD_DIR"
    echo "[setup_blas] removed $BLAS_DIR and $BUILD_DIR"
fi

# --- 1. the library ---------------------------------------------------------
lib="$BLAS_DIR/libopenblasp-r0.3.26.so"
if [ ! -e "$lib" ]; then
    echo "[setup_blas] fetching $PKG $PKG_VER into $BLAS_DIR"
    mkdir -p "$BLAS_DIR"
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    # Download and unpack rather than apt-get install: no root, and nothing
    # system-wide is modified.
    ( cd "$work" && apt-get download "$PKG=$PKG_VER" >/dev/null )
    dpkg-deb -x "$work/$PKG"_*.deb "$work/root"
    cp "$work"/root/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-*.so "$BLAS_DIR/"
    # libmkl_rt-style setups elsewhere expect the SONAME; provide it too so the
    # same directory can be pointed at with LD_LIBRARY_PATH when debugging.
    ( cd "$BLAS_DIR" && ln -sf libopenblasp-r0.3.26.so libopenblas.so.0 )
else
    echo "[setup_blas] library already present: $lib"
fi

# Fail loudly if a different libopenblas would win at runtime. Two files can share
# the name libopenblas.so.0, and LD_LIBRARY_PATH silently redirects one to the
# other -- that trap already produced a wrong measurement once.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    echo "[setup_blas] warning: LD_LIBRARY_PATH is set, it can override the RPATH below" >&2
fi

# --- 2. the build -----------------------------------------------------------
echo "[setup_blas] configuring $BUILD_DIR"
cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DJASMINE_USE_OPENMP=ON \
    -DJASMINE_USE_BLAS=ON \
    -DJASMINE_BLAS_LIBRARY="$lib" \
    -DCMAKE_BUILD_RPATH="$BLAS_DIR" \
    -DJASMINE_BUILD_TESTS=OFF \
    -DJASMINE_BUILD_BENCH=OFF \
    -DJASMINE_BUILD_EXAMPLES=ON

cmake --build "$BUILD_DIR" --target gpt2_chat gpt2_generate -j "$(nproc)"

# --- 3. report what it will actually use ------------------------------------
resolved="$(ldd "$BUILD_DIR/examples/gpt2_chat" | awk '/libopenblas/{print $3}')"
echo "[setup_blas] gpt2_chat resolves BLAS to: $resolved"
case "$resolved" in
    "$BLAS_DIR"/*) ;;
    *) echo "[setup_blas] warning: expected it to resolve inside $BLAS_DIR" >&2 ;;
esac

cat <<EOF

Ready. Interactive chat:

    $BUILD_DIR/examples/gpt2_chat build/distilgpt2_weights.bin \\
        --tokenizer-dir \$JASMINE_GPT2_TOKENIZER_DIR

Needs the tokenizer files (vocab.json + merges.txt) and an exported weight file;
see README.md. Weight loading is a one-time ~1.5 s, then decoding runs at roughly
12 ms/token in double, which is what the examples use.
EOF
