FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=4

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CC=/usr/bin/gcc-13 \
    CXX=/usr/bin/g++-13 \
    CUDAHOSTCXX=/usr/bin/g++-13 \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    JASMINE_GPT2_WEIGHTS=/opt/jasmine/models/distilgpt2_weights.bin \
    JASMINE_LLAMA_WEIGHTS=/opt/jasmine/models/tinyllama_weights.bin

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates gnupg \
 && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc-13 g++-13 cmake ninja-build make git \
      libopenblas-dev libgomp1 python3 python3-pip python3-dev \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 130 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade 'pip<26' \
 && python3 -m pip install --no-cache-dir \
      'numpy==1.26.4' \
      'transformers==5.17.0' \
      'sentencepiece>=0.2.0,<0.3'

WORKDIR /opt/jasmine
COPY . /opt/jasmine

# Keep GoogleTest in the image so CMake can rebuild offline.
RUN if [ ! -d /opt/jasmine/third_party/googletest ]; then \
      mkdir -p /opt/jasmine/third_party && \
      git clone --depth 1 --branch v1.14.0 \
        https://github.com/google/googletest.git \
        /opt/jasmine/third_party/googletest && \
      rm -rf /opt/jasmine/third_party/googletest/.git; \
    fi

RUN cmake -S /opt/jasmine -B /opt/jasmine/build-a800 -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCUDAToolkit_ROOT=/usr/local/cuda \
      -DJASMINE_USE_CUDA=ON \
      -DJASMINE_CUDA_ARCHITECTURES=80 \
      -DJASMINE_BUILD_TESTS=ON \
      -DJASMINE_BUILD_BENCH=OFF \
      -DJASMINE_BUILD_EXAMPLES=ON \
      -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/opt/jasmine/third_party/googletest \
 && cmake --build /opt/jasmine/build-a800 -j "${BUILD_JOBS}"

RUN mkdir -p /opt/jasmine/models

WORKDIR /opt/jasmine
