# A800 Docker image

The A800 host driver can run the CUDA 12.8 image. The existing base image

```text
nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
```

is sufficient as a build/runtime base for `sm_80`.

## Build the image

Run from the package root; the build host needs network access:

```bash
docker build \
  --build-arg BUILD_JOBS=20 \
  -f docker/Dockerfile.cuda128 \
  -t jasmine-a800:cuda128 .
```

The image contains the source, toolchain, offline GoogleTest, test binaries and
examples. The default build target is `sm_80`.

## Run tests on A800

```bash
docker run --rm --gpus all jasmine-a800:cuda128 bash -lc '
  ./build-a800/tests/unit_tests --gtest_brief=1 &&
  JASMINE_CUDA_STRESS=0 ./build-a800/tests/cuda_tests --gtest_brief=1 &&
  JASMINE_CUDA_STRESS=1 ./build-a800/tests/cuda_tests --gtest_brief=1 \
    --gtest_filter="CudaDeviceTest.LargeMatrixFusionMatchesHostWhenStressEnabled:CudaDeviceTest.LargeGemmMatchesHostWhenStressEnabled"
'
```

`cuda_tests` does not need model weights. GPT-2/LLaMA alignment tests skip when
their weight files are absent.

## Weights and chat demos

The C++ chat demos need both:

1. a Jasmine `.bin` weight file and its `.json` manifest, exported by
   `tools/export_gpt2.py` or `tools/export_llama.py`;
2. a local HuggingFace tokenizer directory.

Mount them read-only:

```bash
docker run --rm -it --gpus all \
  -v /data/jasmine/models:/opt/jasmine/models:ro \
  -e JASMINE_GPT2_WEIGHTS=/opt/jasmine/models/distilgpt2_weights.bin \
  -e JASMINE_LLAMA_WEIGHTS=/opt/jasmine/models/tinyllama_weights.bin \
  jasmine-a800:cuda128 bash
```

Run a chat demo with a local tokenizer path:

```bash
./build-a800/examples/llama_chat \
  /opt/jasmine/models/tinyllama_weights.bin \
  --model /opt/jasmine/models/tinyllama-tokenizer

./build-a800/examples/gpt2_chat \
  /opt/jasmine/models/distilgpt2_weights.bin \
  --model /opt/jasmine/models/distilgpt2-tokenizer
```

For a fully self-contained offline image, copy the model directory into a
container and commit it as a derived image before `docker save`.
