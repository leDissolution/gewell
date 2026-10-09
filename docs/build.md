# Build

Build on Linux with an NVIDIA CUDA toolkit that supports `sm_120a`. The
development toolchain uses CUDA 13.1. The executable targets compute capability
12.0, including RTX PRO 6000 Blackwell and RTX 5090.

In theory, there is nothing that prevents you from building and running it on Windows, but it was not tested yet.

Required tools and libraries:

- CMake 3.25 or newer, a C++17 compiler supported by your CUDA toolkit, and Ninja
  or Make.
- CUDA development libraries, including cuBLAS and cuBLASLt, and a compatible
  NVIDIA driver for execution.
- cuDNN 9 development headers and libraries for the EmbeddingGemma 2 audio tower.
- OpenSSL, libpng, libjpeg, and FFmpeg 8 development packages, plus `pkg-config`
  and the `patch` command. Video decoding uses libavformat, libavcodec, libavutil
  and libswscale; audio preparation also uses libswresample.

On Debian/Ubuntu, the non-CUDA packages can be installed with:

```bash
sudo apt-get install build-essential cmake ninja-build patch pkg-config \
  libssl-dev libpng-dev libjpeg-dev libavformat-dev libavcodec-dev \
  libavutil-dev libswscale-dev libswresample-dev
```

The distribution must supply FFmpeg 8 development packages. Installing the
`ffmpeg` executable alone does not provide the headers or `.pc` files used by
CMake. Check discovery with:

```bash
pkg-config --modversion libavformat libavcodec libavutil libswscale libswresample
```

For a separate FFmpeg development installation, add
`-DCMAKE_PREFIX_PATH=/path/to/ffmpeg/prefix` when configuring. This is saved in
the build cache for later `cmake --build` calls; keep the installation outside
temporary directories. The prefix must contain usable headers, libraries and
pkg-config metadata.

If cuDNN is outside the system search path, also add
`-DCUDNN_ROOT=/path/to/cudnn`, containing `include` and `lib` or `lib64`.
The development environment's Python package can supply it at
`.venv/lib/python3.12/site-packages/nvidia/cudnn` when that package is installed.
Production execution links the native cuDNN library and does not invoke Python.

CMake downloads XGrammar and CUTLASS sources during
configuration. To use already downloaded
sources, set `FETCHCONTENT_SOURCE_DIR_XGRAMMAR` and
`FETCHCONTENT_SOURCE_DIR_CUTLASS` to their source directories. The XGrammar
directory must already have `vendor/xgrammar/unicode-escapes.patch` applied
when bypassing the download step.

## Engine

Run from the source checkout:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DGEWELL_BUILD_DIAGNOSTICS=OFF
cmake --build build --target gewell --parallel 4
build/gewell --help
```

If CUDA is outside the compiler's search path, add
`-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` at configuration time. Lower the
parallel job count if compilation exhausts host memory.

Optionally install the executable into a chosen prefix:

```bash
cmake --install build --prefix "$HOME/.local"
```

The executable dynamically links CUDA and native system libraries. Continue with
[model preparation](models.md) and [server launch](launch.md).

## Tests

The CPU libraries and their tests can be built without CUDA or model weights:

```bash
cmake -S . -B build/host -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGEWELL_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build/host --parallel 4
ctest --test-dir build/host --output-on-failure
```

The tokenizer, grammar, and HTTP API tests skip when their reference tokenizer is
absent. To run them, add `-DGEWELL_TOKENIZER_DIRECTORY=/path/to/serving-snapshot`
using `tokenizer.json` for the reference fixtures.
CPU mode builds libraries and tests; it does not build a CPU inference engine.

For CUDA tests, configure the main build with `-DBUILD_TESTING=ON`, build it,
then run `ctest --test-dir build --output-on-failure` on the supported GPU.
`-DGEWELL_BUILD_DIAGNOSTICS=ON` additionally builds `gewell_diagnostics` for
fixed oracle captures and profiling; those commands need their own fixtures.

The retained Python artifact tests use Python 3.10+ and NumPy:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy pytest
.venv/bin/python -B -m pytest tests/test_*.py
```

These Python tests exercise conversion, packing, corruption rejection, and
structural compatibility with small fixtures. They do not load a full model on a GPU.
The optional real-asset copy test skips unless `GEWELL_TEST_SERVING_SNAPSHOT`
points to a local serving directory; set it to include that check.
