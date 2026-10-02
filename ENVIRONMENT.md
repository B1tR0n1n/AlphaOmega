# Environment

Exact environment used for AlphaOmega harness runs and judging. Recorded 2026-10-02 on workstation `sable`.

## Hardware

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 9 9950X3D (16C/32T) |
| RAM | 128 GB DDR5-6000 (120 GiB visible) |
| GPU 0 | NVIDIA GeForce RTX 5090, 32 GB, sm_120, VBIOS 98.02.2E.40.EF, 600 W limit |
| GPU 1 | NVIDIA GeForce RTX 5070 Ti, 16 GB, sm_120, VBIOS 98.03.58.40.A3, 300 W limit |

## OS & driver

| Component | Version |
|---|---|
| OS | Ubuntu 25.10 |
| Kernel | 6.17.0-41-generic |
| NVIDIA driver | 590.48.01 (open kernel module), max CUDA 13.1 |
| CUDA toolkit (nvcc) | 12.9 (V12.9.86), `/usr/local/cuda` |
| System gcc | 15.2.0 (llama.cpp built with gcc-14 / g++-14 14.3.0) |
| CMake | 3.31.6 |

## Python environment (`~/ml-env`)

Plain `venv` on system Python 3.13.7 (`python3 -m venv ~/ml-env`), no conda.

| Package | Version |
|---|---|
| torch | 2.10.0+cu129 (official pip wheel, **not** a source build) — git `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` |
| CUDA (torch runtime) | 12.9 |
| cuDNN | 9.10.2 (91002) |
| torch arch list | sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120, compute_120 |
| triton | 3.6.0 |
| transformers | 5.3.0 |
| peft | 0.18.1 |
| accelerate | 1.13.0 |
| bitsandbytes | 0.49.2 |
| datasets | 4.8.2 |
| torch_geometric | 2.7.0 |
| numpy | 2.3.5 |
| openai | 2.30.0 |
| anthropic | 0.88.0 |

Not installed: `flash_attn`, `xformers`, `trl`, `mamba_ssm`.

**bitsandbytes check (2026-10-02):** an NF4 `Linear4bit` forward and backward pass in bf16 on the RTX 5090 ran cleanly, with finite gradients. 4-bit quantization works on Blackwell (sm_120) with this stack. A short QLoRA training run has not been tested yet.

## llama.cpp (built from source, local inference / judge)

| Item | Value |
|---|---|
| Path | `~/llama.cpp` |
| Commit | `2e4a6edd4ac6ebb2459fca373249298291acfc5e` (tag `b8389`, 2026-03-17), clean tree |
| Binary | `llama-server` version 8389, built Mar 16 2026, GNU 14.3.0 |
| Backends | `ggml-cpu`, `ggml-cuda` |

Build configuration (from `build/CMakeCache.txt`, non-default/relevant only):

```
CMAKE_BUILD_TYPE=Release
CMAKE_C_COMPILER=/usr/bin/gcc-14
CMAKE_CXX_COMPILER=/usr/bin/g++-14
CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc   # CUDA 12.9
GGML_CUDA=ON
GGML_NATIVE=ON          # CPU + CUDA arch auto-detected (no explicit CMAKE_CUDA_ARCHITECTURES)
GGML_CUDA_FA=ON
GGML_CUDA_GRAPHS=ON
GGML_OPENMP=ON
GGML_CCACHE=ON
BUILD_SHARED_LIBS=ON
```

Reproduce:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 2e4a6edd4ac6ebb2459fca373249298291acfc5e
CC=gcc-14 CXX=g++-14 cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --config Release -j 32
```

## Local models (`/mnt/vault/models`)

| Model | File | Size |
|---|---|---|
| Nemotron-3-Nano-30B-A3B | `nemotron-nano/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf` | 22.8 GB |
| Qwen3.5-35B-A3B | `qwen3.5-35b/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` | 22.2 GB |
| Qwen3.5-122B-A10B | `qwen3.5-122b/Qwen3.5-122B-A10B-UD-Q2_K_XL.gguf` | 41.8 GB |

## API judges / generators

| Role | Model |
|---|---|
| Generators | `claude-opus-4-7`, `claude-haiku-4-5` (Anthropic API) |
| Judge 1 | `gpt-4o` (OpenAI API) |
