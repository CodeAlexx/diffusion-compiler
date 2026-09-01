# Diffusion Compiler

Diffusion Compiler is an independent C++20 compiler and native runtime for
diffusion inference and training. Model frontends produce verified DiffIR;
the compiler plans execution and the shared runtime lowers that plan to the
installed hardware backend.

```text
model checkpoint + frontend
          |
          v
      verified DiffIR
          |
          v
 compiler / optimizer / memory planner
          |
          v
 shared native C++ runtime
          |
          v
 NVIDIA backend: CUDA Driver API, NVRTC, cuBLASLt, cuDNN, custom kernels
```

PyTorch is used only as a development oracle for fixtures and controlled
comparisons. The accepted native executables do not link libtorch or invoke a
Python worker. Model checkpoints and generated artifacts are not distributed
in this repository.

## What exists today

- Checksummed DiffIR programs, tensor interchange, structural verification,
  fingerprints, transformation provenance, and replayable execution plans.
- Shared CPU and NVIDIA runtimes with typed tensors, views, resident and
  streamed weights, lifetime-aware allocation, asynchronous staging, reusable
  workspaces, CUDA event profiling, NVRTC kernels, cuBLASLt, and cuDNN.
- Generic BF16/F16/F32 and selected integer execution, attention with GQA and
  masks, multi-axis rotary position encoding, normalization, modulation,
  activations, layout operations, convolution, VAE primitives, and PNG/media
  handoff.
- Reverse-mode autodiff, gradient accumulation, AdamW state transitions,
  checkpoint/resume, and LoRA training scaffolding through the same DiffIR and
  runtime used for inference.
- MiniMax-H3 as the first production-scale video/audio proving frontend.
- Krea 2 as model family number two, reusing the same verifier, planner,
  executor, allocator, attention implementation, and NVIDIA backend.

H3 and Krea-specific tensor packing, block ordering, checkpoint names,
conditioning policy, and scheduler rules stay in their frontends. Generic
operations stay in DiffIR and the shared runtime; there is no Krea-specific
executor or second tensor framework.

## Krea 2 Turbo BF16 baseline

The frozen baseline uses the official Krea 2 Turbo checkpoint and creator
recipe on one RTX 3090 Ti:

- 1024x1024 output, 4096 image tokens, 512 text tokens
- 28 MMDiT blocks, width 6144, 48 query heads, 12 KV heads
- BF16 production math
- 8 Euler steps, CFG disabled, `mu=1.15`
- identical checkpoint, prompt, seed, initial latent, schedule, and conditioning
  boundary for the compared denoisers

The native path and the strict ComfyUI/PyTorch development comparator measured:

| Measurement | Native C++ | ComfyUI/PyTorch BF16 |
|---|---:|---:|
| One-time denoiser preparation | 1.227 s | included in cold first step |
| Cold first denoise step | 4.332 s | 28.287 s |
| Hot denoise step median | 4.309 s | 2.000 s |
| Complete 8-step denoise | 34.477 s | 42.316 s |
| Peak NVML GPU use / host RSS | not captured with the same method | 25.487 GB / 27.028 GB |

The instrumented native 8-step denoise loop is **1.227x faster**. Including
native's separately measured preparation gives 35.703 s, still **1.185x
faster** than the strict Comfy denoise loop, though the two runtimes expose
setup at different boundaries. The warmed ComfyUI loop is **2.155x faster per
repeated step**, while its cold first step is **6.53x slower**. This split is
the measured optimization target: preserve the native cold-start advantage
while improving prepared-plan residency, prefetch, launch count, layout
persistence, workspace reuse, GEMM planning, and attention execution across
all 28 blocks and 8 steps. No complete UI-to-image end-to-end speed ratio is
claimed by this denoiser benchmark.

The strict comparator uses the creator's padding mask and the same
post-TextFusion conditioning boundary. Its final latent versus the exact
creator/native trajectory measured cosine `0.997472`, relative L2 `0.070970`,
and zero nonfinite values. The stock ComfyUI Krea frontend was also measured,
but it omits that creator mask at the main blocks in the checked revision and
therefore is retained only as a product-speed observation, not a numerical
parity arm. No threshold was relaxed to admit it.

Qwen-Image VAE decode through the native runtime measured 0.977 s after a
separate 4.519 s one-time plan build. Against the creator decode, the native
PNG measured 56.81 dB PSNR and 0.999073 SSIM and passed visual inspection.

These are source-faithful baseline numbers, not an optimized-plan claim. The
next performance phase begins from this frozen baseline; it does not change
the prompt, seed, BF16 quality class, scheduler, or output gate.

## Build and test

Requirements are CMake 3.24 or newer, a C++20 compiler, and Ninja or another
CMake-supported build tool. CUDA Toolkit and cuDNN enable the NVIDIA backend.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDIF_ENABLE_CUDA=ON \
  -DDIF_ENABLE_CUDNN=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure -j1
```

For a CPU-only development build:

```sh
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDIF_ENABLE_CUDA=OFF \
  -DDIF_ENABLE_CUDNN=OFF \
  -DDIF_ENABLE_OPENCL=OFF
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure -j1
```

## Command-line surface

Core tools include:

- `difc`, `difinspect`, and `difrun` for DiffIR construction, inspection, and
  execution.
- `difweights`, `difcast`, `difcompare`, and `difquant` for model storage and
  numerical gates.
- `difschedule` for authenticated flow schedules.
- `difopt` and `diftune` for gated plan search and backend candidate tests.
- `diftrain` for compiled forward/backward, optimizer state, and resume.
- `difh3layout`, `difh3infer`, `difh3media`, and `difvaedecode` for the H3
  proving path.
- `difkrea2block`, `difkrea2denoise`, `difkrea2sample`, and `difkrea2vae` for
  the Krea 2 source-faithful gates and native image path.

Run a tool without arguments for its current usage. Binary `.difir`,
`.diftensor`, `.difbind`, and `.diftrain` files are canonical; CLI text is an
operations surface, not a second IR format.

## Repository layout

```text
include/dif/       Public C++ and backend ABI headers
src/               IR, compiler, runtime, frontend, support, and training code
backends/opencl/   OpenCL reference plugin
tools/             Native command-line programs
tests/             CPU, CUDA, plugin ABI, numerical, and product-path gates
scripts/           Development-oracle fixture and benchmark scripts
third_party/       Vendored dependencies and license notices
```

The canonical IR is backend-neutral. NVIDIA is the first production backend;
the OpenCL reference path does not constitute untested AMD support. Build
success alone is never a quality claim: model admission requires numerical
gates and inspection of the final decoded artifact.
