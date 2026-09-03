# Diffusion Compiler

Diffusion Compiler is a C++20 compiler and native runtime for diffusion
inference and training. A model frontend turns a checkpoint into verified
DiffIR; the compiler plans execution; one shared runtime lowers the plan to
the installed hardware backend. The accepted executables link no libtorch and
invoke no Python worker. PyTorch is used only as a development oracle for
fixtures and controlled comparisons.

```text
checkpoint + frontend  ->  verified DiffIR  ->  compiler / optimizer / memory planner
                                                          |
                                             shared native C++ runtime
                                                          |
                     NVIDIA backend: CUDA Driver API, NVRTC, cuBLASLt, CUTLASS INT8, cuDNN
```

## What it does

- **DiffIR.** Checksummed, backend-neutral graph programs with tensor
  interchange, structural verification, fingerprints, transformation
  provenance, and replayable execution plans (`.difir`, `.diftensor`,
  `.difbind`, `.difplan`, `.diftrain`).
- **Compiler.** Residency planning for resident and streamed weights, fusion,
  precision policy, a physical-format registry (`fp32`, `bf16`, `fp16`,
  `fp8-e4m3`, `int8-convrot`, `int4-group`, `int5-group`, SquareQ hooks, and
  Blackwell `mxfp8-block-scaled`), and an optimizer search whose acceptance
  order is fixed: verify, execute, numerical gate, memory, then timing.
- **Runtime.** One CPU/CUDA executor with typed tensors and views,
  lifetime-aware allocation, asynchronous pinned staging, reusable workspaces,
  CUDA event profiling, an on-disk PTX cache keyed on source, architecture,
  NVRTC version, and options, and attributed tracing with NVTX ranges.
- **Operators.** BF16/F16/F32 and integer execution, attention with GQA and
  masks, multi-axis rotary encoding, normalization, modulation, activations,
  layout operations, convolution, VAE primitives, and PNG/MP4/WAV handoff.
- **Training.** Reverse-mode autodiff, gradient accumulation, AdamW state
  transitions, checkpoint/resume, and LoRA through the same DiffIR and runtime
  used for inference.
- **Tools.** A command-line suite for compiling, inspecting, running, tuning,
  benchmarking, tracing, bisecting, quality-gating, and regression-testing
  programs, plus per-model frontends. See [USAGE.md](USAGE.md).

## What it runs today

| Model | Task | GPU | Precision route |
|---|---|---|---|
| MiniMax-H3 | FL2VA: prompt + two keyframes to H.264/AAC MP4 with audio | RTX 3090 Ti | ConvRot INT8 projections; exact attention on the first three of seven evaluations, INT8 attention on the rest |
| Krea 2 Turbo | Prompt to 1024x1024 PNG | RTX 3090 Ti | BF16, bit-identical trajectory |
| FLUX.2 [klein] Base 9B | Prompt to 1024x1024 PNG | RTX 5080 | ConvRot W8A8 plus INT8 weight-only, MXFP8 hooks |

Each frontend also carries its conditioners and VAEs natively: Qwen3-VL
vision/text conditioning, the H3 video encoder, video VAE, and audio VAE, the
Qwen-Image VAE for Krea 2, and the FLUX.2 VAE. Model checkpoints and generated
artifacts are not distributed in this repository.

**Hardware and software.** NVIDIA RTX 3090 Ti (Ampere, `sm_86`) and RTX 5080
(Blackwell, `sm_120a`) are the supported targets. CUDA Toolkit 12.4 or newer
with cuDNN builds the NVIDIA backend; cuBLASLt 12.8 or newer is needed only
for the MXFP8 block-scaled Linear on Blackwell, and a program that carries it
fails closed at prepare on any other host. FFmpeg with ffprobe is required for
video muxing and verification. The OpenCL plugin is a reference path, not AMD
support.

## How it decides

- **The runtime discovers hardware facts; the compiler chooses policy.**
  `difprobe` reports architecture, SM count, tensor-core dtypes, VRAM, and
  library versions as a static target profile separate from the dynamic
  budget. Format legality, residency, precision, fusion, and staging are
  compiler decisions recorded in the plan and replayed exactly. The runtime
  never chooses a policy heuristically.
- **Frontends own model semantics.** Tensor packing, block ordering,
  checkpoint names, conditioning policy, and scheduler rules stay in the
  frontend. Generic operations stay in DiffIR and the shared runtime. There is
  no model-specific executor and no second tensor framework.
- **One runtime.** Inference, training, every model, and every tool execute
  through the same verifier, planner, executor, allocator, attention
  implementation, and backend.
- **Fail closed.** A plan bound to a target refuses to replay when any
  compatibility condition changes. Fused inference plans that would remove a
  tensor a backward opcode reads are refused at prepare. Deterministic
  algorithm policies fail when no qualifying algorithm exists. A check that
  cannot run is `BLOCKED`, never `PASS`.
- **Parity before performance.** Fixed inputs and creator outputs gate every
  primitive, block, trajectory, scheduler, and decoder before an optimization
  is admitted. Acceptance bars are trusted inputs the search cannot change.
- **Decoded quality is a gate.** Finite tensors and a high cosine do not
  replace looking at the image, watching the video, or listening to the
  audio. `difquality` passes only with numeric admission and a recorded human
  review.
- **The complete prompt-to-saved-output wall is the only speed metric.**
  `difbench` times a literal prompt through the saved PNG or MP4 across fresh
  processes. Stage timings are diagnostics. A local kernel win that regresses
  the complete wall is removed.

## Measured results

Complete product walls on the same GPU and frozen workload. Compare within a
row only. See [PERFORMANCE.md](PERFORMANCE.md) for workloads, gates, stage
timings, memory, and comparator disclosures.

| Model and product boundary | GPU | Native C++ | Matched baseline | Result |
|---|---|---:|---:|---:|
| FLUX.2 [klein] Base 9B, prompt to PNG | RTX 5080 | **52.809 s** | 99.242 s ComfyUI | **1.879x faster** |
| Krea 2 Turbo, prompt to PNG | RTX 3090 Ti | **26.58 s** | 59.14 s framework | **2.225x faster** |
| MiniMax-H3 FL2VA, prompt to MP4 | RTX 3090 Ti | **89.77 s** | 81.095 s ComfyUI | **1.107x slower** |

FLUX.2 saves 46.433 seconds (46.788%) on the frozen 1024x1024, 50-step
workload and passes the unchanged numerical and visual gates. It meets the
approved near-55-second target; it is not labeled 2x.

The H3 recipe runs exact attention on the first three of its seven
evaluations and INT8 attention on the rest. Decoded against exact attention
on every evaluation, same inputs, the INT8-only route (which the ComfyUI
comparator also runs) measured worst-frame PSNR 24.5 dB, SSIM 0.850, audio
SNR 1.7 dB; the mix measures 32.3 dB, 0.958, 12.9 dB, passes the video bars,
and passed the owner's listening review. It costs 10.9 seconds over the
retired INT8-only recipe (78.88 s, 1.028x faster than the comparator) and is
8.7 seconds (10.7%) slower than that comparator, which has not been measured
at matched exactness. No H3 speed claim is made.

## Build and test

Requirements: CMake 3.24 or newer, a C++20 compiler, and Ninja or another
CMake-supported build tool. CUDA Toolkit and cuDNN enable the NVIDIA backend.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDIF_ENABLE_CUDA=ON \
  -DDIF_ENABLE_CUDNN=ON \
  -DDIF_CUDA_ARCHITECTURES=native
cmake --build build -j2
ctest --test-dir build --output-on-failure -j1
```

`DIF_CUDA_ARCHITECTURES` defaults to `native`. Blackwell RTX 50-series builds
use `-DDIF_CUDA_ARCHITECTURES=120` with a toolkit that supports SM 120.

CPU-only development build:

```sh
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDIF_ENABLE_CUDA=OFF \
  -DDIF_ENABLE_CUDNN=OFF \
  -DDIF_ENABLE_OPENCL=OFF
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure -j1
```

`ctest` runs 24 gates: IR and runtime tests, FLUX.2 frontend and prompt
tests, LoRA, autodiff, cuDNN attention backward, and DiT backward tests, fusion and epilogue byte-identity tests,
optimizer, owned-attention, target, telemetry, quality, bench, noise, format, bisect, difopt
CLI, plugin ABI, OpenCL plugin, and tokenizer tests. The tiered regression
suite in `perf/regress/suite.json` adds smoke checks and the model-level
prompt-to-output recipes; run it with `difregress`.

## Usage

Every tool prints its usage when run without arguments. Command forms, flags,
the benchmark recipe and regression suite formats, environment variables, and
the runtime knobs that matter are documented in [USAGE.md](USAGE.md).

## Repository layout

```text
include/dif/       Public C++ and backend ABI headers
src/               IR, compiler, runtime, frontend, support, and training code
backends/opencl/   OpenCL reference plugin
tools/             Native command-line programs
tests/             CPU, CUDA, plugin ABI, numerical, and product-path gates
perf/              Benchmark recipes, regression suite, baselines, fixtures
scripts/           Development-oracle fixture and benchmark scripts
third_party/       Vendored dependencies and license notices
```

Build success alone is never a quality claim. Model admission requires
numerical gates and inspection of the final decoded artifact.

## License

Copyright © 2026 Alex Bukoski. All rights reserved.

Diffusion Compiler is currently under active development. The source code is publicly available for viewing, evaluation, research, and development purposes, but no general open-source license has been granted at this time.

Commercial licensing, redistribution, incorporation into commercial products or services, and other commercial uses may be made available under separate licensing terms.

The licensing model for Diffusion Compiler is still being evaluated and may change as the project develops.

For commercial licensing inquiries, please contact the project owner.

### Contributions

Please note that the project is in an early stage of development. Before submitting substantial code contributions, contributors should understand that a formal contributor licensing policy may be introduced in the future to preserve the project's ability to offer both community and commercial licensing options.
