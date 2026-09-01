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

## Krea 2 Turbo BF16 benchmark

The frozen baseline uses the official Krea 2 Turbo checkpoint and creator
recipe on one RTX 3090 Ti:

- 1024x1024 output, 4096 image tokens, 512 text tokens
- 28 MMDiT blocks, width 6144, 48 query heads, 12 KV heads
- BF16 production math
- 8 Euler steps, CFG disabled, `mu=1.15`
- identical checkpoint, prompt, seed, initial latent, schedule, and conditioning
  boundary for the compared denoisers

The frozen pre-optimization native path, the first admitted whole-system plan,
and the strict ComfyUI/PyTorch eager-mode development comparator measured:

| Measurement | Native frozen | Native admitted plan | ComfyUI/PyTorch BF16 |
|---|---:|---:|---:|
| One-time denoiser preparation | 1.227 s | 1.224–1.230 s | included in cold first step |
| Cold first denoise step | 4.332 s | 3.565–3.599 s | 28.287 s |
| Hot denoise step median | 4.309 s | 2.249–2.258 s | 2.000 s |
| Complete 8-step denoise | 34.477 s | 19.364–19.379 s | 42.316 s |
| Preparation + 8-step denoise | 35.703 s | 20.594–20.602 s | 42.316 s |
| Peak NVML GPU use / host RSS | not captured together | 23,364 MiB / 26,771,764 KiB | 25.487 GB / 27.028 GB |

Across two accepted repetitions, the admitted native plan is **2.054x faster**
including its explicit preparation, and its denoise loop is **2.184x faster**.
The warmed ComfyUI iteration remains about **1.12x faster** than the admitted
native hot iteration; the native end-to-end advantage comes from retaining its
small setup boundary and populating compiler-selected resident weights once
during the first semantic evaluation instead of paying the framework's large
cold step.

The admitted plan keeps 22.072 GB of reusable weights resident, streams 2.775
GB, selects residency largest-first under an explicit 22,000 MiB plan budget,
aliases immutable reshapes, and uses four bounded host staging participants.
Krea modulation now uses the shared `AffineLastDim` semantic instead of
materializing redundant full-sequence broadcasts. All eight recorded states
and the final latent remain bit-identical to the frozen creator trajectory; no
quality or numerical threshold changed. This is the first 2x checkpoint, not
the final optimization target. No complete prompt-to-image speed ratio is
claimed by this denoiser-only table.

A separate literal-prompt-to-PNG validation measured every external stage with
fresh processes and a warm operating-system page cache:

| External stage | Native C++ | Creator / ComfyUI / PyTorch BF16 |
|---|---:|---:|
| Tokenizer + Qwen3-VL | 2.28 s | 9.05 s |
| TextFusion | 1.29 s | 7.52 s |
| Prepared 8-step denoise | 19.54 s | 34.26 s |
| Qwen-Image VAE + PNG | 3.43 s | 8.31 s |
| **Literal prompt to PNG** | **26.58 s** | **59.14 s** |

That is a measured **2.225x complete-chain speedup** on the RTX 3090 Ti. The
native run recorded zero filesystem input, so this is explicitly a warm-page-
cache result, not a cold-disk claim. A cold-filesystem native diagnostic took
55.58 s, dominated by 27.42 s of Qwen weight page-in; it is preserved as a
separate I/O result rather than folded into the admitted number. The validated
native denoiser remained bit-identical to the creator trajectory, the VAE gate
passed at cosine `0.99999339` and relative L2 `0.00363512` after clamping, and
the inspected PNG retained SHA-256
`eea79ee7d84a703235481b0e1859ca087fa20d40304aa0243d9929da8333fbfd`.

The framework comparator is ordinary ComfyUI/PyTorch eager execution. It does
not use `torch.compile` or Inductor kernel fusion; that would be a separate
matched benchmark.

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

The original numbers remain the frozen pre-optimization baseline. Successor
plans continue to use the same prompt, seed, BF16 quality class, scheduler,
and output gate.

## MiniMax-H3 ConvRot INT8 development checkpoint

The current H3 optimization checkpoint uses the project-owned native ConvRot
INT8 projection cache with exact cuDNN attention. It does not use Sage, SOL,
SLA, ComfyUI weights, or an approximate-attention path. Measurements below are
from one RTX 3090 Ti at its deliberate 300 W power limit and a real-dimensional
Ref2VA execution fixture: 1024x576, 124 delivered frames, sequence length
23,185, 50 transformer blocks, and eight schedule points / seven denoiser
evaluations.

| Measurement | Exact streamed BF16 | Resident ConvRot INT8 + exact cuDNN |
|---|---:|---:|
| First measured denoiser evaluation | 30.998 s device / 33.230 s wall | 19.872 s device |
| Hot denoiser evaluation median | not yet captured | 18.082 s |
| Seven-evaluation denoise | not yet captured | 128.510 s device / 131.023 s wall |
| Runtime-accounted VRAM requirement | 6.332 GB | 22.941 GB |

An explicit auxiliary-residency diagnostic measured an 18.053 s hot median
with a 23.008 GB accounted requirement and 1.231 GB reported free. It removed
repeated hot H2D staging but improved only about 29 ms, so transfer is no longer
the main hot-step bottleneck. Relative to the 30.998 s streamed-BF16 device
checkpoint, the current best repeated evaluation is 1.717x faster. This is a
development comparison, not yet a matched prompt-to-video claim.

The native 50-block ConvRot cache is 19,279,810,048 bytes and fits without OOM.
The scaled CUTLASS projection path reduced reusable scratch by 176,160,768
bytes. Its complete seven-evaluation video latent, audio rows, and final audio
latent are bit-identical to the prior accepted ConvRot control: cosine 1,
relative L2 0, norm ratio 1, zero nonfinites, and zero bit mismatches.

Nsight attributes 67.9% of hot device kernel time to exact cuDNN attention and
24.5% to the scaled INT8 projection GEMMs. A 15.229 s candidate using
approximate CK attention crossed the provisional 2x per-evaluation timing bar
but failed the unchanged trajectory gate (video relative L2 `0.02756`, final
audio-latent relative L2 `0.06401`) and was rejected.

The required literal-input FL2VA and Ref2VA prompt-to-video, decoded visual,
and audio acceptance runs remain open. No 2x prompt-to-video result is claimed
until both complete paths pass those gates.

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
