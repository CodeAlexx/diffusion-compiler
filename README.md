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
- FLUX.2 [klein] Base 9B as a fully native prompt-to-PNG frontend on RTX 5080,
  reusing the same DiffIR, prepared executor, residency planner, quantization
  semantics, cuBLASLt/cuDNN backend, VAE primitives, and PNG path.

Model-specific tensor packing, block ordering, checkpoint names, conditioning
policy, and scheduler rules stay in their frontends. Generic operations stay
in DiffIR and the shared runtime; there is no model-specific executor or
second tensor framework.


## Performance

The main benchmark summary stays compact here. See
[PERFORMANCE.md](PERFORMANCE.md) for the speed charts, frozen workloads,
quality gates, stage timings, memory data, and comparator disclosures.

| Model and product boundary | GPU | Native C++ | Matched baseline | Result |
|---|---|---:|---:|---:|
| FLUX.2 [klein] Base 9B, prompt to PNG | RTX 5080 | **52.809 s** | 99.242 s ComfyUI | **1.879x faster** |
| Krea 2 Turbo, prompt to PNG | RTX 3090 Ti | **26.58 s** | 59.14 s framework | **2.225x faster** |
| MiniMax-H3 FL2VA, prompt to MP4 | RTX 3090 Ti | **78.88 s** | 81.095 s ComfyUI | **1.028x faster** |

FLUX.2 saves 46.433 seconds (46.788%) on the frozen 1024x1024, 50-step
workload and passes the unchanged numerical and visual gates. It meets the
approved near-55-second target; it is not labeled 2x.

The H3 result saves 2.215 seconds (2.732%) on the complete seven-evaluation
FL2VA chain. Decoded parity is not fully accepted and the result remains far
from the 2x ceiling, so no H3 2x or final-quality claim is made.

## Hardware target and observability foundation

Phase A adds one model-neutral hardware target layer shared by the compiler,
runtime, and tools. `difprobe` reports static target capability separately from
the dynamic budget for the current execution:

```sh
build/difprobe
build/difprobe --json
```

On the development RTX 3090 Ti it identifies Ampere `sm_86`, 84 SMs,
25,248,202,752 bytes total VRAM, BF16/FP16/INT8 tensor-core support, no FP8 or
NVFP4 support, and the installed CUDA driver/runtime, cuBLASLt, and cuDNN
versions. Free/usable VRAM, host RAM, pinned staging, and workspace are dynamic
budget fields and are not folded into static model semantics.

The JSON document uses the versioned `diffusion-compiler-telemetry` schema.
Optimization plans now serialize as version 2 with optional compiler revision,
target-capability fingerprint, runtime-budget class, precision policy, and
minimum VRAM/workspace requirements. Target-bound replay fails closed when any
required compatibility condition changes; version-1 plans remain readable as
explicitly unbound historical plans.

## Truth surfaces: difbench, diftrace, difplan

Phase B adds the agent-facing measurement and explanation surfaces on top of
the Phase A hardware layer. All three emit the same versioned
`diffusion-compiler-telemetry` JSON: one `schema`/`kind`/`provenance` head,
shared `hardware`, `runtime_budget`, and launch-telemetry sections, and one
attribution vocabulary (`include/dif/telemetry/vocabulary.hpp`).

- `difbench run RECIPE.json --workdir DIR [--json]` is the canonical
  literal-prompt-to-saved-output boundary. A recipe
  (JSON, kind `diffusion-compiler-benchmark-recipe`; see `perf/recipes/`)
  names the fresh-process chain for one workload with `after` dependencies; difbench owns the timer, dependency-ordered process stages with
  per-stage rusage, page-cache residency of the declared model files
  (cold/warm/mixed), NVML peak VRAM and power under the configured cap, and
  native PNG/MP4 verification of the saved output. Total complete wall is the
  only acceptance metric; stage timings are diagnostics. The frozen H3 FL2VA
  and Krea 2 Turbo chains are transcribed as `perf/recipes/*.json` and are
  validation fixtures, not part of the tool.
- Stage cache. A recipe stage may declare `"cache": {"key": [files...],
  "outputs": [files...]}`; with `difbench run ... --stage-cache DIR` (also
  `diftrace recipe --stage-cache DIR`) a stage whose key (its argv with the
  work directory normalized, plus every key file's path, size, mtime outside
  the work directory and SHA-256 up to 64 MiB) is already in DIR has its
  outputs restored and its process never started. Every stage record carries
  `cache_status` (`disabled`, `none`, `miss`, `hit`, `store-failed`) and
  `cache_key`, and the run conditions list the hits, so a cached wall is
  never mistaken for a cold one. Without `--stage-cache` the declarations are
  inert. The H3 recipe caches presentation, vision and conditioner on the
  prompt, keyframes, program files and the conditioner ConvRot cache;
  measured on the 3090 Ti the three stages go from 9 s warm (240 s cold) to
  0.06 s on a repeat prompt, and the cached conditioning makes the denoiser
  bit-reproducible run to run.
- `diftrace recipe|program|merge` attributes where total wait goes. The shared
  runtime now records attributed events at its centralized submission sites
  (GEMM, attention, convolution, generated kernels, H2D/D2H/D2D bytes, staging,
  waits, synchronization, layout, allocation) with the DiffIR operation that
  submitted them, and pushes NVTX ranges (`dif::prepare`, `dif::run`,
  `op<id> <opcode>`) for Nsight Systems correlation. Any tool can be traced
  without new flags: `DIF_TRACE_FILE=path` appends one `runtime-trace`
  document per execution; `DIF_NVTX=1` enables the ranges. A prepared
  execution that runs many times emits one document per run; only the first
  carries `execution.preparation_reported = true`, and `diftrace` counts the
  one-time preparation once. Operation timings carry a `plan` label when a
  fused backend plan executed at that operation's slot (for example the H3
  INT8 QKV projection at the QKV weight layout op), so consumers do not
  classify a projection GEMM as layout.
- The on-disk PTX cache is keyed on the generated source, the compute
  architecture, the NVRTC version and the option list, so two toolkits on one
  host (12.4 and 12.6 here) never share an entry. Frontend provenance
  sidecars carry `facts` (for example `rope_table_dtype`), creator numerics
  the frontend asserted while building.
- Two gates paid for elsewhere and adopted here. `dif_noise_tests` measures
  every shipped noise source (the torch-parity CPU generator and the H3 GPU
  generator behind `difh3noise --rng serenity`) for mean, std, sign balance
  on both halves of each Box-Muller pair, and 3-sigma tail mass, and proves
  the gate can fail by running a deliberately poisoned Box-Muller (the
  historical 2^53-divisor bug: mean +0.57, std 1.17, one half never
  negative). The grad-flow gate refuses, at prepare, any fused inference plan that
  eliminates a tensor a backward opcode reads or declares no backward at all
  (`fuse_linear_swiglu_operations`, ConvRot, W8A8, groupwise, modulation
  cache, CK attention) on a program that contains backward or optimizer
  opcodes; absorbing a BiasAdd into the cuBLASLt epilogue stays allowed
  because no backward opcode reads the unbiased intermediate and the
  epilogue tests gate its byte-identity on the MLP training graph; the DiT
  backward tests additionally require every analytic gradient to be non-zero.
- Host file-cache policy for GPU-resident weights is explicit runtime policy,
  not an accident of the loader. The historical default releases each
  resident weight's mapped range with `posix_fadvise(DONTNEED)` after the
  upload, so every fresh process rereads those bytes from storage.
  `RunOptions::resident_evict_host_pages = false` (`difh3infer
  --keep-resident-host-pages`) keeps the clean, reclaimable pages so a later
  process on the same host stages them from cache, the way a warm server
  keeps weights in RAM; outputs are byte-identical. The runtime discovers the
  process' effective cgroup-v2 memory limit and charge
  (`probe_host_cgroup_memory`) and fails closed to eviction when the limit
  cannot hold the resident bytes, printing a `RESIDENT_HOST_PAGES` line with
  the facts it used. The choice is candidate identity for `difopt`.
- `difplan show|diff|residency|explain tensor|op` reads `.difplan` files and
  the decisions the compiler recorded while planning: streamed-residency
  admission arithmetic from the residency planner, every measured candidate's
  verdict and diagnostic from the optimizer search, and the precision policy
  and target requirements bound by `difopt --plan`. Decisions are provenance,
  excluded from plan identity, and never inferred after the fact: a subject
  without a recorded decision is reported as such.

## Debugging surfaces: difbisect and difinspect provenance

Phase C adds the two debugging surfaces on the same telemetry schema.

- `difbisect pairs|manifest|program` is the generic first-divergence finder
  between native captures and an oracle fixture. Boundaries are compared in
  semantic order (an explicit `--order`, a manifest, or the producer order of
  a DiffIR program whose mapped tensors are captured by the runtime), each
  judged against explicit bars (cosine, relative L2, norm ratio, max
  absolute, non-finite). The report names the last boundary that passed and
  the first that failed, lists the uncaptured operations between them as an
  unobserved span, and never asserts a divergence at a boundary nobody
  observed. A boundary missing on one side is reported as not observed.
- `difinspect FILE.difir --source [--provenance F] [--bundle F] [--plan F]
  [--trace F] [--op ID] [--json]` navigates creator semantic, frontend,
  DiffIR, compiler region, and selected backend implementation per
  operation. Frontends record provenance as they build (module, block,
  section, creator revision, and checkpoint weight names) into a sidecar
  `FILE.difir.provenance.json`; the Krea 2 frontend records it for every
  block and denoiser operation. Weight storage identity comes from the sealed
  bundle, compiler transforms and decisions from the plan, and the backend
  implementation from the events a runtime trace actually observed. Nothing
  is inferred from tensor names; an absent link is reported as absent.

## Bounded implementation competition and physical-format hooks

Phase D makes the optimizer's candidate space target-aware without adding a
second search. `include/dif/opt/physical_format.hpp` registers the
compiler-wide physical formats a generic `Linear` (or any uniform-float
operation) may be lowered to: `fp32`, `bf16`, `fp16`, `fp8-e4m3`,
`int8-convrot`, `int4-group`, `int5-group`, the SquareQ hooks
`squareq-w8`, `squareq-w4`, `squareq-nvfp4`, and `mxfp8-block-scaled` (the
FLUX.2 Blackwell MXFP8 Linear: FP8 tensor cores plus a linked cuBLASLt with
block-scaled matmul, 12.8 or newer, both discovered by the target probe; the
CUDA backend compiles the plan only against such a toolkit and a program that
carries `LinearFp8BlockScaled` fails closed at prepare on any other host with
the library and capability facts in the message). Each format carries the
architecture capabilities it requires and how this build can use it: a
search candidate (a DiffIR transform the optimizer measures), execution
policy (ConvRot INT8 over a prepared cache, reported but not searched), or a
hook only (identity and requirements registered, no backend implementation).
Legality is decided from the probed `TargetProfile`, never from a product
name.

- `difopt --formats LIST ...` runs the existing search with the precision and
  quantization candidates derived from the requested formats; only formats
  that are legal on the probed target and implemented as search candidates
  produce transforms. Every requested format is recorded in the plan as a
  `physical-format` decision with the capability and availability facts, so
  an agent can see that FP8 or SquareQ was excluded and why. The acceptance
  order is unchanged: verify, execute, numerical gate, memory, then timing.
- `difopt --formats-table --backend cuda [--json]` prints the legality and
  availability of every format for the probed target.
- `diftune --json [--report FILE]` reports its block-size and Linear-math
  candidates with verdicts on the shared schema.
- `difweights stats FILE.safetensors|INDEX.json [--json]` reports checkpoint
  storage statistics: counts, dtypes, bytes, ranks, repeated shape patterns,
  and the largest tensors. Tensor names appear only as examples; no model
  semantics are inferred from them.

## Validation surfaces: difquality and difregress

Phase E closes the roadmap's tool sequence.

- `difquality CANDIDATE [--reference FILE] [--json]` is the generic image,
  video, and audio gate assistant. It computes decodability and sanity
  (constant image, silent audio), and against a reference PSNR, an 8x8-window
  SSIM, SNR, and for MP4s ffmpeg-sampled frame and audio-track comparisons.
  The verdict is `PASS`, `FAIL`, or `MANUAL REVIEW REQUIRED`: a numeric
  failure or a recorded rejection fails, and `PASS` requires both numeric
  admission (or sanity when no reference exists) and a recorded human review.
  Scalar metrics never replace perceptual inspection.
- `difregress run SUITE.json --tier smoke|model NAME|full [--baseline FILE]
  [--json]` runs strict correctness checks (exit status, JSON assertions) and
  noise-aware performance checks against recorded baselines (`difregress
  record`). A check that cannot run on the host is `BLOCKED`, never `PASS`.
  `perf/regress/suite.json` holds this repository's suite; its model tiers
  wrap the difbench recipes.
- Oracle fixtures follow one manifest protocol
  (`scripts/oracle_fixture_manifest.py` writes it, `difbisect validate-oracle`
  checks it) so per-model oracle scripts stay development-only while the
  tools that consume their output stay model-neutral.

## Build and test

Requirements are CMake 3.24 or newer, a C++20 compiler, and Ninja or another
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

`DIF_CUDA_ARCHITECTURES` defaults to `native`. Set it explicitly when building
for another deployment target; for example, Blackwell RTX 50-series uses
`-DDIF_CUDA_ARCHITECTURES=120` with a toolkit that supports SM 120.

H3 media output defaults to portable `libx264`. Set `--encoder h264_nvenc`
(or `H3_MEDIA_ENCODER=h264_nvenc` in the replay script) only when the selected
FFmpeg build exposes that encoder.

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
- `difprobe` for the shared hardware `TargetProfile` and current
  `RuntimeBudget`, with stable JSON output.
- `difbench`, `diftrace`, and `difplan` for the complete prompt-to-saved-output
  wall, attributed runtime tracing, and recorded plan decisions, all with
  `--json`.
- `difbisect` for last-known-good / first-known-bad boundaries against an
  oracle fixture, and `difinspect --source` for creator-to-backend provenance
  per operation.
- `difopt --formats` / `--formats-table`, `diftune --json`, and
  `difweights stats` for target-aware physical-format competition, tuning
  reports, and checkpoint storage statistics.
- `difquality` and `difregress` for the final artifact gate and tiered
  regression, plus `difbisect validate-oracle` for oracle fixture manifests.
- `difweights`, `difcast`, `difcompare`, and `difquant` for model storage and
  numerical gates.
- `difschedule` for authenticated flow schedules.
- `difopt` and `diftune` for gated plan search and backend candidate tests.
- `diftrain` for compiled forward/backward, optimizer state, and resume.
- `diftokenize`, `difcondition`, `difh3layout`, `difh3convrot`, `difmodcache`,
  `difh3vision`, `difh3encode`, `difh3state`, `difh3infer`, `difvaedecode`,
  `difaudiodecode`, and `difh3media` for the H3 proving path.
- `difkrea2block`, `difkrea2denoise`, `difkrea2sample`, and `difkrea2vae` for
  the Krea 2 source-faithful gates and native image path.
- `difflux2block` and `difflux2sample` for FLUX.2 source-faithful block gates
  and the complete native prompt-to-PNG path.

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
