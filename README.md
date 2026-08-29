# Diffusion Compiler

Diffusion Compiler is an independent C++20 compiler and runtime for diffusion
inference and training. It defines a checksummed semantic IR, lowers verified
programs to multiple execution backends, manages resident and streamed model
weights, and exposes command-line tools for compilation, execution, tuning,
quantization, inference, and training.

The project has no runtime dependency on Python, PyTorch, Mojo, MAX, Rust,
ComfyUI, or another diffusion framework.

> [!IMPORTANT]
> This is an experimental compiler/runtime. The repository does not distribute
> model checkpoints, generated artifacts, private fixtures, or credentials.

## Implemented capabilities

- Checksummed binary DiffIR programs and tensor interchange.
- Structural, shape, dataflow, dtype, and operation verification.
- Portable F32/BF16/F16 CPU reference execution.
- CUDA Driver API and NVRTC lowering with PTX caching, cuBLASLt linear
  operations, and optional explicit cuDNN attention candidates.
- A pure-C backend plugin ABI with reusable resident constants.
- An OpenCL 1.2 reference plugin for the current semantic operation set.
- A backend-neutral optimization and search layer between verified DiffIR and
  backend lowering: explicit transforms, legality discovery, candidate
  generation, a trusted acceptance oracle, and replayable optimization plans.
- Lifetime-aware device memory planning and dual-stream CUDA weight prefetch.
- Actual-pass CUDA profiling for mapped checkpoint staging, H2D copies,
  per-operation GPU work, attention, and resident-weight upload.
- Signed INT4 and INT5 weight formats with explicit dequantization semantics.
- Generic rectified-flow scheduling, patching, row packing, timestep, rotary,
  attention, normalization, and projection operations.
- MiniMax-H3 proving frontends for conditioning layout, denoising, scheduling,
  video VAE decoding, and joined bounded inference.
- Functional reverse-mode autodiff, gradient accumulation, AdamW state
  transitions, and checksummed training checkpoint/resume.

## Backend status

| Backend | Role | Current status |
|---|---|---|
| CPU | Typed reference | Implements every current opcode |
| CUDA | Performance backend | NVRTC, CUDA Driver API, cuBLASLt, optional cuDNN |
| OpenCL 1.2 plugin | Portable reference backend | Tested on an NVIDIA OpenCL device; non-NVIDIA hardware is not yet validated |
| Plugin ABI v1/v2 fixtures | ABI conformance | Test backends, not accelerator implementations |

Backend-neutral IR and an OpenCL implementation do not by themselves establish
support for a second GPU vendor. Hardware support should be claimed only after
the unchanged serialized program passes numerical gates on that hardware.

## Build

Requirements:

- CMake 3.24 or newer
- A C++20 compiler
- Ninja or another CMake-supported build tool
- Optional: CUDA Toolkit for the CUDA backend
- Optional: cuDNN for the cuDNN attention candidate
- Optional: OpenCL headers, loader, and a device for the OpenCL plugin

Configure, build, and test:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Disable optional backends explicitly when needed:

```sh
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDIF_ENABLE_CUDA=OFF \
  -DDIF_ENABLE_CUDNN=OFF \
  -DDIF_ENABLE_OPENCL=OFF
```

To use a nonstandard cuDNN installation:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDIF_CUDNN_ROOT=/path/to/cudnn
```

CMake fails closed to the CUDA stub when CUDA is unavailable. A program that
explicitly selects cuDNN attention is rejected if cuDNN support was not built;
the runtime does not silently substitute a different attention algorithm.

## Command-line tools

| Tool | Purpose |
|---|---|
| `difc` | Construct, transform, verify, and fingerprint DiffIR programs |
| `difinspect` | Inspect tensors, operations, and the memory plan |
| `difrun` | Execute a program on CPU, CUDA, or a plugin backend |
| `diftune` | Evaluate CUDA candidates under numerical acceptance bars |
| `difopt` | Search DiffIR transformations under numerical and memory gates |
| `difweights` | Inspect and bind SafeTensors-backed weight bundles |
| `difquant` | Create backend-neutral INT4/INT5 candidates |
| `difcast` | Convert checked tensors between supported floating-point formats |
| `difcompare` | Compare checked tensors with explicit numerical bars |
| `difschedule` | Construct authenticated rectified-flow sigma schedules |
| `difh3layout` | Construct H3 conditioning and row-timestep layouts |
| `difh3infer` | Run joined H3 denoising, scheduling, VAE decode, and validation |
| `difvaedecode` | Run tiled and temporally assembled video VAE decoding |
| `difimage` | Validate decoded video tensors and export viewable frames |
| `diftrain` | Run or resume compiled training graphs |

Run a tool without arguments to print its complete usage. Binary `.difir`,
`.diftensor`, `.difbind`, and `.diftrain` files are canonical; CLI text is an
inspection and operations surface, not a second IR format.

## Optimization search

`difopt` searches over explicit DiffIR transformations rather than over emitter
flags. It discovers which rewrites are legal on a verified program, generates
candidate programs, verifies and executes each one, admits a candidate only after
it clears the numerical gate and the memory constraint, and only then compares
performance. The winning transformation sequence is written as a plan that
rebuilds the optimized program byte-for-byte from the base program.

```sh
build/difopt --h3-denoiser --layers 1 --hidden 256 --heads 8 --head-dim 32 \
    --ffn 512 --rotary 24 --audio-input-dim 16 --synthetic-bindings 13 \
    --objective memory --warmups 1 --iterations 2 --depth 3 --beam 2 \
    --max-candidates 48 --no-memory --blocks 64,512 --quant-groups 16 \
    --memory-budget-mib 4 --plan plan.json --journal journal.json
```

A candidate is never accepted for being faster. The acceptance order is: the
candidate DiffIR verifies, it executes without non-finite values or runtime
failure, it clears the numerical bars, it fits the memory budget, and only then
is it timed. The acceptance oracle is fixed at construction and the search
cannot relax it.

The design, the transformation vocabulary, and every measured result are in
[`docs/OPTIMIZER.md`](docs/OPTIMIZER.md), including the controls used and the
cases where no improvement is claimed.

`difopt` binds a sealed checkpoint with `--weight-bundle FILE.difbind`
(`--verify-shards` re-digests every shard). A bundle and explicit `--bind`
tensors compose; `--synthetic-bindings` is the alternative to both and never a
supplement, so a real run cannot silently fall back to invented values.
[`docs/PHASE2_READINESS.md`](docs/PHASE2_READINESS.md) records the state of the
real-checkpoint proving run.

### Pipeline profiling

`difrun --profile-pipeline` instruments the actual timed CUDA execution. It
reports semantic resident/streamed weight bytes, mapped host-staging wall time,
copy-engine H2D event time, per-operation compute events, attention time, and
the non-operation part of the top-level device timeline.

`difh3infer --profile-pipeline` adds host input/output, conditioning layout,
scheduler, unpatchify, denoiser, and VAE stage measurements.

Host staging includes page faults encountered while copying mapped checkpoint
data into pinned memory. Resident upload combines page-in and device upload.
Copy and host-work sums can overlap GPU work and must not be treated as
universally additive.

## Source layout

```text
include/dif/       Public C++ and backend ABI headers
src/               IR, optimizer, compiler, runtime, frontend, training, and weight code
backends/opencl/   OpenCL reference plugin
tools/             Command-line programs
tests/             CPU, CUDA, plugin ABI, and OpenCL tests
third_party/       Vendored dependencies and their license notices
```

Generated programs, tensors, model weights, benchmark artifacts, build trees,
and developer planning material are intentionally excluded from the public
repository.

## Current limitations

- The included H3 integration uses precomputed conditioning embeddings; it is
  not a native prompt encoder or production-resolution generation product.
- Full model checkpoints are supplied separately by the operator and are never
  embedded in DiffIR or this repository.
- Low-bit H3 candidates remain experimental and require output-level admission.
- Training coverage is a bounded F32 diffusion objective, not production-scale
  H3 mixed-precision or LoRA training.
- OpenCL has not yet been validated on non-NVIDIA hardware.
- The optimization results recorded in `docs/OPTIMIZER.md` were measured on the
  portable CPU reference backend with no CUDA device present. Planned-memory
  results are deterministic; the latency measurements there are noise-dominated
  and no speedup is claimed from them.

The test suite separates build success, backend execution, numerical parity,
and decoded-output validation. A successful compile alone is not a model
quality or production-readiness claim.

## Third-party software

Vendored third-party components retain their original copyright and license
notices under `third_party/`.
