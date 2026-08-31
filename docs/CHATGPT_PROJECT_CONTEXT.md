# Diffusion Compiler: project context for ChatGPT

This is the short, authoritative orientation document to read before discussing
the purpose, current capabilities, or roadmap of Diffusion Compiler. It records
the state of `flame-runtime-integration` on 2026-08-31. For changing implementation
status and measured evidence, follow the linked living documents rather than
assuming this snapshot is current forever.

## The goal

Build one native C++20 diffusion compiler and runtime that supports inference
and training across model families. The intended user experience is conceptually:

```text
diffusion <model> --prompt "..." --output result.mp4
```

The executable must own model loading, tokenization and conditioning, DiffIR
construction, verification, execution planning, GPU execution, sampling,
decode, LoRA, training state, and reproducibility. Model checkpoints remain
external data. CUDA, cuBLASLt, cuDNN, NVRTC, operating-system facilities, and
native codec libraries are legitimate backend dependencies; another model
framework executing part of the graph is not.

The production path must not delegate model execution to Python, PyTorch,
libtorch, Mojo, MAX, Rust FFI, ComfyUI, Serenity, or a model-specific worker.
Those systems may be development oracles while a native implementation is
being compared, but they cannot remain underneath the accepted executable.

## The precise current answer: is this only for H3?

**H3 is currently the only production-scale diffusion model using the compiler
path. The compiler/runtime itself is deliberately not H3-only.**

MiniMax-H3 is the bootstrap and first product-quality gate. It has exercised a
real 50-block video/audio denoiser, rectified-flow scheduling, streamed weights,
video VAE decoding, native BigVGAN audio decoding, and H.264/AAC media output.
The accepted natural-language artifact is preserved under
`artifacts/h3-quality-natural-language-2026-08-30/`.

BigVGAN is useful evidence of generality, but it is not a second diffusion
model. It is a separate 603-operation audio neural network (including 391
Conv1d and 127 SnakeBeta operations) executed by the same DiffIR and CUDA
runtime. That proves the runtime can execute a substantially different graph;
it does not prove broad diffusion-model coverage.

The final native H3 gap is closed. Literal prompt text now runs through the
native tokenizer and a generic-op Qwen3-VL-32B DiffIR conditioner, then the
native denoiser, video/audio decode, and mux path. The complete chain was
process-inspected in an environment where importing torch fails and its output
was visually re-gated. PyTorch is now an H3 development oracle, not a runtime
dependency. See `H3_NATIVE_PROMPT_TO_MP4_GATE_2026-08-31.md` for metrics,
resources, and exact evidence.

See:

- `FLAME_CPP_RUNTIME_PORT.md` for the current port matrix and measured gates.
- `H3_NATIVE_PROMPT_TO_MP4_GATE_2026-08-31.md` for the completed conditioner
  and literal-prompt quality gate.
- `FLAME_PORT_HANDOFF_2026-08-31.md` for exact reproduction commands and open work.
- `RUNTIME_MODULES.md` for ownership and module boundaries.

## What is already real

The existing system is one coherent compiler/runtime, not a proposed diagram:

- Checksummed, fingerprinted, backend-neutral semantic DiffIR with fail-closed
  verification.
- CPU reference execution and an NVIDIA backend using the CUDA Driver API,
  NVRTC, cuBLASLt, cuDNN, and generated CUDA kernels.
- A C backend ABI and an OpenCL reference implementation. OpenCL has only been
  tested on the local NVIDIA device and is not an AMD support claim.
- SafeTensors-backed, SHA-sealed weight bundles; mmap loading; resident and
  streamed weights; asynchronous staging; memory planning and telemetry.
- Generic linear, attention/GQA, normalization, RoPE, activation, residual,
  layout, quantization, Conv1d, and SnakeBeta execution at the currently
  recorded scope.
- Native H3 tokenizer, seeded noise, modulation cache, scheduler, video VAE,
  BigVGAN audio decode, WAV output, and media handoff.
- Functional autodiff, mixed-precision parameter handling, AdamW, gradient
  accumulation, checkpoint/resume, and LoRA training at the admitted DiT-block
  scope.
- Compiler-visible transformation candidates, numerical acceptance gates,
  memory constraints, replayable plans, and measurement provenance.

The M1 migration gate reproduced the accepted H3 video latent, audio rows, and
audio latent byte-for-byte through the consolidated runtime. The M2 training
gate proved 100-step BF16 LoRA training and byte-identical resume on composed
two- and four-block DiT programs without PyTorch at runtime. These are bounded
proofs, not claims that every model or full H3 training is complete.

## The architecture boundary

The compiler owns semantic meaning and policy:

- DiffIR and verification
- graph transformations and provenance
- precision, quantization, fusion, recompute, and memory policy
- execution-plan generation and fingerprints

The runtime owns mechanism:

- device buffers, allocations, streams, events, and synchronization
- vendor-library calls and generated/custom kernel launches
- scratch/workspace storage, weight staging, and telemetry
- execution of the compiler's explicit plan

One DiffIR operation does not have to equal one GPU launch. A verified semantic
region may lower to one or a few fused/generated launches. Hardware assumptions
belong in backend lowering and capability checks, not canonical DiffIR.

H3 packing, modality layout, AdaLN selection, schedule details, and final heads
belong in the H3 frontend. Linear, attention, normalization, convolution,
allocators, streams, tensor storage, and training mechanics belong in the
shared compiler/runtime.

## What "support all models" must mean

It cannot mean that every present and future checkpoint works without a model
description. Different architectures still require frontends, configuration
parsers, weight-name mappings, conditioning rules, schedulers, and quality
fixtures.

The required invariant is:

> A new model family may add a frontend, manifest, weight mapping, scheduler,
> and genuinely general semantic operations. It must not add another executor
> or outsource graph execution to another framework.

The long-term product needs a versioned model-package/manifest format describing
architecture, tensor names, dtypes, preprocessing, conditioning, scheduler,
latent geometry, decoders, and supported tasks. A frontend lowers that model
description into verified DiffIR; the same execution planner and runtime then
run it.

Krea 2 is now model family #2. Its official checkpoint/config intake and full
gap map are in `KREA2_COMPILER_INTAKE.md`. The first admitted frontend slice
uses real 6144/36864 projection dimensions and the shared runtime; it is not a
claim of full Krea inference. Generic tanh GELU has landed and is creator-exact
on CPU/CUDA. Full Krea blocks still require layout/broadcast, masked batched
attention, and interleaved three-axis RoPE; Qwen-Image VAE decode remains open.

## Remaining capabilities for broad inference

The next model families will require generic functionality that H3 did not:

1. **Layout and algebra**
   - Transpose/Permute and explicit view/reshape semantics
   - general Concat, Split, Slice, gather/scatter, and deinterleave
   - broadcast semantics and dynamic-but-bounded image/video shapes

2. **Transformer variants**
   - GEGLU and additional modulation/gating forms (tanh GELU is now present)
   - batched, masked, cross-, joint-, and windowed-attention contracts
   - model-specific RoPE policies expressed through general attributes or ops

3. **Convolutional image/video paths**
   - Conv2d and Conv3d
   - GroupNorm and remaining normalization forms
   - upsample, downsample, interpolation, padding, and resampling
   - tiled/spatial/temporal VAE encode as well as decode

4. **Conditioners**
   - generalize the native Qwen3-VL conditioner for Krea's 4B/twelve-tap mask
   - reusable BPE and SentencePiece-class tokenizer support
   - CLIP, T5/UMT5, Gemma/Llama-family, and vision/reference encoders as needed
   - prompt weighting, negative conditioning, pooled outputs, masks, and
     multi-encoder composition

5. **Tasks and sampling**
   - general Euler, Euler ancestral, DDIM, DPM++ and flow-matching families
   - text-to-image/video, image-to-image/video, video-to-video, inpainting,
     editing, reference conditioning, and ControlNet-style inputs
   - native input preprocessing, latent encoding, masks, preview, and media IO

6. **Large-model execution**
   - production-grade residency/offload selection and deeper asynchronous
     prefetch
   - bounded host page-cache behavior for checkpoints larger than RAM
   - reusable workspaces, CUDA graphs where legal, and optional multi-GPU
     planning when a supported model cannot fit a tested single-device policy

Quantized paths must remain explicit approximate candidates. They are admitted
per model only after trajectory and decoded-quality gates; their existence is
not a blanket quality claim.

## Remaining capabilities for broad training

Training must use the same DiffIR and runtime, not a second training engine.
Broad production training still needs:

- backward coverage for every newly admitted forward operation
- cuDNN SDPA backward and performant convolution backward
- loss scaling and complete F16/BF16 mixed-precision policy
- global gradient norm/clipping, EMA wiring, and optimizer variants
- activation checkpointing/recompute and activation offload
- a device-resident step/optimizer loop
- generic LoRA placement for linear and convolutional families
- full-finetune state handling and sharded checkpoint save/load
- native dataset, caption, image/video/audio preprocessing and batching
- distributed/multi-GPU execution only after it is testable on owned hardware

The current native LoRA/AdamW gates prove the architecture was not designed as
inference-only. They do not yet constitute production training for H3 or every
model family.

## Recommended proof sequence

1. Preserve the completed native H3 literal-prompt visual/audio gate while
   generalizing shared primitives; do not regress its accepted artifact.
2. Provide one unified `diffusion` CLI/model-package path instead of requiring
   operators to compose many development executables manually.
3. Finish Krea 2 through the same runtime. This is the active second DiT-family
   proof and exercises selected-layer text fusion, masked GQA, generic layout,
   image VAE operations, and native image output.
4. Add FLUX as the next family to prove those shared additions generalize
   beyond H3/Krea rather than becoming Krea-shaped runtime behavior.
5. Add a convolution-heavy family such as SDXL to prove the runtime is not only
   transformer-shaped.
6. Add WAN or LTX to prove general temporal/video attention, video VAE, and
   conditioning behavior.
7. Repeat the same sequence for training: small deterministic gate, real
   dimensions, checkpoint/resume, and then a visually evaluated LoRA artifact.

For each supported model, admission means: checkpoint fingerprint, exact prompt
and seed, source/reference fixtures, per-boundary numerical metrics, resource
measurements, deterministic replay where required, decoded artifacts, and human
visual/audio inspection. A build, finite tensor, or high cosine alone is not
model support.

## Packaging definition of the native executable

On Linux, the product will normally be an ELF executable plus model files and
the installed NVIDIA driver/runtime libraries. "One executable" means no Python
environment, framework server, per-model worker, or hidden subprocess executing
the neural network. It does not require embedding checkpoints or reimplementing
CUDA/cuDNN.

`ffmpeg` is currently an explicitly retained native mux boundary. It may remain
a declared packaging dependency or later be replaced by linked codec/container
libraries; it is not a model-runtime dependency.

## Rules for future ChatGPT answers

- Do not say the compiler supports all models today.
- H3 is admitted as fully native prompt-to-MP4 only in the precise sense and
  measured scope documented by `H3_NATIVE_PROMPT_TO_MP4_GATE_2026-08-31.md`;
  do not broaden that evidence to untested tasks or geometries.
- Do not describe Diffusion Compiler as an H3 runtime; H3 is its first proving
  frontend and currently its only production-scale diffusion consumer.
- Distinguish generic runtime capability from admitted model support.
- Distinguish tensor parity, decoded-quality acceptance, and performance.
- Never weaken a gate to admit a port or optimization.
- Treat PyTorch, Serenity, Mojo, and Flame as development/reference oracles, not
  permissible permanent execution dependencies.
- Verify live repository state before quoting branch status, timings, memory,
  hashes, or remaining work.
