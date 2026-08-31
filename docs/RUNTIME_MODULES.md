# Runtime module map (Flame-style)

One paragraph per module: what it is, where it lives, its entry points, and
its current status. Companion to REPOSITORY_MAP.md (tables) and
FLAME_CPP_RUNTIME_PORT.md (port matrix). Status vocabulary: EXISTING |
PORTING | SCAFFOLD | PROVEN | PROVEN-REAL-DIMS | PERFORMANCE-GATED | BLOCKED.
Last updated: 2026-08-31 (Waves 1-3 merged).

## IR and verification

**DiffIR core** (`include/dif/ir/ir.hpp`, `src/ir/`) — the semantic program
representation: 38 dense opcodes, 5 dtypes (F32/BF16/F16/I8/I32), 7 tensor
roles, typed attributes, SSA-like single-writer dataflow. No strings in
canonical form; binary codec with trailing SHA-256 and deterministic
fingerprints (`codec.hpp`). Training ops are functional (new tensor ids, no
mutation/aliasing). PROVEN at recorded gate scope. Wave-1 change in flight:
training-op dtype pins relaxed to the float set (w1-training).

**Verifier** (`src/ir/verify.cpp`) — fail-closed structural/shape/dtype/
dataflow validation run at every layer boundary (codec, autodiff, memory
planning, emission, search). Per-opcode rules; unknown anything rejects.
PROVEN (focused negative coverage). The single most load-bearing file in the
stack; every port lands behind it.

## Compiler

**CUDA emission** (`src/compiler/compiler.cpp`) — one whole-program NVRTC TU;
one `__global__` per op with shapes baked as literals; dtype handled by
`dif_load/dif_store/dif_round` macro families (F32 compute, round at store —
this IS the flame BF16-storage/F32-accumulate contract). Linear and
Attention-impl-2 are not emitted (dispatched to cuBLASLt/cuDNN). Chain
subsumption exists via `GeneratedCuda.launch_inputs`/`skipped_operations`
(used by the INT5→QKV→Linear fusion) — the designated hook for future
region→one-launch work (Wave 2). EXISTING/PROVEN at gate scope.

**Memory planning** (`src/compiler/memory_plan.cpp`) — liveness-interval
best-fit slot assignment; dedicated slots for inputs/outputs/resident
constants; streamed-constant intervals widened by prefetch distance. No
recompute, no activation offload (Wave 2+/3 targets). EXISTING.

**Low-bit weight IR** (`src/compiler/int4.cpp`) — backend-neutral INT4/INT5
packing, group scales, outlier correction, column companding. EXISTING;
full-stack uniform INT5 rejected on evidence (see gates).

**Optimizer/search** (`src/opt/`, `tools/difopt.cpp`, `tools/diftune.cpp`) —
16 serializable transform kinds, const acceptance gates (verify → execute →
nonfinite → numerical → memory → perf), noise-widened margins, both-order
median timing, full provenance + byte-identical `.difplan` replay, tuning DB
v4 with raw samples. Rule for all runtime work: execution-policy knobs enter
candidate identity here, never ad-hoc runtime state. PROVEN at Phase-1/2 scope.

## Runtime

**Executor interface** (`include/dif/runtime/executor.hpp`,
`src/runtime/executor.cpp`) — RunOptions/RunResult, prepare/run contract,
CPU/CUDA factories. Wave-1 additions in flight: LaunchTelemetry (launch/
memcpy/host-stall counters) and streaming-policy knobs with
current-behavior defaults (w1-runtime).

**CPU executor** (`src/runtime/cpu_executor.cpp`, `scalar.cpp`) — portable
typed reference for every opcode at F32/BF16/F16 storage; the in-tree
correctness oracle for CUDA parity. PROVEN.

**CUDA executor** (`src/runtime/cuda_executor.cpp`, single 4.5k-line TU) —
Driver API contexts, compute+copy streams, NVRTC+PTX cache, per-slot device
buffers realized from the memory plan (all allocation at prepare; zero
steady-state allocs), cuBLASLt LinearPlans (pedantic strict / TF32 opt-in /
bias epilogue), deduped cuDNN SDPA forward plans, CUTLASS menu, W8A8 INT8
path with resident-prefix + reusable tail, groupwise INT8, modulation-cache
plans, depth-1 dual-stream StreamedPrefetcher with pinned double-buffer,
event telemetry + per-op profile mode. PROVEN at recorded H3 scope.
PORTING (Wave 1): staging overhaul (page-drop policy, deeper prefetch ring,
threaded staging, tail uploads on copy stream, pinned I/O), telemetry.
Known limits: no arena/pool consolidation, no CUDA graphs, forward-only
attention, heuristics not persisted across prepares.

**cuDNN attention** (`src/runtime/cudnn_attention.cpp`) — cudnn_frontend
SDPA graph, BF16/F16 with F32 compute, stats off (forward-only). Now
supports GQA natively: K/V are declared with their own head count and the
plan key carries `kv_heads`, so dense and grouped plans over the same query
geometry cannot collide. cuDNN SDPA *backward* remains open work; the
decomposed `AttentionBackward` opcode is its parity reference. EXISTING.

**Tensor I/O** (`src/runtime/tensor_io.cpp`, `include/dif/runtime/tensor.hpp`)
— owned or mmap-backed host tensors, checksummed `.diftensor` container,
page-discard helpers. EXISTING.

## Weights

**SafeTensors** (`src/weights/safetensors.cpp`) — strict parser (full
coverage check), mmap slices, bounded streaming writer. **Bundles**
(`src/weights/bundle.cpp`) — SHA-sealed `.difbind` manifests binding tensor
ids to shard/name/offset, program-fingerprint-checked. Residency is a
per-constant Streamed role bit (program-wide toggle today). PROVEN at
recorded scope. Wave-1 adjacent: native modcache builder (w1-deps) removes
the Serenity-generated-cache dependency.

## Training

**Autodiff** (`src/training/autodiff.cpp`) — functional reverse-mode builder
over the linear op list; explicit Add accumulation; fails closed on any
active op without a rule. Rules today: MseLoss, BiasAdd, Linear, SiLU, Add,
Multiply (+Fill leaf). Wave-1 in flight: Cast backward + float-dtype targets
(w1-training), frozen-weight dW skip (w1-lora). Wave-2: the DiT backward set
(RmsNorm, RmsNormModulate, SwiGlu, ResidualGate, QkNormPartialRope,
Attention) from flame's proven equations. EXISTING/PORTING.

**Training frontend** (`src/frontend/training.cpp`) — canonical MLP mechanics
+ rectified-flow velocity objective; graph-unrolled microbatch accumulation
(≤64); per-parameter AdamWUpdate bindings with per-group hyperparameters.
PROVEN via 100-step PyTorch parity on CPU/CUDA/OpenCL. Wave-1: dtype
parameterization (w1-training) and LoRA builder + export map (w1-lora).

**Checkpoint** (`src/training/checkpoint.cpp`) — versioned, program-bound,
SHA-256'd state serialization; dtype-agnostic; byte-identical resume gate.
PROVEN.

**Runner** (`tools/diftrain.cpp`) — executes the canonical graph families;
host-mediated step loop (outputs moved back to inputs per step). Wave-1:
generalized LoRA mode (w1-lora). Device-resident optimizer loop is a Wave-3
target.

**Audio VAE / BigVGAN** (`include/dif/frontend/h3_audio_vae.hpp`,
`src/frontend/h3_audio_vae.cpp`, `tools/difaudiodecode.cpp`,
`src/support/wav.cpp`) — the released DAC/BigVGAN decoder as a 603-op DiffIR
program (391 `Conv1d`, 127 `SnakeBeta`), rank-3 `[B=2,C,L]` with stereo as
batch, in-program latent denormalization, float64 weight-norm folding at
import, and a native int16 WAV writer matching the reference quantizer.
PROVEN-REAL-DIMS: reproduces the accepted `audio.wav` at 86.87 dB SNR with
a maximum one-LSB deviation, in 0.63 s. OpenCL execution of the two new
opcodes is a recorded gap (fail-closed arms only).

**Qwen3-VL text conditioner** (`include/dif/frontend/qwen3vl_conditioner.hpp`,
`src/frontend/qwen3vl_conditioner.cpp`, `tools/difcondition.cpp`) — the
MiniMax-H3 text tower as one DiffIR program built entirely from generic
opcodes: `GatherRows` embedding, `RotaryPosition` tables, then per layer
`RmsNorm` -> `Linear` q/k/v -> `QkNormPartialRope` -> causal grouped-query
`Attention` -> `Linear` o -> `Add` -> `RmsNorm` -> `Linear` gate/up ->
`SiLU` -> `Multiply` -> `Linear` down -> `Add`. No Qwen-specific opcode
exists: the model's identity is the frontend plus its checkpoint weight
names. Weights bind straight to the checkpoint's own `text_encoder` shards
and stream through the plan-slot prefetcher (551 streamed tensors, ~0.98
GiB/layer). Extraction is the RAW residual stream after 50 of 64 layers —
`model.norm` is deliberately not applied. PROVEN-REAL-DIMS: rel-L2
3.449e-03 vs the transformers oracle, inside that framework's own
eager-vs-sdpa envelope
(docs/H3_NATIVE_PROMPT_TO_MP4_GATE_2026-08-31.md).

## Frontends

**H3** (`src/frontend/h3*.cpp`) — block/stack/denoiser/conditioning-layout/
latents/VAE/media builders for MiniMax-H3; the proving model, not the
architecture. PROVEN-REAL-DIMS (accepted 175-frame natural-language video).
H3-shaped IR to generalize in Wave 2: H3AdaLNSelect, H3DeinterleaveQkv*,
FlowEulerStep (per OPTIMIZER.md: DeinterleaveGroups + ReshapeView, no new
model-specific opcodes). Missing for FLUX/WAN class: Transpose/Permute,
Concat/general Split, GELU, Conv/GroupNorm/upsample, batched/masked
Attention.

## Backends beyond CUDA

**Plugin ABI** (`include/dif/backend/abi.h`, `src/backend/plugin.cpp`) —
pure-C v2 ABI with v1 fallback; host-tensor views (zero-copy deferred by
contract). **OpenCL reference** (`backends/opencl/`) — full current op set,
real-device tested on the local NVIDIA OpenCL driver only; not a
second-vendor claim. EXISTING/PROVEN at conformance scope.

## Tools

`difc` (construct/verify), `difinspect`, `difrun` (bind/run/trace),
`diftune`, `difcast`, `difschedule`, `difh3layout`, `difweights`,
`difquant`, `difimage`, `difvaedecode`, `difh3infer` (joined H3 pipeline +
diagnostic slicing flags), `diftrain`, `difh3media` (native RGB24 + ffmpeg
mux), `difcompare`, `difopt`, `difslice`. Wave-1 adds: `difimport` (native
SafeTensors→diftensor, w1-deps), modcache builder (w1-deps), LoRA
make/run/export modes (w1-lora).

## External boundary (current honest claim)

Torch-free and Python-free at link level across every generation-path
binary; ffmpeg subprocess for mux (retained by declaration); Mojo-dependent
nowhere in the H3 path. The dependency chain is CLOSED: prompt text ->
native tokenizer -> native Qwen3-VL conditioner -> native denoise -> native
video VAE -> native BigVGAN audio -> ffmpeg mux, proven at process level in
an environment where `import torch` fails. cuDNN is vendored outside any
Python tree. PyTorch survives only as a development oracle.
PyTorch remains only in oracles (/home/alex/diffusion-fixtures) and
evaluation tooling.
