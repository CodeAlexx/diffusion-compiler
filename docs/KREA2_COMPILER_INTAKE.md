# Krea 2 compiler intake and gap map

Status: **K2-A started; real-dimension time-conditioning DiffIR scaffold
verified. Full denoiser and image generation are not yet admitted.**

This document is the source-of-truth intake for model family #2. Krea 2 must
enter through a frontend and execute through the existing verifier, planner,
runtime, and NVIDIA backend. Nothing here authorizes a second executor, a
Python production worker, or a PyTorch/libtorch production dependency.

## Semantic authority

The primary oracle is Krea AI's official release, pinned to commit
[`db3984fbc6e13b34c0064990fc2d95ac64d00058`](https://github.com/krea-ai/krea-2/tree/db3984fbc6e13b34c0064990fc2d95ac64d00058):

- [`inference.py`](https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/inference.py)
  defines the released 6144-wide configuration and BF16 load boundary.
- [`mmdit.py`](https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/mmdit.py)
  defines MMDiT, modulation, GQA, RoPE, text fusion, and final-head semantics.
- [`encoder.py`](https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/encoder.py)
  defines the Qwen3-VL-4B prompt template, selected hidden layers, padding, and
  output mask.
- [`sampling.py`](https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/sampling.py)
  defines noise, patch packing, the shifted flow schedule, CFG, Euler update,
  and latent unpacking.
- [`autoencoder.py`](https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/autoencoder.py)
  pins the Qwen-Image VAE and its latent mean/std transform.

Historical local implementations are useful secondary evidence, not semantic
authority over the official release:

- `/home/alex/EriTrainer/trainer/crates/eridiffusion-core/src/models/krea2.rs`
- `/home/alex/EriTrainer/trainer/crates/eridiffusion-cli/src/bin/train_krea2raw.rs`
- `/home/alex/mojodiffusion/serenitymojo/models/dit/krea2_dit.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/krea2/krea2_block.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/krea2/krea2_stack.mojo`
- `/home/alex/mojodiffusion/serenitymojo/sampling/krea2_sampler.mojo`
- `/home/alex/mojodiffusion/serenitymojo/models/text_encoder/krea2_qwen3vl_4b.mojo`
- `/home/alex/EriDiffusion/inference-flame/src/models/krea2/PORT_SPEC.md`

Those ports predate the current creator release in important places. In
particular, the official code always builds the boolean outer-product mask,
pads the combined sequence to a multiple of 256, uses an encoded empty prompt
for CFG, and stores the initial noise/Euler state in BF16. These official
semantics supersede older assumptions about a mask-free batch-1 path, zeroed
unconditional conditioning, or F32 latent storage.

## Local checkpoint and config inventory

The inspected Raw checkpoint is:

```text
snapshot: /home/alex/.cache/huggingface/hub/models--krea--Krea-2-Raw/
          snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494/raw.safetensors
resolved: /home/alex/.cache/huggingface/hub/models--krea--Krea-2-Raw/
          blobs/f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7
file bytes: 26,283,332,608
safetensors header bytes: 43,720
tensors: 430
dtypes: 256 BF16, 174 F32
tensor payload bytes: 26,283,288,880
parameters: 12,820,073,036
block indices: 0..27
```

The resolved blob name is the local Hugging Face content-addressed identity;
this intake did not reread 26 GB merely to print another whole-file digest.
The local `raw.safetensors.fp8cache.safetensors` is an existing approximate
cache and is **not** part of the source-faithful baseline.

Checkpoint census:

| Prefix | Tensors | Parameters | Payload bytes |
|---|---:|---:|---:|
| `blocks` | 364 | 12,156,476,416 | 24,315,719,680 |
| `first` | 2 | 399,360 | 1,597,440 |
| `last` | 4 | 411,712 | 1,646,848 |
| `tmlp` | 4 | 39,333,888 | 157,335,552 |
| `tproj` | 2 | 226,529,280 | 906,117,120 |
| `txtfusion` | 49 | 343,430,156 | 686,903,344 |
| `txtmlp` | 5 | 53,492,224 | 213,968,896 |

The checkpoint mixes BF16 transformer projections with F32 top-level time,
text, and final-head tensors. The creator loads the state dict and then calls
`.to(device, dtype=torch.bfloat16)`, so the accepted native binder must perform
the same F32-to-BF16 storage conversion. It must not bind F32 payload bytes to
a BF16 descriptor or silently execute those weights in a different precision.

Conditioner config:

```text
snapshot: Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17
hidden=2560 layers=36 heads=32 kv_heads=8 head_dim=128
intermediate=9728 vocab=151936 rms_eps=1e-6 rope_theta=5,000,000
selected hidden states=(2,5,8,11,14,17,20,23,26,29,32,35)
final conditioner shape at batch 1=[1,512,12,2560], plus bool [1,512] mask
```

VAE config:

```text
snapshot: Qwen/Qwen-Image@75e0b4be04f60ec59a75f475837eced720f823b6
class=AutoencoderKLQwenImage base_dim=96 dim_mult=[1,2,4,4]
latent_channels=16 spatial compression=8
latent mean/std: 16 creator-config values, applied before decode
```

## Released architecture and default real geometry

| Field | Released value |
|---|---:|
| model | single-stream MMDiT |
| transformer width | 6144 |
| main blocks | 28 |
| query heads / KV heads | 48 / 12 |
| head dimension | 128 |
| MLP width | 16,384 |
| latent channels | 16 |
| patch | 2x2 |
| timestep input | 256 |
| text input | 12 taps x 2560 |
| text-fusion attention heads | 20 |
| layerwise text-fusion blocks | 2 |
| sequence-refiner text-fusion blocks | 2 |
| main RoPE axes | [32,48,48], theta=1000 |
| final head | RMSNorm + modulation + Linear 6144->64 |

At the creator default 1024x1024, VAE compression 8 gives a 128x128 latent.
Patch 2 produces a 64x64 token grid: 4096 image tokens. The conditioner emits
512 text tokens, so the combined sequence is 4608 and already satisfies the
mandatory 256-token alignment. A `[1,4608,6144]` BF16 residual alone is
56,623,104 bytes. The bool outer-product mask has 21,233,664 elements. The
26.28 GB checkpoint makes streamed weights a functional requirement on a 24 GB
GPU, not an optional performance experiment.

The current official README recommends Raw at 52 steps with CFG 3.5 and Turbo
at 8 steps, CFG 0, fixed `mu=1.15`. The CLI's Python defaults remain 28/4.5;
the production acceptance recipe must choose explicitly and record which it
uses rather than conflating defaults with the current recommendation.

## Source-faithful forward semantics

1. Qwen3-VL tokenization applies the exact system/user/assistant template,
   pads to the creator length, appends the assistant suffix, runs all 36 text
   layers, selects 12 residual states, and removes the first 34 positions.
2. Two TextFusion blocks attend across the 12 selected layer taps. A learned
   12-to-1 projection collapses the tap axis. Two more TextFusion blocks refine
   the 512-token sequence under its padding mask.
3. BF16 latent noise is packed into 2x2 patches. Text precedes image tokens.
   Position rows are `[0,0,0]` for text and `[0,y,x]` for image.
4. The combined sequence, positions, and validity mask are padded together to
   a multiple of 256. Attention receives the boolean query-and-key outer
   product; padding is not merely a key mask and is not discarded at batch 1.
5. Each main block applies shared six-way modulation, RMSNorm, GQA with Q/K
   norm, interleaved three-axis RoPE, sigmoid output gating, residual update,
   then the modulated SwiGLU branch and second residual update.
6. The final head uses the unprojected time vector for a two-way modulation,
   projects 6144 to 64, and slices away text/padding before latent unpack.
7. Raw CFG encodes an actual empty negative prompt. Velocity is
   `cond + guidance * (cond - uncond)` exactly as released.
8. The resolution-derived `mu` time-shifts a 1-to-0 float grid. Euler is
   `img = img + (t_next - t_current) * velocity`; this sign/order differs from
   the existing H3-specific flow update and must remain explicit.
9. Unpacked BF16 latents are denormalized with the Qwen-Image 16-channel
   mean/std, decoded, clamped to [-1,1], mapped to [0,255], and cast to bytes.

## Required capability classification

The primary category is the component's status *now*. A new generic opcode
must subsequently pass runtime and NVIDIA-lowering gates before it can move to
`ALREADY_SUPPORTED`.

| Component | Classification | Evidence / required action |
|---|---|---|
| BF16/F32/I32 tensors, views owned by one runtime | **ALREADY_SUPPORTED** | Existing Tensor/runtime; no parallel tensor type added. |
| SafeTensors mmap and named binding | **ALREADY_SUPPORTED** | Existing weight runtime reads the Raw container. |
| F32 checkpoint tensor -> BF16 source storage conversion | **GENERIC MISSING RUNTIME IMPLEMENTATION** | Needed by many mixed-checkpoint models; binder must create faithful converted storage. |
| 26 GB streamed checkpoint, prefetch, slot reuse | **SUPPORTED BUT NEEDS REAL-DIM GATE** | Existing shared streaming path; must be measured with Krea access order. |
| 6144-wide / 16384-wide cuBLASLt Linear | **SUPPORTED BUT NEEDS REAL-DIM GATE** | Generic Linear exists; exact Raw shapes have not run. |
| Add, Multiply, SiLU, Cast, Bias Linear | **ALREADY_SUPPORTED** | Generic DiffIR + CPU + NVIDIA paths exist. |
| tanh GELU | **ALREADY_SUPPORTED** | Added generically as `Gelu` + explicit `Approximation=Tanh`; creator parity below. |
| tanh GELU on OpenCL reference backend | **BLOCKED / UNKNOWN** | Explicitly fails closed; OpenCL is outside the NVIDIA-first K2-A admission and needs a separate future parity gate. |
| GELU backward | **TRAINING-BACKWARD MISSING** | Autodiff fails closed until the generic derivative is added and gated. |
| RMSNorm | **SUPPORTED BUT NEEDS REAL-DIM GATE** | Kernel exists; Krea scale-delta (`1+scale`) loading and 6144/128 widths need fixtures. |
| Qwen3-VL transformer primitives | **SUPPORTED BUT NEEDS REAL-DIM GATE** | H3's generic Qwen frontend is reusable after configuration/tap/mask generalization. |
| Krea prompt template, suffix, 34-token removal | **CONDITIONER-SPECIFIC** | Belongs in the Krea conditioner frontend. |
| Twelve selected Qwen residual taps | **CONDITIONER-SPECIFIC** | Expose selected existing residuals as outputs; do not create a Krea executor. |
| Fixed 512-token conditioner padding mask | **CONDITIONER-SPECIFIC** | Exact tokenizer/mask construction and fixtures required. |
| Padding-aware causal Qwen attention | **GENERIC MISSING DIFFIR OP** | Existing Attention has no explicit mask input. |
| TextFusion layer-axis reshape/permute/squeeze | **GENERIC MISSING DIFFIR OP** | Add generic view/layout semantics; no Krea-named primitive. |
| TextFusion 12-to-1 projection | **SUPPORTED BUT NEEDS REAL-DIM GATE** | Generic Linear after correct generic layout. |
| 2D patchify/unpatchify, batch preserved | **GENERIC MISSING DIFFIR OP** | Generalize Patchify3D or add generic 2D semantics; do not hide rank changes in frontend memory. |
| Text/image concat, sequence pad, output slice | **GENERIC MISSING DIFFIR OP** | Add concat/pad/slice or equivalent verified generic layout ops. |
| Last-dimension/scalar broadcasting | **GENERIC MISSING DIFFIR OP** | Required by modulation and `[B,H]` gates over `[B,L,H]`. |
| Sigmoid | **GENERIC MISSING DIFFIR OP** | Required by attention output gating. |
| Unmasked rank-3 GQA Attention | **ALREADY_SUPPORTED** | `KvHeads` and exact/cuDNN backends exist. |
| Batched boolean-masked GQA SDPA | **GENERIC MISSING DIFFIR OP** | Canonical op must carry mask semantics; CPU/runtime and cuDNN lowering follow. |
| Main [32,48,48] interleaved three-axis RoPE | **GENERIC MISSING DIFFIR OP** | Existing H3/Qwen rotate-half op is not source-equivalent. |
| Krea block ordering and six-way modulation selection | **KREA2 FRONTEND SEMANTIC** | Frontend owns ordering/weight names, using generic broadcast/norm/attention ops. |
| 28-block full graph | **BLOCKED / UNKNOWN** | Block cannot be represented faithfully until masked attention, layout, broadcast, and RoPE land. |
| Resolution-derived `mu` policy and Raw/Turbo recipes | **KREA2 FRONTEND SEMANTIC** | Scheduler configuration belongs to the frontend. |
| Shifted 1->0 schedule core | **SUPPORTED BUT NEEDS REAL-DIM GATE** | Algebra matches the generic shifted-grid helper when `shift=exp(mu)`; creator rounding fixture still required. |
| Krea Euler `x + (next-current)*v` | **GENERIC MISSING DIFFIR OP** | Existing `FlowEulerStep` pins H3 data-ward semantics/sign and must not be repurposed. |
| Encoded-empty-prompt CFG and exact combine formula | **KREA2 FRONTEND SEMANTIC** | No zero-conditioning shortcut. |
| Qwen-Image latent mean/std | **VAE/IMAGE-PATH MISSING** | Generic affine/broadcast decode boundary plus exact config constants. |
| Conv2d/Conv3d, GroupNorm, VAE up/downsample | **VAE/IMAGE-PATH MISSING** | Add generic DiffIR/runtime/NVIDIA primitives, source fixture by source fixture. |
| PNG byte conversion/writer | **SUPPORTED BUT NEEDS REAL-DIM GATE** | Shared image handoff exists; Qwen-Image output gate still required. |
| LoRA injection into Krea Linear projections | **TRAINING-BACKWARD MISSING** | Shared LoRA architecture exists; full Krea forward/backward primitives do not. |
| Full block backward, masked-attention backward, GELU/Sigmoid backward | **TRAINING-BACKWARD MISSING** | Must use the same DiffIR/autodiff/runtime after inference parity. |
| FlowEdit instruction-edit path found in Serenity | **BLOCKED / UNKNOWN** | Explicitly deferred until T2I is green; it must later reuse this same runtime. |

## K2-A implementation admitted in this branch

Files:

- `include/dif/frontend/krea2.hpp`
- `src/frontend/krea2.cpp`
- `include/dif/ir/ir.hpp`, `src/ir/verify.cpp`, `src/ir/codec.cpp`
- `src/runtime/cpu_executor.cpp`, `src/compiler/compiler.cpp`
- `tests/dif_tests.cpp`

`Krea2Config` pins the released checkpoint dimensions rather than exposing
mutable architecture knobs. `inspect_krea2_architecture()` proves default
1024 geometry. `make_krea2_time_conditioning()` builds this real-dimension,
verified source slice:

```text
BF16 t [1]
  -> Cast F32
  -> creator temb [1,256] (period=1e4, factor=1e3, cos then sin)
  -> Cast BF16
  -> Linear 256->6144 + bias
  -> GELU(tanh)
  -> Linear 6144->6144 + bias        => t [1,6144]
  -> GELU(tanh)
  -> Linear 6144->36864 + bias       => tvec [1,36864]
```

This is deliberately a scaffold, not a mislabeled full block. It stops before
the first semantics that canonical DiffIR cannot yet express faithfully.

Measured scaffold identity:

```text
source commit: db3984fbc6e13b34c0064990fc2d95ac64d00058
DiffIR fingerprint: d8e0997e9fa8f75b648fccd2eb7c556a31373a8bbdceab46fc22414bed1817af
tensors: 15
operations: 8 (3 Linear, 2 Gelu, 2 Cast, 1 SinusoidalTimestep)
checkpoint bindings: 6
streamed memory-plan estimate: 531,740,672 bytes
bounded dimensions: batch=1 only; all feature/projection widths are real
```

The six bound names and shapes are exact:

```text
tmlp.0.weight [6144,256]       tmlp.0.bias [6144]
tmlp.2.weight [6144,6144]      tmlp.2.bias [6144]
tproj.1.weight [36864,6144]    tproj.1.bias [36864]
```

## Generic GELU creator parity

The official source uses `nn.GELU(approximate="tanh")`. Canonical DiffIR now
requires `Approximation=1 (Tanh)` explicitly and rejects a missing or unknown
approximation. It is not eligible for silent optimizer rewriting.

Fixture runtime: PyTorch 2.10.0+cu128, official formula, fixed values from -8
through 8. The committed test compares declared-storage payloads, not just
float printouts.

| Storage | Backend | cosine | relative L2 | max abs | norm ratio | nonfinite | bit mismatches |
|---|---|---:|---:|---:|---:|---:|---:|
| F32 | native CPU | 1 | 0 | 0 | 1 | 0 | 0/13 |
| BF16 | native CPU | 1 | 0 | 0 | 1 | 0 | 0/13 |
| F16 | native CPU | 1 | 0 | 0 | 1 | 0 | 0/13 |
| F32 | NVIDIA NVRTC, RTX 3090 Ti | 1 | 0 | 0 | 1 | 0 | 0/13 |
| BF16 | NVIDIA NVRTC, RTX 3090 Ti | 1 | 0 | 0 | 1 | 0 | 0/13 |
| F16 | NVIDIA NVRTC, RTX 3090 Ti | 1 | 0 | 0 | 1 | 0 | 0/13 |

The CUDA gate ran only after Claude's full H3 run released
`/tmp/dc-gpu.lock`; no competing GPU job was submitted. This local CMake
configuration did not discover cuDNN, which is immaterial to the elementwise
GELU gate but remains mandatory for the later real-dimension masked-attention
gate.

## Exact reproduction

CPU build and gate:

```bash
cd /home/alex/dc-krea2
cmake -S . -B build-krea2-cpu \
  -DDIF_ENABLE_CUDA=OFF -DDIF_ENABLE_OPENCL=OFF -DDIF_BUILD_TESTS=ON
cmake --build build-krea2-cpu -j2
./build-krea2-cpu/dif_tests | tee /tmp/dc-krea2-dif-tests-cpu.log
```

NVIDIA build and non-optimizer gates:

```bash
cmake -S . -B build-krea2-cuda \
  -DDIF_ENABLE_CUDA=ON -DDIF_ENABLE_CUDNN=ON \
  -DDIF_ENABLE_OPENCL=OFF -DDIF_BUILD_TESTS=ON
cmake --build build-krea2-cuda -j2
flock /tmp/dc-gpu.lock -c './build-krea2-cuda/dif_tests'
flock /tmp/dc-gpu.lock -c \
  "ctest --test-dir build-krea2-cuda --output-on-failure \
   -E 'dif_opt_tests|dif_difopt_cli_tests' -j1"
```

Both CPU and CUDA configurations passed all eight non-optimizer CTest targets.
The two optimizer/search targets were excluded deliberately because this phase
explicitly forbids optimizer-search work; no search result is part of K2-A.

Header-only checkpoint inspection (development oracle; no tensor payload is
loaded and this is not a production dependency):

```bash
python3 - <<'PY'
import collections, json, math, struct
p = "/home/alex/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494/raw.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
t = {k: v for k, v in h.items() if k != "__metadata__"}
print(n, len(t), collections.Counter(v["dtype"] for v in t.values()))
print(sum(math.prod(v["shape"]) for v in t.values()))
PY
```

## Ordered next gates

1. Generalize the now-landed native Qwen frontend through configuration and
   selected-output taps without changing its accepted H3 behavior or copying
   its executor.
2. Add generic layout primitives (reshape/view, permute, concat, pad, slice)
   with codec/verifier/CPU/NVIDIA tests.
3. Extend generic Attention with explicit boolean mask and batch semantics;
   gate CPU, exact CUDA, and cuDNN against official real-dimension fixtures.
4. Add generic three-axis interleaved RoPE and broadcast/Sigmoid semantics.
5. Build and bind one real Raw block. Gate inputs after modulation, Q/K norm,
   RoPE, attention, sigmoid gate, MLP, and both residual boundaries.
6. Extend to several blocks, then 28, preserving BF16 boundaries and streamed
   weight order. Measure load, host RAM, H2D, VRAM, launch count, and hot time.
7. Configure native Qwen3-VL-4B for the exact template, mask, and 12 taps; gate
   token IDs, every selected hidden state, TextFusion, and final context.
8. Add Krea schedule/Euler semantics and matched Raw CFG with encoded empty
   conditioning.
9. Port Qwen-Image VAE primitives generically, then produce and inspect a real
   prompt-conditioned PNG.
10. Only after T2I is green, map FlowEdit. Then add missing backwards and prove
    a Krea-compatible LoRA step through the same DiffIR/runtime.

The first full product gate remains: literal prompt -> native conditioner ->
verified DiffIR -> shared native runtime/NVIDIA backend -> native VAE -> PNG,
with no torch import, Python worker, libtorch linkage, or alternate executor in
that accepted path.
