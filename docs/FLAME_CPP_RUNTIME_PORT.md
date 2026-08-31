# Flame -> C++ runtime port: inventory, matrix, and plan

Status snapshot: 2026-08-31. Branch `flame-runtime-integration` (from main
@305a321; preservation commit 6d3ec3d). Baseline: 5/5 ctest suites green on
this branch before any port work.

Mission: evolve the existing Diffusion Compiler runtime into the one native
C++ Flame-grade runtime (inference AND training) by porting flame-core's
proven semantics — kernels, backward equations, memory/streaming policies,
optimizer behavior, dtype contracts — into the existing architecture. Not a
second runtime; not a Rust FFI; not a mechanical Rust-to-C++ translation.

Status vocabulary: EXISTING | PORTING | SCAFFOLD | PROVEN | PROVEN-REAL-DIMS
| PERFORMANCE-GATED | BLOCKED.

---

## 1. What the C++ stack already is (verified 2026-08-31)

One coherent 27k-LOC C++20 compiler/runtime already exists and is the
architecture to evolve:

- **DiffIR v1**: 38 opcodes (ir.hpp:34-73), SSA-like single-writer,
  fail-closed verifier at every layer boundary, checksummed binary codec,
  fingerprints. Training ops (32-38: MseLoss..AdamWUpdate) hard-pinned F32.
- **CPU reference executor**: every opcode, F32/BF16/F16 storage.
- **CUDA runtime** (src/runtime/cuda_executor.cpp, 4471-line single TU):
  Driver API, two streams, NVRTC whole-program TU + SHA-keyed PTX cache,
  cuBLASLt (pedantic strict / TF32 opt-in, bias epilogue), optional cuDNN
  SDPA (forward-only), CUTLASS menu, W8A8 INT8 path, groupwise INT8, INT4/5
  dequant, plan-slot memory realization (one cuMemAlloc per slot, zero
  steady-state allocs), depth-1 dual-stream weight prefetcher, event
  telemetry + per-op profile mode.
- **Memory planning**: liveness-interval best-fit slot reuse +
  streamed-constant prefetch intervals. No recompute, no activation offload.
- **Weights**: mmap SafeTensors, SHA-sealed .difbind bundles, resident vs
  program-wide-streamed constants.
- **Training**: functional reverse-mode autodiff for {MseLoss, BiasAdd,
  Linear, SiLU, Add, Multiply} in F32 only; graph-unrolled microbatch
  accumulation; per-parameter AdamWUpdate ops; versioned checkpoint;
  100-step PyTorch-parity gates (MLP + rectified-flow) on CPU/CUDA/OpenCL.
- **difopt optimizer/search**: 16 transform kinds, const acceptance gates,
  noise-disciplined measurement, full provenance + byte-identical replay.
  Any new execution-policy knob must enter candidate identity — never
  ad-hoc runtime state.
- **H3 product proof**: accepted 175-frame natural-language T2VA video
  (426 s denoise vs Serenity 3031 s; lower VRAM), golden artifact at
  `artifacts/h3-quality-natural-language-2026-08-30/` (never overwrite).

## 2. Flame module -> C++ equivalent port matrix

Legend: DC = diffusion-compiler current state. Details per row in §3-§6.
(Flame-side source anchors are being finalized from the flame-core
extraction pass; rows marked ⏳ get source refs in the next revision.)

| Flame area | DC equivalent | Verdict | Gate | Perf reference |
|---|---|---|---|---|
| Tensor/storage/dtype (BF16 storage, F32 accumulate contract) | DiffIR dtypes + emitter dif_load/store macros (compiler.cpp:172-256) | EXISTING — keep; contract already matches | existing opcode tests | n/a |
| cuBLASLt GEMM + epilogues | LinearPlan (cuda_executor.cpp:282-701), bias epilogue only | EXISTING/PARTIAL — extend epilogues, persist heuristics | bit-gates per epilogue | flame fused_linear3d parity work |
| SDPA dispatch policy (cuDNN, fallback, budget tiling) | cuDNN forward plans + naive generated fallback | PARTIAL — no backward, naive fallback | cuDNN bwd gate + fallback parity | flame sdpa docs |
| Backward equations (matmul/norm/act/rope/modulate) | autodiff has 6 rules, F32 only | PORTING — the core training port | per-op torch-oracle fixtures | flame autograd arms ⏳ |
| RMSNorm/AdaLN/RoPE/SwiGLU fused fwd kernels | coarse opcodes, 1 NVRTC kernel each + 2 Serenity fast paths | EXISTING — good shape; extend fusion via skipped_operations hook | existing gates | flame fused_*.cu ⏳ |
| Device alloc pool / scratch ring / pinned staging | per-slot cuMemAlloc + ~10 feature workspaces; pinned only in prefetcher/W8A8 | PORTING — arena + scratch frame + pinned pool | byte-identical outputs + alloc telemetry | flame cuda_alloc_pool / ring_alloc ⏳ |
| Weight offload/residency (BlockOffloader lessons) | program-wide Streamed bit; depth-1 prefetch; re-stage EVERY iteration + madvise(DONTNEED) | PORTING — top measured cost (157.5 s of 206 s wall) | byte-identical outputs + streamed-stage telemetry | H3_PIPELINE_PROFILE_2026-08-29 |
| AdamW (incl. BF16 param / F32 moment split) | AdamWUpdate F32-uniform (verify.cpp:242-247) | PORTING — dtype split | 100-step torch parity re-run | existing training gates |
| 8-bit Adam | none | MISSING — later wave | new opcode + parity | flame adam8bit ⏳ |
| LoRA fwd/bwd/save | activation-path form expressible in F32 today; no runner | PORTING — LoRA vertical | grad-flow + torch parity + save/load round-trip | flame lora.rs lessons (alpha-scale export bug) ⏳ |
| Gradient checkpointing / recompute | none (RematerializeProducer only) | MISSING — wave 2+ | loss-identical recompute gate | flame gradient_checkpointing ⏳ |
| Grad clipping | none in-graph | MISSING — wave 2 | torch parity | flame grad_norm ⏳ |
| EMA | LinearBlend op is exactly the update | SCAFFOLD — frontend wiring only | state round-trip | n/a |
| Mixed-precision training boundaries | Cast op fwd-only; no Cast backward | PORTING — Cast bwd is the choke point | mixed-precision MLP gate | flame mixed_precision ⏳ |
| Quantization runtime | INT4/5 + W8A8 + groupwise already native | EXISTING — keep; W8A8 cache builder to nativize | existing gates | n/a |
| Conv/GroupNorm/upsample (conv-VAE class) | none (H3 VAE is ViT) | MISSING — generality wave (FLUX/WAN) | torch conv parity | flame conv/cudnn ⏳ |
| Launch-count telemetry | none | PORTING — cheap, wave 1 | counter vs nsys spot-check | speed-contract discipline |
| CUDA graph capture | none | MISSING — wave 3 (resident programs first) | byte-identical replay | flame cuda_graph ⏳ |

## 3. DC runtime findings that drive wave 1 (from the runtime audit)

Ranked measured/verified inefficiencies:
1. **Streamed-weight staging**: streamed constants are re-staged host-side
   and re-uploaded EVERY warmup and EVERY iteration; pages are
   `madvise(MADV_DONTNEED)`d after each copy, forcing re-page-in; prefetch
   depth is exactly 1; staging memcpy is synchronous on the submitting
   thread. Measured: 157.5 s of a 206 s H3 wall in host staging at
   0.39 GiB/s vs 2.6 s of actual H2D at 23.7 GiB/s.
2. **Run defaults** warmups=2/iterations=5 (executor.hpp:30-31) stream a
   61.7 GiB checkpoint 7x per defaulted "run".
3. **No fusion/no graphs**: 1 op = 1 launch outside hand-rolled paths;
   ~98-100 launches/block on the W8A8 fast path (~5k/step); per-op
   `find_if` plan scans on the submission path.
4. Host-blocking event syncs in staging parity reuse; W8A8 tail H2D rides
   the compute stream.
5. Per-run O(data) CPU validation (every index element + modcache memcmp).
6. cuBLASLt heuristics/plans rebuilt every prepare; nothing persisted but
   PTX text.
7. Pageable dynamic-input/output transfers; fresh output allocation per run.
8. CUTLASS plans freeze device pointers at prepare (blocks relocation).

Strong properties to preserve: all allocation at prepare (zero steady-state
allocs); fail-closed verification everywhere; candidate-identity discipline;
byte-identical replay; strict/TF32 split; classification labels
(exact vs approximate) load-bearing.

## 4. Training gap analysis (from the compiler/IR audit)

To train a real DiT block the stack needs, in order:
1. **Dtype generalization of existing training ops**: relax the F32 pins
   (verify.cpp:149-259) to the float set with F32 accumulate; split
   AdamWUpdate's enforced dtype uniformity for BF16-param/F32-moment;
   honor the already-declared-but-unconsumed AccumulatorDType attr.
2. **Cast backward** — the single choke point for any mixed-precision
   boundary.
3. **Backward opcodes for the DiT op set**: RmsNorm, RmsNormModulate,
   SwiGlu, ResidualGate, QkNormPartialRope, Attention (cuDNN SDPA backward
   — note flame's recorded trap: cuDNN SDPA bwd with a cast d_o produced
   inf grads; parity harness required before enabling), LayerNorm,
   GatherRows/IndexedUpdateRows. Flame's proven backward equations are the
   reference; PyTorch fixtures are the gate.
4. **LoRA vertical**: activation-path LoRA (y = Wx + scale*B(Ax)) is
   expressible and differentiable TODAY in F32 with zero new autodiff
   rules. Needs: frontend builder + name<->id map for export, a
   generalized diftrain runner (current one accepts only 2 fingerprinted
   program families), dead frozen-dW pruning (EliminateDeadOperations
   exists in opt/, unwired in training path), and the alpha/rank scaling
   contract done right (flame lesson: exports that omit alpha mis-scale at
   inference).
5. Device-resident optimizer loop (current loop is host-mediated per step)
   — later wave, after the semantics are proven.

## 5. External-dependency inventory and removal order (from the deps audit)

Torch-free TODAY at link level: every compiler binary in the accepted chain
(ldd-verified: CUDA + cuDNN + libstdc++/glibc only; no libtorch, no
libpython; only subprocess is ffmpeg).

Honest current claim: torch-free generation path; Python-free except a
stdlib-only one-time importer; **Mojo-dependent at three points** —
(1) tokenizer + Qwen3-VL conditioner (prompt -> embedding), (2) modulation-
cache provenance (Serenity generates; compiler only validates), (3) BigVGAN
audio decode.

Removal order (S/M/L/XL = effort):
1. S — vendor `mem_safe_runtime.sh` into compiler `scripts/` (exists at
   /home/alex/mojodiffusion/scripts/mem_safe_runtime.sh; compiler scripts/
   is empty); optionally port the stdlib importer to C++.
2. M — native modulation-cache builder (`--build-modulation-cache`):
   timestep-embed + AdaLN projections over the checkpoint's own shards;
   removes a silent Serenity-artifact dependency. Same pattern later for
   the optional W8A8 cache.
3. S — native seeded noise matching Serenity's sampler (byte-repro of
   existing evidence only).
4. L — native BigVGAN audio decode as a DiffIR program (conv + snake +
   upsample; weights `audio_vae/model.safetensors`).
5. XL — native tokenizer (BPE over checkpoint `processor/tokenizer.json`)
   + Qwen3-VL-32B hidden-layer-50 conditioner as a streamed DiffIR
   program (MRoPE collapses to 1-D for all-text prompts). Long pole.
6. Declare ffmpeg retained for mux (recommended) — external native binary,
   no torch/python.

Torch remains only in oracles (/home/alex/diffusion-fixtures) and
evaluation tooling (whisper, media metrics) — allowed as development
oracles by policy.

## 6. Open threads that must not be lost (from the open-threads audit)

- **Block-1 BF16 divergence vs Serenity** (final video-latent cosine
  0.886): tooling built at HEAD (sliced programs through_{1..50},
  `--max-evaluations`, `--first-eval-input-dir`,
  `--h3-modulation-total-layers`), next experiments identified (refiner-
  boundary compare; attention A/B via CK DSO; intra-block stages; per-eval
  accumulation curve). Parity thread, tracked separately from the runtime
  port; runtime work must not alter exact-path numerics without re-running
  the trace.
- 50-layer weight-placement search launched, not completed (logs header-only).
- difopt generalization queue (backend-selectable diftune; policy knobs
  into candidate identity; attention/precision/quant/recompute hooks).
- IR generality recommendations (OPTIMIZER.md): DeinterleaveGroups,
  ReshapeView; no new model-specific opcodes.
- Linked worktree /home/alex/diffusion-compiler-phase2 (branch
  phase2-real-h3) holds Phase 2 evidence — leave untouched.
- All rejected-experiment artifacts (int4/int5 families, failed searches)
  are preserved evidence — never clean.

## 7. Execution plan (waves)

Ownership is per-file-region to avoid collisions; agents work in
worktrees; one integrator merges; one GPU-heavy test owner at a time;
compile parallelism bounded (machine: 12 CPU, 62 GiB, one RTX 3090 Ti).

**Wave 1 (in flight):**
- W1-R "runtime-streaming": overhaul streamed-weight pipeline (drop
  DONTNEED between iterations, stage-once policy knobs, deeper prefetch,
  threaded staging, tail H2D onto copy stream) + arena allocator behind
  DeviceBuffers/Workspace + pinned I/O pool + launch-count telemetry.
  Owner file: src/runtime/cuda_executor.cpp (single owner by design).
  Gate: byte-identical outputs on recorded programs + measured staging
  delta; policy knobs surfaced through RunOptions/candidate identity.
- W1-T "training-dtype": training-op dtype generalization + Cast backward
  + AdamW BF16/F32-moment split + mixed-precision MLP gate vs PyTorch.
  Owner files: src/ir/verify.cpp (training arms), src/training/autodiff.cpp,
  compiler.cpp AdamW/backward emitters, frontend/training.cpp.
- W1-L "lora-vertical": F32 activation-path LoRA builder + generalized
  runner + adapter-only AdamW + export name map + dead-dW pruning wiring.
  Owner files: include/dif/frontend/, src/frontend/, tools/diftrain.cpp.
- W1-D "dependency-hygiene": vendor mem_safe_runtime.sh, C++ importer,
  native modulation-cache builder (difweights or new mode), native seeded
  noise. Owner files: scripts/, tools/ (new files).

**Wave 2:** backward opcodes for the DiT set (flame equations + torch
fixtures); fused elementwise region emission via skipped_operations;
cuBLASLt epilogue extension + heuristic persistence; grad clipping; EMA
wiring; IR generality ops (DeinterleaveGroups, ReshapeView, Transpose,
Concat/Split, GELU, batched/masked Attention).

**Wave 3:** cuDNN SDPA backward (parity-harness-gated); BigVGAN native
decode; conv/GroupNorm/upsample primitives (FLUX/WAN class); CUDA graph
capture for resident programs; device-resident optimizer loop; gradient
checkpointing; native tokenizer + Qwen3-VL conditioner (long pole).

**Milestone gates:** (M1) golden-artifact-matched H3 run through the
consolidated runtime, byte-identical latents; (M2) BF16 LoRA training
proof (forward -> DiffIR backward -> AdamW -> checkpoint resume) with no
torch at runtime; (M3) torch/Mojo dependency-removal ledger complete
through step 4 of §5.

## 8. Testing discipline

Nothing becomes PROVEN by compiling. Every port: reference semantics ->
C++ implementation -> deterministic test -> GPU parity -> real dims ->
performance measurement -> PROVEN. Numerical gates record cosine, relative
L2, max abs, norm ratio, nonfinite count, bit mismatches. Bars never
lowered to pass. Runtime-policy changes must keep recorded-gate outputs
byte-identical or be shipped as new fingerprinted candidates through
difopt's gate discipline.
