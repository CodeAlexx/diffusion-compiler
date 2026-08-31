# Elementwise region fusion — 2026-08-31 (W2-F)

## What landed

`emit_cuda` can now collapse a single-consumer tree of pointwise operations
into ONE kernel, generalizing the one-off INT5 dequant→Linear chain fusion
into a principled region fuser.  Launch reduction is the point: every
interior operation of a region stops being a kernel launch and its
intermediate never touches global memory (it is excluded from the memory
plan through the existing `skipped_operations` → `excluded_internal_tensors`
path; the executor is untouched).

Fusable family: `Add, Multiply, SiLU, Clamp, Cast, BiasAdd (last-dim
broadcast), ResidualGate, SwiGlu (interior or terminal)`.

## The knob (difopt owner: candidate-identity rule)

Fusion is a **deliberate candidate property, default OFF**.  An operation
participates only when it carries `Implementation=2`, stamped by:

```
difc set-elementwise-fusion IN.difir OUT.difir on|off
```

This mirrors `set-linear-math` / `set-attention-implementation`: the stamp
is an IR attribute, so it enters the program fingerprint — difopt can flip
it as a fingerprinted transform, and recorded gates/fingerprints are
unaffected by default.  The command prints the region census
(`regions= fused_ops= eliminated_launches=`), also available as
`dif::compiler::census_elementwise_fusion`.  `off` removes the attribute,
restoring the original fingerprint byte-for-byte.

Stamping a program in which nothing fuses is proven inert: the stamped
gate-w1r program emits an IDENTICAL generated-source hash and telemetry
(below).  CPU and OpenCL executors ignore the stamp entirely.

Composition boundary: the executor-side H3 lowering plans
(`--fuse-linear-swiglu`, `--h3-w8a8-*`, `--h3-groupwise-*`,
`--h3-modulation-*`) claim operations AFTER `emit_cuda` runs and are
invisible to it.  Do not combine a stamped program with those flags when
the claimed operations intersect stamped ones (today they are H3-inference
flags and stamped programs are training/gate programs, so the sets are
disjoint in practice — but difopt must treat them as mutually exclusive
dimensions on the same operations).

## Mechanism

- **Anchor = region terminal.**  `launch_inputs` overrides only inputs and
  the executor takes outputs and launch geometry from the anchor operation,
  so the kernel is emitted at the terminal (element-per-thread geometry,
  matching the unfused terminal exactly).  Interior ops land in
  `skipped_operations`; their outputs leave the memory plan.
- **Byte-identical numerics.**  Every stage is computed in F32 registers
  and rounded to the intermediate's storage dtype in-register
  (`dif_round_*`) exactly where the unfused kernel rounds at its store.
  Expression trees are structural copies of the per-op emitters, so
  INTRA-stage FMA contraction stays identical on both sides (e.g.
  ResidualGate's `gate*branch` may fuse into its residual add on both
  sides).  CROSS-boundary contraction — a mul-topped stage (Multiply,
  SwiGlu) feeding a consumer's add — is prevented by emitting that
  top-level multiply as `__fmul_rn`, bit-identical to the lone rn-rounded
  FMUL of the unfused kernel and guaranteed never contracted.  Two
  cheaper barriers were tried and MEASURED to fail the multiply→add byte
  gate: an empty `asm("" : "+f")` value barrier (ptxas contracts
  `mul.f32`+`add.f32` at the SASS level) and an inline
  `mul.rn.f32 x,x,1.0` (ptxas algebraically eliminates the identity
  multiply, then contracts).  The byte gate in `dif_fusion_tests` is the
  permanent regression trap for this class.
- **Planner alias safety.**  The memory planner places tensors at slot
  base and best-fit-reuses non-dedicated slots by liveness, so an external
  input that DIES inside the region can have its slot handed to the
  anchor's own output (or restaged for a streamed constant) while the
  fused kernel is reading it.  Rules, checked per region at emit time:
  inputs that are dedicated (Input/Output/resident-Constant) or still
  live at the anchor are always safe; a dying input is admitted only when
  the program has no streamed constants, the region is contiguous in
  program order, and one of (a) its aligned size is smaller than the
  output's, (b) the anchor output is dedicated, or (c) every region read
  of it is at the thread's own element index with equal element count and
  dtype size.  Anything else fails safe to per-op emission.
- 16-pointer executor launch marshaling caps a region at 15 external
  inputs (enforced, tested).

## Gates (all measured on RTX 3090 Ti, this branch)

- Full ctest: 9/9 suites green before AND after (the 9th is the new
  `dif_fusion_tests`, which requires CUDA byte-identity and exact
  launch-count deltas).
- Synthetic full-family chains (Add→SiLU→BiasAdd→Multiply→Clamp→
  ResidualGate, BF16 and F32; SwiGlu both orderings; Add→Cast→SiLU dtype
  boundary; multiply→add FMA hazard): fused vs unfused BYTE-IDENTICAL,
  launches 6→1, 3→1, 2→1.
- Composed 2-block DiT training program (fwd+bwd+AdamW, 113 ops): census
  honestly **0 regions** — its pointwise ops are separated by
  Linear/Attention and gradient-accumulation Adds die at their consumers
  (the dying-input rule declines them).  Byte-identity of the stamped
  program still gated on CPU+CUDA (113 launches unchanged).
- gate-w1r streamed program (168 ops, 2.7 GiB streamed constants,
  fingerprint `0ee902b1…` reproduced from
  `difc make-h3-stack-bf16 … 512 3072 24 128 8192 64 12 256 streamed`):
  census **0 regions** (every pointwise op is GEMM/attention-separated);
  stamped run output SHA `148b694c…` IDENTICAL to unfused, identical
  source hash `55adaa16…`, identical telemetry (384 kernel launches, 288
  cuBLASLt matmuls), mean 453.5 vs 454.2 ms (noise).
- Recorded training program families (deterministic inputs, one run,
  every output byte-compared):
  - rectified-flow (`difc make-rectified-flow-training`, 2 microbatches):
    2 regions (the per-microbatch Multiply/Multiply/Add noise-mix trees),
    launches 60→56, 23/23 outputs byte-identical.
  - LoRA (`diftrain make-lora`): 2 regions, launches 56→52, 26/26
    outputs byte-identical.
- Perf demonstration, 6-op BF16 chain at 4096×4096, 3 interleaved
  sessions of 50 iterations: unfused mean 0.6484/0.6489/0.6487 ms,
  fused 0.1332/0.1330/0.1333 ms (**4.9×**, spread ≤0.0003 ms — far
  outside noise), 318→53 launches per run, outputs byte-identical.
  The win is bandwidth (5 intermediate round-trips eliminated), not
  launch overhead, at this size; on launch-bound programs the count
  reduction is the claim.

## Declined by design (and why)

- **RmsNorm/RmsNormModulate as region HEAD**: the executor derives launch
  geometry from the anchor opcode; a norm-headed region anchors at a
  pointwise terminal (element-per-thread), where reproducing the norm's
  one-block-per-row reduction costs O(columns) redundant work per thread
  — a measured-perf loss and against the mission's fail-safe rule.  Needs
  executor-side geometry override for fused entrypoints (peer-owned) —
  next tier.
- **Norm as region TERMINAL** (pointwise prologue feeding the norm's
  input): geometrically sound (row-block anchor), deferred — it requires
  replicating all three RmsNormModulate emitter variants with
  chain-expression loaders, and neither gate program has a candidate
  (norm inputs are multi-consumer block-residuals).
- **Dying-input chains without contiguity** — notably the backward
  gradient-accumulation Add chains of the DiT program (contributions die
  at non-adjacent Adds).  This is THE blocker for fusing real DiT
  training programs and is a planner problem, not an emitter problem.
- Fill (leaf constant producer), AffineLastDim: mechanically trivial
  additions, left out to keep v1 exactly the mission's op set.
- Backward-op regions (SiLUBackward etc.): next tier with the same
  machinery.

## Next tier proposal

1. **Planner-aware lifetime extension**: let `emit_cuda` hand
   `plan_memory` a set of (tensor, extended-last-use) pairs for region
   external inputs, removing the dying-input declines entirely — unlocks
   backward Add-accumulation chains in every training program.  Small,
   but crosses into `memory_plan.cpp`/executor call-site ownership.
2. **cuBLASLt epilogue absorption** (BiasAdd/SiLU/residual into Linear
   epilogues) — the flame `fused_linear3d` lesson; covers the
   GEMM-separated chains that dominate the DiT/gate programs.
3. Executor geometry override for fused entrypoints → norm-headed
   regions with true row-block structure.
4. Backward-op elementwise regions; AffineLastDim + Fill absorption.
