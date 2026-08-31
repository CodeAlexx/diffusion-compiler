# DiT backward opcodes + composed block training gate — 2026-08-31

## Result

DiffIR now differentiates the full DiT forward set.  Eight new opcodes
(39–46) carry flame-core's proven backward equations; every one passed a
per-opcode PyTorch-autograd fixture gate (104/104 comparisons, F32 and BF16,
CPU and CUDA), and the composed gate — a real DiT transformer block graph
(RmsNormModulate → q/k/v Linear+bias → per-head QK-norm + partial RoPE →
attention → out Linear → gated residual → RmsNormModulate → fc1 → SwiGLU →
fc2 → gated residual), stacked 2 and 4 deep, trained 100 F32 AdamW steps —
matched a PyTorch reference end-to-end on CPU and CUDA (322/322 and 642/642
comparisons inside frozen bars; identical loss endpoints on both backends).

Everything runs through the generic one-thread-per-output-element NVRTC
launch path with F32 register math and F32 serial accumulators for every
cross-element reduction; `src/runtime/cuda_executor.cpp` is untouched.
These are correctness kernels: the attention backward is the O(S²)
decomposed recompute path (admitted S ≤ 4096); cuDNN SDPA backward is
Wave 3.  This gate is not a performance claim.

## Opcodes (39–46)

| Opcode | Gradients | Semantics and flame lessons honored |
|---|---|---|
| RmsNormBackward (39) | dx [, dweight] | dx = g·w·inv − x·inv³·Σ(g·w·x)/C; dweight arity is optional — the autodiff rule emits it only when the weight is a target or produced (Linear-style frozen economy) |
| RmsNormModulateBackward (40) | dx, dscale, dshift [, dweight] | modulate ordering matches the forward exactly; dshift = g, dscale = g·n |
| SwiGluBackward (41) | packed dx [.., 2W] | d_gate = dsilu(gate)·up·g, d_up = silu(gate)·g in one kernel; GateFirst respected |
| ResidualGateBackward (42) | dbranch = g·gate, dgate = g·branch | d_res = g via direct accumulation. DiffIR's gate is FULL-SHAPE per the verifier, so d_gate is elementwise — flame's sum-over-sequence applies only to its broadcast [B,1,C] gate.  The torch references use the same full-shape gate, so the fixtures arbitrate exactly these semantics (integrator-ruled 2026-08-31) |
| LayerNormBackward (43) | dx, dweight, dbias | the saved-stats form dx = rstd·(gw − mean(gw) − x̂·mean(gw·x̂)), with mean/rstd recomputed inside the kernel from the original input in F32 (DiffIR is functional; flame's non-affine-LN cancellation escape mandates this form) |
| QkNormPartialRopeBackward (44) | dx [, dweight] | exact rotation transpose (ga = c1·g0 + s2·g1, gb = −s1·g0 + c2·g1) composed with per-head RMSNorm backward; both table conventions (T = R and T = R/2) and the pass-through tail (R < D); RotaryDim is explicit on the op, stamped by the autodiff rule with the executor's default — never shape-sniffed (flame's HiDream-O1 Q/K gradient-collapse lesson) |
| AttentionLse (45) | — (F32 [S,H] logsumexp) | cross-element accumulator ⇒ pinned F32 regardless of storage dtype |
| AttentionBackward (46) | dq, dk, dv | decomposed recompute: P = exp(scale·qk − lse) in F32; dS = P·(dP − rowsum(dO∘O))·scale; delta consumes the forward output BY DIRECT TENSOR ID (flame's saved-O identity trap → grad_norm=inf); hard-skip causal matching the forward's key_end loop; scale/causal stamped explicitly so forward and backward can never resolve different defaults |

Every verifier arm enforces uniform float storage (gradient dtype = forward
dtype, flame Option A), documents its F32 cross-element accumulators via
`check_accumulator_f32` (a non-F32 AccumulatorDType fails closed), and
fail-closes on every malformed arity/shape/dtype (negative tests in
`tests/dit_backward_tests.cpp`).

## Linear backward flatten form (primitive fix)

The forward Linear admits split trailing shapes ([S,Hd] × [Hd,Hd] →
[S,H,D] — the q/k/v projection pattern).  Its backward verifier arms only
admitted the same-rank form, an asymmetry that surfaced the moment the DiT
block builder differentiated a projection.  Fixed in the primitive (never
in the builder): LinearBackwardInput/Weight admit the same-rank broadcast
form OR the forward's flatten form; BiasBackward admits the final-dimension
width or the flattened row width; the weight-grad kernels now derive their
geometry from the unambiguous [N,K] gradient tensor.  All previously
recorded programs verify unchanged (the F32/BF16/LoRA gates re-passed).
Proven by `linear_rank3` FD (cos 1.0) + CUDA parity (max_abs 7.5e-9,
through the cuBLASLt forward plan).

## In-tree suite (ctest `dif_dit_backward_tests`, always on)

- Central finite-difference gradchecks of every autodiff rule against the
  CPU forward (F32): cos = 1.0 on every target except attention
  (softmax FD noise: worst cos 0.999988), worst rel-L2 4.8e-3
  (bars 0.9995 / 0.02).
- CPU-vs-CUDA gradient parity: F32 worst max_abs 8.9e-8 (bar 1e-5);
  BF16 worst 3.9e-3 = one BF16 ulp at gradient magnitude (bar 1.6e-2).
- Frozen-weight economy assertions (dx-only RmsNormBackward arity when the
  weight is not a target), verifier fail-closed negatives, builder
  determinism (stable fingerprints), 3-step 2-block CPU loss-decrease
  smoke, and a full 2-block one-step CUDA parity check
  (measured max_abs 1.21e-7, bar 1e-4).

## Per-opcode PyTorch fixture gate

```
flock /tmp/dc-gpu.lock -c 'bash tools/run_dit_backward_gate.sh WORKDIR'
```

`tools/export_dit_backward_fixtures.py` (torch 2.10.0+cu128, deterministic,
TF32 off) exports 11 cases × {f32, bf16}: both SwiGlu orderings,
weighted/plain modulate, LayerNorm, ResidualGate (full-shape gate),
both RoPE table conventions with partial rotation (R=6 < D=8), and
full+causal attention.  `tools/difditops.cpp` runs each backward op on
cpu/cuda (attention as the forward+lse+backward chain the autodiff rule
emits); difcompare admits every gradient.

BF16 dtype contract: BF16-valued inputs, F32 reference math, ONE round at
the store — mirroring the kernels.  Torch's own BF16 autograd (per-primitive
rounding) is NOT the reference.

Measured worst over all 104 comparisons, then frozen:

| dtype | worst max_abs | worst rel_l2 | worst cos | admitted bars |
|---|---|---|---|---|
| F32 | 4.77e-7 (rms_norm dx) | 2.69e-7 | 1.0 | cos ≥ 0.999999, rel-L2 ≤ 5e-5, max-abs ≤ 2e-6 |
| BF16 | 3.91e-3 (attention dv, one ulp) | 3.97e-3 | 0.999992 | cos ≥ 0.999, rel-L2 ≤ 2e-2, max-abs ≤ 3e-2 |

Enforce run: 104/104 PASS, 0 failures.

## Composed DiT block training gate

```
flock /tmp/dc-gpu.lock -c 'bash tools/run_dit_block_gate.sh WORKDIR 2 100'
flock /tmp/dc-gpu.lock -c 'bash tools/run_dit_block_gate.sh WORKDIR 4 100'
```

`dif::frontend::make_dit_block_training` (S=16, H=2, D=8, mlp 16, rotary 8,
16 parameters/block — norm weights, q/k/v/out/fc1/fc2 weights+biases,
qk-norm weights) differentiated wrt every parameter, driven by
per-parameter AdamWUpdate; `tools/export_dit_block_training_reference.py`
is the exactly-mirrored torch reference with the manual AdamW receipt
(decoupled decay, F32 moments); `tools/difdittrain.cpp` runs it through the
real executors.

- 2 blocks (32 params): torch loss 0.6570979357 → 0.0003048621; CPU and
  CUDA both report 0.657098 → 0.000304861 (identical endpoints across
  backends); program fingerprint `364d809d…`.  322/322 comparisons pass.
- 4 blocks (64 params): torch loss 0.6593846679 → 0.0000208449; CPU
  0.659385 → 2.08446e-05, CUDA → 2.08448e-05; fingerprint `a0a11311…`.
  642/642 comparisons pass.

Measured worst per category (union of both depths and backends) and the
frozen bars are recorded in `tools/run_dit_block_gate.sh`.  The headline
composition measurement: 100-step final-parameter drift vs torch grows
1.8e-6 (2 blocks) → 2.4e-4 (4 blocks) — flame's composition-amplifies-error
lesson, measured; the param bars are therefore set from the 4-block worst
(max-abs ≤ 1e-3, rel-L2 ≤ 5e-4, cos ≥ 0.999999), and single-block parity
must never be quoted as depth parity.

## Explicit exclusions / NOT-DONE

- cuDNN SDPA backward (Wave 3; the decomposed path is O(S²) compute and is
  a correctness reference, not a performance path).
- BF16 composed training gate (per-op BF16 is gated; the composed gate is
  F32 — BF16 at depth needs its own measured bars).
- RmsNormModulateBackward always computes dweight in the weighted arity
  (no dx-only economy variant yet); LayerNormBackward always emits all
  three gradients.
- No gradient checkpointing, loss scaling, or attention masks (mask/GQA
  backward out of scope with the forward).
- OpenCL: the new opcodes fail closed (approved ownership deviation —
  compile necessity under -Werror exhaustive switches; OpenCL never
  executes them).  The OpenCL "full op set" claim now carries this
  recorded exception.
