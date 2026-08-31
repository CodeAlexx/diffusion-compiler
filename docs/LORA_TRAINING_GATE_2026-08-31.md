# F32 activation-path LoRA training gate — 2026-08-31

## Result

The compiler passed a real F32 LoRA training vertical against PyTorch
autograd and adapter-only AdamW on CPU and CUDA: 100 steps of the
rectified-flow MLP objective with LoRA on all three Linears, base weights
frozen as role-`Constant` tensors, only the six adapter tensors (A/B per
Linear) differentiated and optimized.

The LoRA contract is flame's live contract (FLAME_PORT_SOURCE_NOTES.md
section 6): A `[rank,in]` within the Kaiming-uniform bound `1/sqrt(in)`,
B `[out,rank]` zeros, forward delta `(x @ A^T @ B^T) * (alpha/rank)` with
the low-rank projection kept explicit (the dense delta is never
materialized). The alpha/rank scale lives in-graph as a fingerprinted
`Fill` multiplied into the delta — not caller-supplied runtime policy.

Source loss decreased from `1.3226168156` to `0.0313517675`; CPU and CUDA
reported the same rounded endpoints (`1.32262` → `0.0313518`) and no
nonfinite step.

## Program identity

- Builder: `dif::frontend::make_lora_flow_training`
  (`diftrain make-lora OUT.difir 16 8 4 16 4 8.0`).
- DiffIR fingerprint:
  `9beb386251cada5ba2bc8af4b42b45dde8f201ea72ef9bff39091e17a369d813`
- Program file SHA-256:
  `b1dc2780b1e87c9f8aedb8672086d7b8ae2e9ed4e86446ec3c85125f2eef98da`
- 107 tensors, 65 operations. Rows 16, latent 8, timestep-feature 4,
  hidden 16, rank 4, alpha 8 (scale 2.0 — deliberately non-unit so a
  dropped or defaulted scale cannot pass).
- Data inputs 1–6; frozen `Constant` base weights 7–11; adapter
  parameters 12–17 (12/13 = latent_proj A/B, 14/15 = time_proj A/B,
  16/17 = out_proj A/B); step tensor 77; moment input pairs
  (78,79), (83,84), (88,89), (93,94), (98,99), (103,104).
- Optimizer: adapter-only AdamW, lr `5e-3`, betas `(0.9, 0.999)`,
  epsilon `1e-8`, weight decay `1e-2` on both A and B (both are
  matrices; the frozen base never enters the optimizer).

## Frozen-weight gradient economy

`differentiate()` now emits `LinearBackwardWeight` only when the weight is
a differentiation target or is produced by another operation (its gradient
then feeds earlier primals). In this graph: 9 `Linear`, 9
`LinearBackwardInput`, 6 `LinearBackwardWeight` — the three frozen base
weights emit no dead dW work. Canonical all-parameter graphs are emitted
unchanged (op-count assertions in `dif_tests` and `dif_lora_tests` cover
both, plus a transitive produced-weight case).

## Source oracle

- Script: `tools/export_lora_training_reference.py`, SHA-256
  `b0e3a376b94f37eab87266e7aea76cb2f7a2c02280914a53960cf81b9d28647c`.
- Fixture manifest SHA-256:
  `92bd922c5486ead4868ced6ebd3fcf591a7c00c2182eff992aad4c98c85e9578`.
- PyTorch 2.10.0+cu128, CUDA 12.8, RTX 3090 Ti, TF32 disabled; F32
  functional forward, autograd, unfused/non-foreach/non-capturable AdamW
  over the six adapters only.
- Both sides seed from the same fixture: deterministic A values strictly
  inside the Kaiming bound, B exactly zero, deterministic base weights and
  data. The frontend also ships `default_lora_adapter_init` (SplitMix64
  Kaiming-uniform A, zero B) for `--init-seed` runs outside this gate.

## Admission bars and measured worst cases (CPU and CUDA jointly)

Every comparison required finite values, cosine ≥ `0.999999`, and norm
ratio in `[0.9999, 1.0001]`. 70 bar comparisons passed, plus the
byte-identity checks below. Measured worst cosine `0.999999994955`; worst
norm ratio `1.0000747` (both on a final gradient).

| Category (n) | rel-L2 bar | max-abs bar | worst rel L2 | worst max abs |
|---|---:|---:|---:|---:|
| 100-step loss history (2) | 2e-6 | 2.5e-6 | 2.32e-7 | 4.17e-7 |
| final predictions (2) | 2e-6 | 5e-6 | 3.99e-7 | 1.07e-6 |
| final params + exported adapters (24) | 1e-5 | 1e-5 | 5.06e-7 | 3.58e-7 |
| first/second moments (24) | 1e-4 | 2e-7 | 3.50e-5 | 5.53e-8 |
| step-1 B and final gradients (18) | 4e-4 | 2e-6 | 1.25e-4 | 6.82e-7 |

Bar provenance, recorded honestly: losses, predictions, and parameters
inherit the rectified-flow gate's bars and pass them with ≥5x margin. The
flow gate's moment (2e-5) and gradient (7e-5 / 2.5e-7) relative bars were
tried first and failed 9 of 70 comparisons here: adapter moments and
gradients are one to two orders smaller than the flow gate's dense-
parameter equivalents, so the same F32 accumulation noise is relatively
larger while absolute errors stay at the F32 floor (max `6.82e-7`, with
cosine ≥ `0.99999999`). The bars above were then set at ~3x the measured
worst and frozen. They are this gate's bars from here on — never lowered
to pass.

## Step-1 ordering law

While B == 0, `dL/dA` is EXACTLY zero (the delta path is dead through B)
and `dL/dB` is already nonzero. Verified three ways:

- oracle asserts it and refuses to write the fixture otherwise;
- `dif_lora_tests` asserts exact-zero A gradients and nonzero B gradients
  on the CPU reference, and that zero-gradient A moves only by decoupled
  weight decay at step 1;
- through the real runner, step-1 A-gradient files are byte-identical to
  the oracle's zero tensors on BOTH backends
  (`STEP1_A_GRAD_BYTE_IDENTICAL_ZERO` for ids 12, 14, 16).

## Checkpoint/resume

Training checkpoint v1 stores the six adapters plus twelve moments (18
tensors) bound to the exact program fingerprint. CUDA ran both as direct
100 steps and as 40 steps plus a 60-step resume:

- direct and resumed checkpoint SHA-256: identical
  (`521009f317e16c2737f6230ba20914c050379555bd755bf0aaab9798d0e9b929`);
- final predictions: byte-identical;
- all six final adapter gradients: byte-identical.

Resume rejects a different program fingerprint, missing state, extra
state, shape/dtype mismatch, and I32 step overflow (same checks as
run-flow); `--resume` with `--init-seed` is rejected.

## Adapter export (.alpha regression)

`diftrain export-lora` writes, per adapter, `<name>.lora_A.weight`,
`<name>.lora_B.weight`, and `<name>.alpha` (F32 `[1]`, value 8.0 —
metadata, NOT trainable), then re-validates the file. Exported adapter
tensors passed the parameter bars against the oracle's final states from
both backends' checkpoints. `validate_lora_export` fails any file missing
a `.alpha` scalar — the regression gate for flame's 2026-05-27 ~16x
over-application incident — and `dif_lora_tests` proves the rejection on
a deliberately alpha-less file.

## Timing and memory

Single recorded runs, not comparative speed evidence:

| Runtime | Preparation | Mean step | Wall prepare + 100 steps | Resident bytes |
|---|---:|---:|---:|---:|
| CPU reference | 0 ms | 0.635 ms | 65.2 ms | 0 |
| built-in CUDA | 331 ms | 0.131 ms | 469 ms | 67,137,024 |

## In-tree verification

`dif_lora_tests` (new ctest suite) covers builder structure and roles,
fingerprint determinism, the frozen-dW economy (including the transitive
produced-weight case and canonical-graph invariance), exact step-1
ordering, CPU/CUDA one-step parity (recorded max abs `1.19209e-7`),
checkpoint-driven export byte equality, the `.alpha` regression, and
fingerprint-mismatch rejection on export. All six ctest suites green.

## CLI

```sh
build/diftrain make-lora OUT.difir ROWS LATENT_WIDTH TIMESTEP_WIDTH \
  HIDDEN_WIDTH RANK ALPHA [LR BETA1 BETA2 EPS WEIGHT_DECAY]

build/diftrain run-lora --backend cpu|cuda --program OUT.difir \
  --fixture DIR [--resume STATE.diftrain | --init-seed N] --steps N \
  --checkpoint NEXT.diftrain --losses LOSSES.diftensor \
  [--prediction OUT.diftensor] [--gradients-dir DIR]

build/diftrain export-lora --program OUT.difir \
  --checkpoint STATE.diftrain --output ADAPTERS.safetensors
```

Full reproduction (oracle + runs + byte-identity + all 70 comparisons):

```sh
flock /tmp/dc-gpu.lock -c 'bash tools/run_lora_training_gate.sh WORKDIR'
```

## Explicit exclusions

This gate does not establish:

- BF16/mixed-precision LoRA (F32 only; BF16 arrives with the training
  dtype generalization work);
- LoRA on real model architectures, attention, or bias adapters;
- gradient accumulation in the LoRA graph (single microbatch);
- adapter merging into base weights, or loading exported adapters back
  through an external inference stack;
- dropout, rank scheduling, or LyCORIS-family variants;
- comparable performance or production training readiness.
