# BF16 mixed-precision training gate — 2026-08-31

## Result

DiffIR now expresses flame-style BF16-storage / F32-accumulate mixed-precision
training, and a 100-step BF16 MLP training vertical passed against a PyTorch
BF16 reference on CPU and CUDA. Every BF16-stored output (parameters,
gradients, prediction — at step 1 and at step 100) was **bit-identical** to
the reference; the F32 loss history and F32 optimizer moments matched inside
the bars recorded below. The pre-existing F32 gates re-passed unchanged.

This is a mechanics and source-parity gate for the small MLP topology. It is
not H3 training, LoRA, loss scaling, or a performance claim, and the
bit-exactness of the BF16 tensors is a property of this workload's scale and
short reductions — flame's recorded lesson stands that composition amplifies
BF16 error, so larger models must be re-gated at their own depth.

## Semantics landed

1. **Verifier dtype relaxation** (`src/ir/verify.cpp`). MseLoss and
   MseLossBackward admit F32/BF16/F16 prediction/target (uniform) while the
   loss and upstream `grad_loss` stay F32 `[1]`. LinearBackwardInput,
   LinearBackwardWeight, BiasBackward, and SiLUBackward admit a uniform
   dtype from the same float set. The CUDA emitters compute every
   intermediate in F32 registers and round once at the typed store;
   BiasBackward, both Linear backwards, and the MSE sum keep F32
   accumulators across elements (flame burn list).
2. **AdamWUpdate dtype split** (`src/ir/verify.cpp`,
   `src/compiler/compiler.cpp`, CPU reference already dtype-generic).
   Parameter and gradient storage are verified independently in
   {F32, BF16} (the flame kernel matrix: BF16p/BF16g, BF16p/F32g, F32p/F32g,
   F32p/BF16g); both moments are F32 ALWAYS; the updated parameter keeps the
   parameter's dtype. F16 parameters/gradients fail closed (no flame kernel
   precedent). The update receipt is unchanged and matches the recorded
   PyTorch AdamW path: F32 math, biased moment updates from the raw
   gradient, bias correction, and DECOUPLED weight decay applied to the
   parameter — never folded into the gradient before the moment updates
   (flame's measured LoRA-A "unlearning" runaway).
3. **Cast backward** (`src/training/autodiff.cpp`). Gradient of
   `Cast(x, dt)` is `Cast(g, dtype(x))`. Autodiff targets may be
   F32/BF16/F16; gradient tensors carry the dtype of their forward tensor
   (flame BF16_GRAD_DECISION Option A); the loss entry check stays F32 `[1]`.
4. **AccumulatorDType (attr key 8) consumed**. The seven training opcodes
   accept the attribute only when it names F32, matching the kernels'
   unconditional F32 accumulators; any other value fails closed. Elsewhere
   the attribute keeps its existing role as a recorded reduction-identity
   marker (`src/opt/semantics.cpp`).
5. **Frontend dtype parameterization** (`include/dif/frontend/training.hpp`,
   `src/frontend/training.cpp`). `MlpTrainingConfig.compute_dtype` defaults
   to F32 and keeps the historical graph byte-for-byte — the canonical
   fingerprint below is asserted in-tree. BF16 builds BF16
   parameters/activations, a Cast boundary into the F32 MseLoss, BF16
   gradients, and F32 moments. `difc make-mlp-training` takes an optional
   trailing `f32|bf16` token; `diftrain` recovers the dtype from the
   features tensor with the fingerprint comparison still the authority.

## Program identities

- F32 (unchanged from the 2026-08-28 gate):
  `local difc make-mlp-training ... 32 8 16 4 0.01 0.9 0.999 1e-8 0.01` →
  fingerprint
  `c33733354ed3be4b5147bb7e4e2fd150400d364a3f4ad00a2c996ae1b54db95f`
  (bit-identical to the recorded candidate; also asserted by
  `test_mixed_precision_bf16_training_step`).
- BF16: same geometry/hypers plus trailing `bf16` → 45 tensors (Cast
  boundary + F32 loss + F32 moments), fingerprint
  `539704f8fbae5c6a47abeab50587085c53eefbdf868f6ce694e08f8068e9ff92`.
  Optimizer bindings (param → grad, moments in, param/moment outputs):
  3→(23; 25,26; 27,28,29), 4→(21; 30,31; 32,33,34),
  5→(19; 35,36; 37,38,39), 6→(17; 40,41; 42,43,44).
- Rectified flow (unchanged):
  `4ee07d61e635169d4b667f277d0ed0a8bbf133ba04df09a1724e731cde27674b`.

## Source oracle (BF16)

- Script: `tools/export_mlp_training_bf16.py` (this repo; PyTorch
  2.10.0+cu128, RTX 3090 Ti, deterministic algorithms, TF32 off).
- BF16 `functional.linear`/`silu` autograd with the prediction cast to F32
  for the mean-squared loss; gradients land in BF16.
- Manual AdamW loop with F32 moments and the exact AdamWUpdate receipt.
  `torch.optim.AdamW` is NOT usable as the reference here: for BF16
  parameters it allocates BF16 moments, which is exactly the state dtype
  this gate forbids.
- Targets are computed from the BF16-quantized features so both sides
  consume identical F32 targets.
- Reference trajectory: loss `0.14499077` → `0.00072364503` over 100 steps.

## Measured parity — BF16 gate, 100 steps, CPU and CUDA

Bars were set AFTER measurement, from the single-step deltas (all
bit-identical) and the observed 100-step drift, mirroring the recorded F32
gate's category structure. Every comparison also required cosine ≥ 0.999999
and zero nonfinite values.

| Category (per backend) | Measured worst (CUDA) | Measured worst (CPU) | Admitted bar |
|---|---:|---:|---|
| loss history (F32, 100) | max_abs `2.98e-8`, rel L2 `1.51e-7` | identical to CUDA | rel L2 ≤ `1e-6`, max abs ≤ `6e-8` |
| final prediction (BF16) | bit-identical (`max_abs=0`, 0 mismatches) | bit-identical | bit-identical required at this scale |
| final parameters ×4 (BF16) | bit-identical | bit-identical | bit-identical required at this scale |
| final gradients ×4 (BF16) | bit-identical | bit-identical | bit-identical required at this scale |
| first moments ×4 (F32) | rel L2 `2.74e-7`, max_abs `1.46e-10` | rel L2 `3.02e-7` | rel L2 ≤ `5e-5`, max abs ≤ `1e-8` |
| second moments ×4 (F32) | rel L2 `1.36e-5`, max_abs `1.14e-9` | rel L2 `1.33e-5` | rel L2 ≤ `5e-5`, max abs ≤ `1e-8` |
| step-1 gradients + updated params (BF16) | bit-identical | bit-identical | bit-identical |

Both backends reported the identical loss endpoints `0.144991` →
`0.000723645`, equal to the reference to F32 display precision. CPU and CUDA
BF16 runs also agree with each other bit-exactly on every exported tensor.

The bit-identical bars are honest for THIS workload (single-step deltas
measured 0; 100-step accumulation still 0) and would be the first thing to
relax — with re-measurement, never silently — if torch or cuBLAS changes
reduction order at larger scale.

## F32 regression evidence (hard bar: existing gates untouched)

Re-ran both recorded 100-step source gates at the new HEAD with the recorded
admission bars (norm-ratio windows taken from the recorded observed ranges):

- **MLP F32** (CUDA + CPU): all 36 tensor comparisons PASS; endpoints
  `0.144686` → `0.000717333` exactly as recorded; worst CUDA moment rel L2
  `4.5186284e-5` reproduces the recorded `4.51863e-5`; worst CPU gradient
  norm-ratio `0.9998735` reproduces the recorded `0.999873`.
- **Rectified flow** (CUDA + CPU): all 46 tensor comparisons PASS inside the
  recorded bars; endpoints `1.09145` → `0.202001` as recorded.
- **Full ctest suite after the change**: 5/5 passed (dif_tests,
  dif_opt_tests, dif_difopt_cli_tests, dif_plugin_tests,
  dif_opencl_plugin_tests), 0 failed, 93.02 s. The same suite was green at
  the starting commit before any change.
- In-tree one-step CUDA-vs-CPU gates at the new HEAD: F32 training
  `max_abs=1.49012e-08`, rectified flow `max_abs=2.98023e-08` (both equal to
  the recorded values), new BF16 gate `max_abs=0`.

## Checkpoint/resume (BF16)

A direct CUDA 100-step run and a 40-step run resumed for 60 steps produce
byte-identical artifacts (BF16 parameters and F32 moments ride the existing
dtype-generic Training checkpoint v1 with no format change):

- both checkpoints:
  `cc244733c3ad9edec7c525e8408368a79fb3d8b970edd39b4f4f04a7ea4e00ca`
- both final predictions:
  `95953d6f7e2f4f20d7e0c6579661b71a9b085a2842f0d87ba0b70a3862358f7a`
- all four final gradient files byte-identical.

## Repro commands

```sh
cd <worktree>; cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release; ninja -C build
# BF16 reference fixtures (CUDA required):
python3 tools/export_mlp_training_bf16.py local/gates/source-bf16-100 --steps 100
# BF16 candidate program:
build/difc make-mlp-training local/gates/programs/mlp-bf16.difir \
  32 8 16 4 0.01 0.9 0.999 1e-8 0.01 bf16
# 100-step run (repeat with --backend cpu):
build/diftrain run-mlp --backend cuda \
  --program local/gates/programs/mlp-bf16.difir \
  --features local/gates/source-bf16-100/features.diftensor \
  --targets local/gates/source-bf16-100/targets.diftensor \
  --w1 local/gates/source-bf16-100/initial-w1.diftensor \
  --b1 local/gates/source-bf16-100/initial-b1.diftensor \
  --w2 local/gates/source-bf16-100/initial-w2.diftensor \
  --b2 local/gates/source-bf16-100/initial-b2.diftensor \
  --steps 100 --checkpoint out/checkpoint.diftrain \
  --losses out/losses.diftensor --prediction out/prediction.diftensor \
  --gradients-dir out/gradients --cache-dir local/gates/cache
build/diftrain export out/checkpoint.diftrain out/state
# Compare (example: second moment of parameter 3):
build/difcompare local/gates/source-bf16-100/state-second-3.diftensor \
  out/state/tensor-26.diftensor --min-cos 0.999999 \
  --max-rel-l2 5e-5 --max-abs 1e-8
```

## Explicit exclusions

- No loss scaling, master-weight duplication, stochastic rounding, F16
  training path, parameter groups, or LoRA (the LoRA vertical is a peer's
  workstream; its frozen-dW skip touches the same autodiff.cpp).
- The reverse-mode builder still differentiates only its bounded opcode set
  (now including Cast); DiT backward opcodes are separate work.
- OpenCL executes the F32 training set only; the BF16 gate ran CPU + CUDA.
- Bit-exact BF16 parity is a measured property of this topology, not a
  general guarantee.
