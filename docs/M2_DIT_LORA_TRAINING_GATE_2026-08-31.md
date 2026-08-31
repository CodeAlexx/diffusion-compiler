# M2: BF16 DiT-block LoRA training gate — 2026-08-31

## Claim

Forward DiffIR → F32 loss → DiffIR-generated backward → adapter-only AdamW
→ checkpoint → byte-identical resume, with activation-path LoRA on a
diffusion-transformer block, in mixed precision (BF16 block storage, F32
adapters/loss/moments), on CPU and CUDA, with no PyTorch at runtime.
PyTorch appears only as the offline reference oracle.

`dif::frontend::make_dit_lora_training`: the proven DiT block topology
(RmsNormModulate → q/k/v Linear → QK-norm+partial-RoPE → Attention → out
Linear → ResidualGate → RmsNormModulate → fc1 → SwiGLU → fc2 →
ResidualGate) with LoRA on all six Linears per block. Base weights, biases,
and norm weights are frozen BF16 role-`Constant` tensors; adapters A
[rank,in] / B [out,rank] are F32 parameters entering through autograd-aware
`Cast` boundaries (gradients land in F32); the delta is scaled by a
fingerprinted in-graph `Fill(alpha/rank)`; the BF16 prediction feeds
`MseLoss` directly to the F32 loss; AdamW is F32 with F32 moments. The
frozen-dW economy holds structurally: 18 `Linear` / 12 `LinearBackwardWeight`
per block (adapter paths only), and no operation or optimizer input touches
a frozen tensor (asserted in ctest AND at runtime — see base-bits below).

Gate geometry: S=16, H=2, D=8, mlp 16, rotary 8, rank 4, alpha 8
(scale 2.0 — deliberately non-unit so a dropped or defaulted scale cannot
pass), lr 5e-3, wd 1e-2 on both adapters, 100 steps, depths 2 and 4 blocks.

- 2 blocks (24 adapter params): fingerprint `64c96ad1…`;
  torch loss `0.6636669040 → 0.0476561226`; CPU `0.663667 → 0.0458218`;
  CUDA `0.663618 → 0.0465112`. 248/248 gate comparisons pass.
- 4 blocks (48 adapter params): fingerprint `40663a34…`;
  torch `0.6635129452 → 0.0007636552`; CPU `→ 5.245e-4`; CUDA `→ 7.933e-4`.
  488/488 gate comparisons pass.

## Source oracle

`tools/export_dit_lora_training_reference.py` (torch 2.10.0+cu128, CPU,
deterministic, TF32 off). It follows the repository's recorded BF16
contract — BF16-valued tensors, F32 reference math, ONE round per stored
tensor — by placing a fake-quant `q()` boundary exactly at every DiffIR
operation output (torch's own BF16 autograd, which rounds at torch-op
granularity, is NOT the reference). Autograd through the cast pairs rounds
gradients at the same stored-tensor boundaries the DiffIR backward kernels
use. Adapters are F32 leaves behind one `q()` (the graph's Cast); AdamW is
the manual F32 receipt (decoupled decay, F32 moments, decay never folded
into the gradient). The BF16 cos/sin tables are the same precision floor
flame's BF16-RoPE audit recorded — mirrored on both sides.

## The BF16-at-depth measurement (the point of this gate)

Measured FIRST, bars frozen AFTER — union of both depths and backends:

| Category | worst rel-L2 | worst max-abs | worst cos | frozen bar |
|---|---:|---:|---:|---|
| loss history (F32, 100) | 6.60e-3 | 9.41e-3 | 0.999978 | rel ≤ 2e-2, abs ≤ 2.5e-2, cos ≥ 0.99995 |
| step-1 dL/dB (F32 ×36) | 1.26e-2 | 3.25e-5 | 0.999924 | rel ≤ 4e-2, abs ≤ 1e-4, cos ≥ 0.9995 |
| step-1 dL/dA (F32 ×36) | — | — | — | byte-identical ZERO required |
| final prediction (BF16) | 1.52e-1 | 3.83e-1 | 0.98854 | rel ≤ 4e-1, abs ≤ 1.0, cos ≥ 0.98 |
| final params (F32 ×144) | 2.37e-1 | 1.73e-1 | 0.97271 | rel ≤ 5e-1, abs ≤ 5e-1, cos ≥ 0.95 |
| final grads (×144) | 3.64 | 1.58e-1 | −0.981 | abs ≤ 5e-1 only; direction UNGATED |
| moment1 / moment2 (×144 ea) | 2.96 / 0.72 | 1.11e-2 / 5.52e-5 | −0.47 / 0.876 | abs ≤ 5e-2 / 5e-4; direction UNGATED |

Why the end-state bars are wide and the direction of converged gradients is
deliberately ungated:

1. **Single-step semantics are tight.** Step-1 dL/dA is byte-identical zero
   (72/72 files across depths and backends — the flame ordering law: B==0
   kills the delta path exactly), and step-1 dL/dB matches torch at
   rel-L2 ≤ 1.26e-2 with cos ≥ 0.999924 — BF16-ulp scale, consistent with
   the per-opcode BF16 fixture gate.
2. **Trajectories separate through BF16 rounding order, and the spread is
   three-way symmetric.** Our own CPU vs our own CUDA — identical semantics,
   different reduction order — end 100 steps with final-param rel-L2 0.074
   (2 blocks) / 0.210 (4 blocks) and prediction rel-L2 0.141 / 0.050;
   torch-vs-ours is the same order (params 0.065 / 0.237). No side is the
   outlier, so the divergence is intrinsic to BF16 at this depth, not a
   semantic error in one implementation.
3. **Drift grows with steps** (2 blocks, CPU): final-param rel-L2 8.6e-3 at
   step 10 → 6.5e-2 at step 100; the loss history stays inside 8.3e-5
   (10 steps) → 6.6e-3 (100). flame's composition-amplifies-error lesson,
   now measured for BF16 training: the F32 composed gate drifted 2.4e-4 at
   4 blocks; BF16 drifts ~1000x that.
4. **Converged gradients are noise.** At final loss ~5e-4 (4 blocks) the
   gradient direction is rounding-noise-dominated (cos spans ±1); an
   absolute-scale bar plus finiteness is the only defensible admission.
   The semantic gradient gate lives at step 1 and in the per-opcode
   fixtures — never quote step-1 parity as trajectory parity, and never
   quote these trajectory bars as single-step accuracy.

Both backends converge: 2 blocks 0.6637 → 0.0458/0.0465 (torch 0.0477);
4 blocks 0.6635 → 5.2e-4/7.9e-4 (torch 7.6e-4). The gate script asserts
final ≤ 0.2× initial per backend.

## Exactness gates (bit-level, both depths)

- **Resume**: CUDA 40+60 steps vs direct 100 — checkpoint SHA-256
  byte-identical (2 blocks: `73d73d4741c544ca9f0f88ca656ae8366295aeb6f7acf3666fe916aa7749aaea`),
  final predictions byte-identical, 24/24 final params byte-identical.
- **Base bits**: difdittrain snapshots every frozen tensor before step 1
  and fails unless byte-identical after the last step
  (`BASE_BITS_UNCHANGED tensors=32/64` on every run); ctest additionally
  proves no operation output and no optimizer input references a frozen id.
- **Step-1 dL/dA**: byte-identical zero through the real runner on both
  backends.
- **Export**: `--export-adapters` writes `<block>.<site>.lora_A.weight`,
  `.lora_B.weight`, `.alpha` (F32 [1], value 8.0) and re-validates via
  `validate_lora_export` (the flame missing-.alpha regression); exported
  tensors equal checkpoint state (ctest) and pass the parameter bars vs
  torch (gate spot-check).

## In-tree verification (ctest `dif_dit_lora_tests`, always on)

Builder structure and dtypes (frozen BF16 Constants; F32 adapters; 24 Cast
ops = 12 forward + 12 gradient; 18/18/12 Linear/LBI/LBW economy; F32
variant emits zero Casts), fingerprint determinism, frozen-immutability,
exact-zero step-1 A-grads with nonzero B-grads and decay-only A movement,
CPU/CUDA one-step BF16 parity (measured max-abs 3.91e-3 (one BF16 ulp) vs the recorded
1.6e-2 BF16 bar), export + fingerprint-mismatch rejection. Full suite:
9/9 ctest green at HEAD.

## Reproduction

```sh
flock /tmp/dc-gpu.lock -c 'bash tools/run_dit_lora_gate.sh WORKDIR 2 100'
flock /tmp/dc-gpu.lock -c 'bash tools/run_dit_lora_gate.sh WORKDIR 4 100'
```

Each run regenerates the torch fixture, trains fresh on CPU and CUDA with
checkpoint+export, replays the resume split, and enforces the frozen bars
(`--measure` re-opens them for re-measurement only).

## Explicitly NOT claimed

- Real H3 (or any real-model) geometry, weights, or data — S=16, H=2, D=8
  is a semantics gate, not a model.
- cuDNN attention backward (the decomposed O(S²) recompute path only) and
  attention masks / GQA (excluded with the forward).
- Tight end-state BF16 parity at depth — measured impossible under
  reduction-order freedom; the bars above are scale bars, not direction
  bars. Loss-trajectory parity is the trained-outcome gate.
- BF16 adapters or BF16 moments (adapters/moments are F32 by contract),
  loss scaling, gradient clipping, checkpointing/recompute, dropout.
- Bias or norm-weight adapters (LoRA is on the six Linears only).
- Performance (generic NVRTC kernels; CUDA wall ~2.2 s / 100 steps at
  2 blocks is not a speed claim).
