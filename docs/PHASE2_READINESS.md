# Phase 2 readiness: one real MiniMax-H3 block

**Historical readiness snapshot.** The environment blockers recorded below
were resolved on the local GPU machine on 2026-08-29. Phase 2A/2B and the
requested 1/3/5 recurrence gate have now run; see
[`PHASE2_H3_BLOCK_RESULT_2026-08-29.md`](PHASE2_H3_BLOCK_RESULT_2026-08-29.md)
for the sealed checkpoint identity, source parity, real-CUDA search, controlled
timing, profile attribution, clean replay, and promotion boundary. This file is
retained to preserve the pre-run requirements and should not be read as current
machine status.

Phase 2 is defined as proving the optimizer on **one real H3 transformer block**:
real released checkpoint tensors, real checkpoint layouts, real model
dimensions, real AdaLN/QKV/FC1/FC2 semantics, and a captured input/output pair
from the source-faithful H3 path. The reduced `hidden = 256` Phase-1 fixture is
explicitly **not** an acceptable evidence target.

## Blockers

Each of these was checked in the session that wrote this file, and none can be
resolved from inside the container.

### 1. No CUDA device

No `nvidia-smi`, no `/dev/nvidia*`, no driver library, and no CUDA toolkit.
CMake reports `DIF_HAS_CUDA=0` and falls closed to `cuda_executor_stub.cpp`.

Phase 2B requires minimizing **real CUDA block latency**, with CUDA events around
the actual prepared execution and real timings for at least two accepted
candidates. `ScriptedLatencyExecutor` is explicitly excluded — it exists to prove
the *selection rule* deterministically (see `OPTIMIZER.md`), and is not a
substitute for a device measurement. Phase 2B cannot begin without a GPU runner.

### 2. No checkpoint access

`MiniMaxAI/MiniMax-H3` is a real released model with a `transformer/` directory,
but this session's egress policy refuses the host:

```
connect_rejected — gateway answered 403 to CONNECT (policy denial)
host: huggingface.co:443
```

That is an organization policy denial, not a transient failure. The weights must
be supplied by allow-listing the host for the environment, or by mounting the
`transformer/` shards and their `.index.json` into the container.

### 3. No source oracle

Phase 2A requires a deterministic captured block input and the matching expected
output **from the creator's own H3 path**. That needs the reference
implementation and the weights on the same machine; PyTorch is not installed and
the weights are unreachable. If such a capture already exists from prior H3 port
work, supplying it is by far the cheapest route.

## What is already in place

### The real-dimension block builds and verifies

No dimension is reduced. Only the sequence length is bounded, and it is called
out below.

```sh
difc make-h3-block-raw-bf16 block0.difir \
    1024   `# S      bounded sequence, see "Sequence geometry"` \
    5376   `# HIDDEN` \
    56     `# HEADS` \
    128    `# HEAD_DIM` \
    14336  `# FFN` \
    96     `# ROTARY` \
    256    `# BLOCK` \
    resident
```

```text
fingerprint = 1bf06b993242264c47460cc4282e81fa5b52db265ffdab10b7dff154d0c5e65a
tensors     = 32
operations  = 13
memory      = 24 slots, planned 1 017 009 664 B (969.9 MiB),
              naive 1 134 450 176 B (1081.9 MiB), saved 117 440 512 B
constants   = 771 118 592 B (735.4 MiB) of bf16 checkpoint weights
```

Real weight shapes, exactly as the checkpoint carries them:

| Tensor | Shape | Bytes |
|---|---|---|
| `attn.qkv_proj.weight` | 21504 x 5376 | 231 211 008 |
| `attn.out_proj.weight` | 5376 x 7168 | 77 070 336 |
| `mlp.fc1.weight` | 28672 x 5376 | 308 281 344 |
| `mlp.fc2.weight` | 5376 x 14336 | 154 140 672 |
| `attn.q_norm.weight` / `attn.k_norm.weight` | 128 each | 256 each |
| `norm1.weight` / `norm2.weight` | 5376 each | 10 752 each |
| rotary cos / sin | 1024 x 96 each | 196 608 each |

The operation sequence is the real block, with AdaLN, QKV, FC1 and FC2
semantics intact:

```text
 1 rms_norm_modulate      AdaLN (norm1, scale/shift)
 2 linear                 QKV projection
 3 h3_deinterleave_qkv    packed QKV split
 4 qk_norm_partial_rope   Q
 5 qk_norm_partial_rope   K
 6 attention
 7 linear                 output projection
 8 residual_gate          AdaLN gate
 9 rms_norm_modulate      AdaLN (norm2, scale/shift)
10 linear                 FC1
11 swiglu
12 linear                 FC2
13 residual_gate          AdaLN gate
```

### Checkpoint binding reaches the optimizer

`difopt` previously accepted only individual tensor files or the synthetic
fixture, so the sealed `.difbind` path that Phase 2A is specified to use could
not reach it. `difopt` now takes `--weight-bundle FILE.difbind` and
`--verify-shards`, mirroring `difrun`, and reusing
`dif::weights::{read,verify,load}_weight_bundle` unchanged.

Composition rules, chosen so a real run cannot silently degrade into a synthetic
one:

- A sealed bundle and explicit `--bind` tensors **compose**. The bundle carries
  the checkpoint constants; `--bind` supplies captured inputs on top, and an
  explicit tensor overrides a bundle entry rather than being dropped.
- `--synthetic-bindings` is the **alternative** to both, never a supplement.
  Mixing them is a usage error.
- The bundle is verified against the program fingerprint before loading, so a
  checkpoint cannot be bound to a graph it does not describe.
- A constant that no source covers is a hard failure
  (`difopt: missing constant tensor N`), not a zero-fill.

Shard digests are hashed **once per invocation**. `load_weight_bundle` already
verifies before it maps anything, so `difopt` does not call
`verify_weight_bundle` separately; doing both would hash every shard twice.
Measured with `strace -e trace=openat` on a `--verify-shards` run: the shard is
opened 4 times before the fix and 3 after (digest, SafeTensors metadata, mmap).
No check was dropped — the program fingerprint, per-binding descriptor, shard
file size, and SafeTensors metadata comparisons all still run.

`--verify-shards` is deliberately **advisory**, not mandatory. A search may seal
and fully verify once, then skip re-hashing every shard for every candidate; a
run that omits it warns on stderr and still enforces every non-digest check.

`tests/difopt_cli_tests.cpp` (CTest target `dif_difopt_cli_tests`, CPU-only,
~0.2 s) is the gate on all of this. It builds a SafeTensors shard through the
public `SafeTensorWriter`, seals a `WeightBundle` against it, and drives the real
`difopt` binary through: bundle binding succeeds; bundle plus explicit `--bind`
succeeds; an explicit tensor overrides the same tensor from the bundle (checked
by the base candidate fingerprint moving, since it covers bound constant values);
real bindings plus `--synthetic-bindings` is a usage error, as is no binding
source at all; a bundle sealed against a different program is refused by
fingerprint; an unbound constant is refused; and `--verify-shards` refuses a
shard whose payload changed after sealing while its size did not, which the
size check alone cannot catch.

The gate was mutation-checked rather than assumed: reverting `insert_or_assign`
to `emplace`, ignoring `--verify-shards`, and loosening the exclusivity rule each
make it fail.

The composition rules were also verified end to end by hand on a small
transformer program with a synthetic SafeTensors shard carrying the real
checkpoint tensor names. The fixture weights are meaningless — the point is only
that the plumbing carries a sealed bundle into the search:

```sh
difc make-h3-transformer-bf16 xf.difir 8 32 2 16 64 8 1 2 16 64 resident packed
difweights make-h3-bundle model.safetensors.index.json xf.difir xf.difbind
difweights verify-bundle xf.difbind xf.difir          # VERIFY_BUNDLE PASS
difopt --program xf.difir --weight-bundle xf.difbind --verify-shards \
       --bind 1=in1.diftensor --bind 2=in2.diftensor --bind 3=in3.diftensor \
       --bind 4=in4.diftensor --bind 5=in5.diftensor \
       --objective memory --no-memory --blocks 64,256 --plan plan.json
```

The run reported the bundle, sealed shard count and index fingerprint, then
searched normally and produced accepted and `rejected_numerical` candidates.
`make-h3-bundle` binds the ten checkpoint tensors per block
(`adaln_proj.linear.{weight,bias}`, `attn.{qkv_proj,q_norm,k_norm,out_proj}.weight`,
`mlp.{fc1,fc2}.weight`, `norm{1,2}.weight`); the rotary cosine/sine tables are
computed, not checkpoint data, and are bound separately.

## Sequence geometry

`S = 1024` was used for the structural check above and is **not yet a decision**.
It matters and should be fixed before Phase 2A runs, because:

- it drives block latency, which is the Phase 2B objective;
- the verifier admits the generated attention path (`Implementation = 1`) only
  for `S <= 4096`, so a longer sequence forces the cuDNN candidate
  (`Implementation = 2`, requiring bf16/f16) and changes which attention
  transforms are legal;
- planned memory is dominated by activations at large `S` and by constants at
  small `S`, which changes what the memory constraint actually binds on.

Whatever is chosen must be recorded with the baseline, per Phase 2A step 6.

## Phase 2A run sequence, once unblocked

1. Build the block program (`difc make-h3-block-raw-bf16`, above) at the agreed
   sequence geometry, and record its fingerprint.
2. Seal the real checkpoint tensors:
   `difweights make-h3-bundle <index>.json block0.difir block0.difbind`,
   then `difweights verify-bundle block0.difbind block0.difir`. Record the
   bundle's program and index fingerprints.
3. Bind the captured source input and run the unoptimized program on CUDA
   (`difrun --backend cuda --weight-bundle ... --verify-shards`).
4. Compare against the captured source output under the existing strict H3
   numerical contract, using `difcompare`. **Do not loosen the bars.**
5. Only if step 4 passes, run `difopt --backend cuda --objective latency` with
   the same bundle, a device-memory budget, and enough warmups and iterations to
   establish a noise band.
6. Replay the winning plan in a clean process and confirm it rebuilds the same
   candidate fingerprint and passes the same gate.

If step 4 fails, stop and fix semantics. The optimizer must not search around a
wrong baseline.

## What is deliberately not done

- No baseline, parity, or latency number is reported here. There is no device
  and no checkpoint, so any such number would be fabricated.
- The reduced Phase-1 fixture was **not** substituted as an evidence target.
- No H3-specific transformation was added. If Phase 2 discovers a missing
  transformation, the question to ask first is whether it is a reusable
  diffusion/compiler transformation — if so it belongs in `dif::opt`
  generically, and if not it belongs in frontend semantics.
- No acceptance threshold was changed.
