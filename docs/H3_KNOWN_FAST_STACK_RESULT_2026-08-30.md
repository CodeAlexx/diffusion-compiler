# MiniMax-H3 known-fast CUDA stack result (2026-08-30)

## Decision

Diffusion Compiler can now execute the accepted local MiniMax-H3 performance
stack below an unchanged DiffIR semantic boundary. At the production profile
`minimax-h3-832x480x73-24fps` (`S=9065`), the retained compiler path is:

- creator-faithful fused BF16 RMSNorm + AdaLN;
- direct resident W8A8 QKV, output, FC1/SwiGLU/FC2, and residual/gate paths;
- creator-faithful fused BF16 Q/K RMSNorm + partial RoPE;
- exact cuDNN attention or explicit admitted SM86 CK approximate attention;
- exact precomputed BF16 AdaLN modulation with byte-checked timestep input;
- resident transformed weights and reusable bounded scratch.

This is manual integration of already accepted Serenity/mojodiffusion
implementations. No autonomous optimizer search selected these primitives, no
H3 semantics moved into the optimizer, and no Sol/SLA work was performed.

The production-shape one-block source gate passes. A real 50-successive-block
compiler stack also matches the independent accepted Serenity W8A8+CK stack
bit-for-bit. This proves the compiled transformer stack, not yet the complete
prompt-to-video product pipeline: scheduler iteration, final output heads, VAE
decode, and media handoff are outside this result.

## Identity and semantic boundary

- worktree: `/home/alex/diffusion-compiler-phase2`
- branch: `phase2-real-h3`
- base commit: `7d8388e77a75c758041f0a69a116b768aa00b740`
- result revision: the commit containing this document
- GPU: NVIDIA GeForce RTX 3090 Ti, SM 8.6,
  `GPU-d45c7495-8107-47d0-dbfe-5b0d7e0b5182`
- device memory: 24,564 MiB
- driver: 580.173.02; driver-advertised CUDA: 13.0
- build toolkit: CUDA 12.4, `nvcc` 12.4.131
- pinned Diffusers revision:
  `e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1`
- accepted source tree: `/home/alex/minimax_h3_ref/diffusers-src/src`
- accepted performance tree: `/home/alex/mojodiffusion/serenitymojo`

Diffusers remains authoritative for packed QKV meaning, AdaLN, Q/K
normalization, partial RoPE, attention, SwiGLU ordering, and residual gates.
The Serenity path is used only as the implementation/performance map beneath
those semantics. The compiler backend recognizes exact source-shaped DiffIR
chains; it does not introduce a second H3 model abstraction.

## Checkpoint, fixture, and geometry

Released checkpoint index:

`/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/model.safetensors.index.json`

Checkpoint index SHA-256 / sealed index fingerprint:

`fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb`

The production fixture is the existing profile
`minimax-h3-832x480x73-24fps`: 8,580 video tokens, 244 audio tokens, and 241
text tokens, for sequence 9,065. The source capture uses deterministic analytic
video/audio/text values but creator code performs the real product packing.
It is not derived from DiffIR. Released block dimensions are hidden 5,376,
56 heads, head dimension 128, FFN 14,336, rotary width 96, and BF16 source
semantics.

Source oracle wrapper:

`tools/export_phase3_h3_oracle.py`

Accepted external exporter SHA-256:

`ceeb20f4719bbaf358d3803ee7c6da8de88f23b2e45089cd9234baec0cd28760`

DiffIR identities:

| Program | Tensors / operations | DiffIR fingerprint |
|---|---:|---|
| one block | 39 / 17 | `85c5f9a840b0e2c6e28ca70023f4b3ebce2de2310876c1b50d70b0070caa7cfc` |
| 50 blocks | 1,705 / 850 | `0a47c31ff8ad1e63deb6e8026189fc0b27f1e286aeb195c6c3ef8a91da8a0425` |

The 50-block bundle was rebound from the previously sealed released
checkpoint. The expensive check completed once:

`VERIFY_BUNDLE PASS ... shards=13 bindings=500`

Ordinary fingerprint, shape, offset, dtype, and metadata checks remained
enabled for subsequent runs; shard SHA-256 was not repeated inside timing
loops.

## Exact baseline

The unoptimized one-block cuBLASLt+cuDNN DiffIR output is bit-exact to the
independent Diffusers oracle over 48,733,440 BF16 values:

| Metric | Result |
|---|---:|
| cosine | 1 |
| relative L2 | 0 |
| norm ratio | 1 |
| max absolute | 0 |
| nonfinite | 0 |
| exact mismatches | 0 |

The controlled exact baseline session means were 133.277, 137.036, and
137.509 ms; mean 135.940667 ms. The between-session half-range was 2.116 ms.
Resident bytes were 3,615,998,464.

The baseline profile (100 hot iterations, preparation excluded) attributed:

| Family | Mean hot time |
|---|---:|
| AdaLN Linear + select | 1.221 ms |
| RMSNorm + modulation, both sites | 1.025 ms |
| QKV layout + three projections | 29.693 ms |
| Q/K norm + partial RoPE | 3.346 ms |
| exact cuDNN attention | 35.363 ms |
| output projection + residual | 9.903 ms |
| FC1 + SwiGLU + FC2 + residual | 57.439 ms |
| non-kernel device timeline | 0.026 ms/iteration |

Resident preparation was separate: 71.486 ms mapped-page prefault, 84.534 ms
H2D, and 156.026 ms total resident upload wall time in that profile.

## Integrated primitives

### Fused source-faithful glue

The retained RMSNorm+AdaLN and Q/K norm+partial-RoPE kernels are direct
semantic ports of the accepted local implementations. They preserve the BF16
normalization boundary required by the source path.

| Primitive | Source comparison |
|---|---|
| RMSNorm + AdaLN | cosine 0.999997120, rel L2 0.002400208, norm 0.999986284, max abs 0.03125, nonfinite 0 |
| Q norm + partial RoPE | cosine 0.999996899, rel L2 0.002492800, norm 1.000110997, max abs 0.0625, nonfinite 0 |

### Direct W8A8 projections and epilogues

The backend consumes the existing transformed Serenity row-scale caches
directly. It preserves cache-native QKV `[Q|K|V]` and FC1 `[value|gate]`
execution beneath creator-facing DiffIR semantics, uses bounded row chunks,
and fuses the established SwiGLU and residual/gate epilogues.

Independent original-primitive comparison results:

- Q, K, and V: bit-exact over 64,977,920 values each;
- output projection + residual/gate: bit-exact over 48,733,440 values;
- full W8A8 MLP block: bit-exact over 48,733,440 values.

Controlled primitive results, each with three reversed-order sessions,
10 warmups, and 100 CUDA-event iterations:

| Path | Baseline mean | W8A8 mean | Ratio |
|---|---:|---:|---:|
| QKV projection/layout | 28.4727 ms | 13.8786 ms | 2.052x |
| output projection + residual | 9.8495 ms | 5.0295 ms | 1.958x |
| MLP + residual | 139.526 ms block | 116.015 ms block | 1.203x |

With all W8A8 projections and exact cuDNN attention, the paired full-block
result was:

| Arm | Session means | Mean | Half-range |
|---|---|---:|---:|
| exact BF16 | 133.277, 137.036, 137.509 | 135.940667 ms | 2.116 ms |
| W8A8 + cuDNN | 90.9392, 91.7498, 94.6711 | 92.453367 ms | 1.866 ms |

The controlled improvement is 43.4873 ms and 1.47037x, well outside the
observed spread. Resident memory falls by 840,822,016 bytes, from
3,615,998,464 to 2,775,176,448 bytes.

W8A8+cuDNN creator parity passes: cosine 0.999940300, relative L2
0.011337235, norm ratio 0.996917667, max absolute 512, nonfinite 0.

### Exact cuDNN and admitted CK approximate attention

Exact attention remains cuDNN. The optional CK route loads the existing raw
C/C++/CUDA launcher DSO directly, checks ABI version and target SM, and fails
closed unless its target equals the active GPU. No Mojo or Python runtime is
present in compiler execution.

DSO:

`/home/alex/mojodiffusion/output/lib/ck/sm86/libserenity_ck_attention.so`

Compiler CK output is bit-exact to the original Serenity CK wrapper over
64,977,920 BF16 values. Against the independent exact source attention:
cosine 0.999989199, relative L2 0.004673706, norm ratio 1.000480540,
max absolute 0.203125, nonfinite 0.

Isolated controlled attention:

| Arm | Session means | Mean | Half-range |
|---|---|---:|---:|
| cuDNN exact | 33.2849, 33.6089, 33.9611 | 33.6183 ms | 0.3381 ms |
| CK approximate | 15.5580, 15.5189, 15.4036 | 15.4935 ms | 0.0772 ms |

The measured ratio is 2.16983x. In the complete W8A8 block, exact cuDNN
measured 92.350967 ms and CK measured 76.904133 ms across the paired
three-session A/B, ratio 1.200858x. The delta 15.4468 ms exceeds both
between-session half-ranges (0.9596 and 0.7602 ms).

The complete W8A8+CK block passes the unchanged creator gate: cosine
0.999939526, relative L2 0.011395111, norm ratio 0.996955633, max absolute
512, nonfinite 0.

### Exact AdaLN modulation cache and residency

The accepted Serenity cache computes timestep SiLU in F32, rounds once to
BF16, runs the source BF16 AdaLN GEMM, and stores the exact per-block BF16
modulation. Diffusion Compiler binds that result beneath
`Linear -> H3AdaLNSelect` only when the activated timestep input matches the
captured DiffTensor byte-for-byte at prepare and every run.

Block 0 cache output is bit-exact to the creator capture over 193,536 values.
The complete compiler block remains bit-exact to the original Serenity W8A8
block when modulation caching is enabled.

This is a preparation/memory win, not a hot-latency claim:

| Arm | Hot mean | Resident bytes | Mean preparation |
|---|---:|---:|---:|
| W8A8, live AdaLN Linear | 91.355567 ms | 2,775,176,448 | 360.768 ms |
| W8A8, exact modulation cache | 91.410800 ms | 2,187,649,280 | 187.257 ms |

The 0.0552 ms hot difference is inside noise. The retained route saves
587,527,168 resident bytes and about 173.5 ms of preparation in this paired
run. A complete one-block W8A8+CK+modulation-cache smoke allocated
2,383,349,504 resident bytes.

### Groupwise INT8 quality route

The compiler also exposes the established QKV16/output64/FC1-32/FC2-64
groupwise cache. It ports Serenity's exact I8 times F16-scale to BF16 decode,
uses one 308,281,344-byte scratch buffer across projections, and restores the
creator-facing FC1 `[gate|value]` order before the unchanged SwiGLU operation.

It passes both independent gates:

- versus creator: cosine 0.999994613, relative L2 0.003295776, norm ratio
  0.999697923, max absolute 256, nonfinite 0;
- versus original Serenity groupwise block: cosine 0.999989993, relative L2
  0.004483899, norm ratio 0.999686712, max absolute 256, nonfinite 0.

The compiler/original result is not bit-identical because their BF16 GEMM
schedules differ; the strict zero-error attempt is retained as a failed gate.
Controlled compiler session means were 133.810, 135.360, and 137.576 ms,
mean 135.582 ms. This does not beat dense BF16 outside noise and is much slower
than W8A8, so it remains an explicit higher-quality/memory route, not the
speed default. Resident bytes were 2,969,136,640.

## Retained production block profile

The instrumented short W8A8+CK profile measured 69.5405 ms total (not the
controlled latency estimator). Its kernel-family attribution was:

| Family | Mean |
|---|---:|
| direct W8A8 QKV | 12.6805 ms |
| fused Q/K norm + partial RoPE | 2.7092 ms |
| CK attention | 18.8030 ms |
| W8A8 output projection + residual | 4.8234 ms |
| W8A8 MLP + residual | 28.1177 ms |
| fused norms and remaining glue | about 2.35 ms |
| non-kernel device timeline | 0.019 ms/iteration |

The final bottleneck is still the W8A8 MLP, followed by approximate attention
and W8A8 QKV. There is no evidence here that another dense BF16 GEMM schedule
or another exact attention rewrite should displace the retained paths.

## Fifty successive real blocks

The compiler ran blocks 0 through 49 with each block's released checkpoint
weights, the exact source modulation cache, direct W8A8 projections, fused
norms, and the admitted SM86 CK attention implementation.

Independent accepted Serenity source-stack output versus compiler:

| Metric | Result |
|---|---:|
| cosine | 1 |
| relative L2 | 0 |
| norm ratio | 1 |
| max absolute | 0 |
| nonfinite | 0 |
| exact mismatches | 0 / 48,733,440 |

Controlled compiler timing used two warmups, five measured CUDA-event
iterations, and three prepared sessions:

| Session | Mean | Min | Max |
|---:|---:|---:|---:|
| 0 | 3822.39 ms | 3792.52 ms | 3850.85 ms |
| 1 | 3902.60 ms | 3881.34 ms | 3921.67 ms |
| 2 | 3959.38 ms | 3945.59 ms | 3968.38 ms |

Aggregate session mean was 3894.79 ms, or 77.896 ms per block. The
between-session range was 136.99 ms (half-range 68.495 ms, 1.76% of the mean).
Prepared resident memory was 21,552,909,568 bytes, leaving 2,550,398,976 bytes
free in the controlled run. Warm-cache preparation was 2,870.31 ms; the first
cold/full upload observed separately was 9,575.25 ms.

The accepted Serenity probe reported 4,725.675 ms wall time for one hot
50-block pass. That source measurement is a single synchronized wall-timer
sample, not a reversed-order CUDA-event study, so no controlled compiler versus
Serenity speed ratio is claimed from it.

## Negative results retained

- Exact cuBLASLt remains the best dense BF16 FC1/FC2 implementation on this
  RTX 3090 Ti; tested CUTLASS schedules did not win.
- The first exact FC1-to-SwiGLU WMMA fusion was 4.879x slower and rejected.
- Groupwise INT8 improves the quality/memory tradeoff but not hot latency.
- Exact modulation caching changes preparation and residency, not hot latency.
- Source-packed dense BF16 QKV is not the production winner; the retained win
  is direct W8A8 packed-QKV handling.
- CK is explicitly approximate. It is never reported as exact attention.

## Verification

- C++/CUDA build: pass with `-Werror`
- complete CTest: 5/5 pass, 0 failures, 94.05 seconds
- `git diff --check`: pass
- one-block exact source baseline: bit-exact
- all direct W8A8 original-primitive gates: bit-exact
- CK original-wrapper gate: bit-exact
- groupwise creator and original-wrapper quality gates: pass
- 50-block source-stack replay: bit-exact
- no acceptance threshold was weakened

## Reproduction

Build and run all tests:

```bash
cd /home/alex/diffusion-compiler-phase2
cmake -S . -B build-phase2 -G Ninja \
  -DDIF_CUTLASS_ROOT=/home/alex/pytorch/third_party/cutlass
cmake --build build-phase2 -j2
ctest --test-dir build-phase2 --output-on-failure
```

Common paths and one-block inputs:

```bash
root=artifacts/phase3-h3-production
checkpoint=/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA
w8=$checkpoint/serenity_runtime_cache_v1/resident_w8a8_row_blocks_1.safetensors
ck=/home/alex/mojodiffusion/output/lib/ck/sm86/libserenity_ck_attention.so
common=(
  --backend cuda --program "$root/programs/block1.difir"
  --weight-bundle "$root/bundles/block1.difbind"
  --input "1=$root/source-oracle/input-1-hidden.diftensor"
  --input "2=$root/source-oracle/input-2-temb-silu.diftensor"
  --input "3=$root/source-oracle/input-3-adaln-indices.diftensor"
  --input "4=$root/source-oracle/input-4-cos.diftensor"
  --input "5=$root/source-oracle/input-5-sin.diftensor"
  --warmups 10 --iterations 100 --session-runs 3
  --cache-dir "$root/cache" --map-inputs
)
```

Run the exact baseline:

```bash
build-phase2/difrun "${common[@]}" \
  --output "39=$root/runs/replay-exact.diftensor"
build-phase2/difcompare \
  "$root/source-oracle/reference-depth-1.diftensor" \
  "$root/runs/replay-exact.diftensor" \
  --min-cos 0.999 --max-rel-l2 0.02 \
  --min-norm-ratio 0.98 --max-norm-ratio 1.02
```

Run the complete known-fast one-block stack:

```bash
build-phase2/difrun "${common[@]}" \
  --output "39=$root/runs/replay-known-stack.diftensor" \
  --h3-w8a8-cache "$w8" --h3-w8a8-layer 0 \
  --h3-ck-attention-dso "$ck" \
  --h3-modulation-cache \
    "$root/runs/serenity-modcache-s9065-capture.safetensors" \
  --h3-modulation-input "$root/source-oracle/input-2-temb-silu.diftensor" \
  --h3-modulation-layer 0
build-phase2/difcompare \
  "$root/source-oracle/reference-depth-1.diftensor" \
  "$root/runs/replay-known-stack.diftensor" \
  --min-cos 0.999 --max-rel-l2 0.02 \
  --min-norm-ratio 0.98 --max-norm-ratio 1.02
```

Run the explicit groupwise quality route by replacing the W8A8 and CK options
with:

```bash
--h3-groupwise-cache \
  "$checkpoint/serenity_runtime_cache_v1/resident_groupwise_q16_o64_fc132_fc264_blocks_48.safetensors" \
--h3-groupwise-layer 0
```

Recreate and verify the 50-block program/bundle once:

```bash
build-phase2/difc make-h3-transformer-bf16 \
  "$root/programs/replay-block50.difir" \
  9065 5376 56 128 14336 96 50 2 2688 256 streamed split cudnn
build-phase2/difweights rebind-program \
  /home/alex/diffusion-fixtures/data/minimax-h3-transformer-50-splitqkv-streamed-cudnn.difbind \
  "$root/programs/replay-block50.difir" \
  "$root/bundles/replay-block50.difbind"
build-phase2/difweights verify-bundle \
  "$root/bundles/replay-block50.difbind" \
  "$root/programs/replay-block50.difir"
```

Run the 50-block compiler stack by using output ID 1705, the 50-block program
and bundle, and:

```bash
--h3-w8a8-cache \
  "$checkpoint/serenity_runtime_cache_v1/resident_w8a8_row_blocks_50.safetensors" \
--h3-w8a8-layer 0 --h3-ck-attention-dso "$ck" \
--h3-modulation-cache \
  "$root/runs/serenity-modcache-s9065-capture.safetensors" \
--h3-modulation-input "$root/source-oracle/input-2-temb-silu.diftensor" \
--h3-modulation-layer 0 --warmups 2 --iterations 5 --session-runs 3
```

## Next boundary

The known-fast transformer toolbox is ready for compiler deployment work.
Before claiming prompt-to-video compilation, the next gate must integrate and
measure the real denoise scheduler loop, final H3 output projection/unpacking,
and VAE/media handoff under the same creator oracle. A decoded visual product
artifact remains a separate acceptance gate. Autonomous optimization should
resume only after those operational boundaries are strong and reproducible.
