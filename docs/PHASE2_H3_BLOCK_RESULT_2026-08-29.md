# Phase 2: real MiniMax-H3 block optimizer result

Date: **2026-08-29 America/Los_Angeles**

## Outcome

Outcome **A**: the real-CUDA optimizer selected an accepted, measurable
one-block speedup. The winning all-site strategy also passed independent
source comparisons after 1, 3, and 5 successive released H3 blocks. This is
not approval to start a 50-block optimization pass; the requested review gate
remains in force.

The work used local branch `phase2-real-h3`, based exactly on remote
`origin/claude/diffusion-compiler-optimizer-w54psa` commit
`22d17a22ada10719e10c902c0277eb2c508b5984`. An adjacent linked worktree at
`/home/alex/diffusion-compiler-phase2` preserved unrelated dirty work in the
original `/home/alex/diffusion-compiler` checkout.

## Fixed identities

- GPU: NVIDIA GeForce RTX 3090 Ti, UUID
  `GPU-d45c7495-8107-47d0-dbfe-5b0d7e0b5182`, SM 8.6, 24,564 MiB
- driver: 580.173.02
- CUDA toolkit: 12.4.131
- compiler backend: `cuda-nvrtc-cublaslt`
- checkpoint:
  `/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer`
- checkpoint index SHA-256:
  `fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb`
- shard 1 SHA-256:
  `0b3386565e476bfdea287e9ea9f269d036e5c649ed14bb9b4afac1dc4661bd2a`
- shard 2 SHA-256 (needed from block 2 onward):
  `9c98fd4579c9bc96d1b1bbf65c1243f9256df60bccd229570cecc61897773003`
- independent source checkout:
  `/home/alex/minimax_h3_ref/diffusers-src` at
  `e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1`
- source attention: forced Diffusers `_native_cudnn`; TF32 disabled

The five-layer bundle was sealed from the released index, then
`difweights verify-bundle` re-digested both shards and passed. The one- and
three-layer bundles were derived with `rebind-h3-bundle`, retaining all normal
program/index/shape/offset/size/SafeTensors checks without repeatedly hashing
the multi-gigabyte shards during benchmark loops.

## Geometry and source oracle

The selected packed sequence is **4**: one real video row, one real audio row,
and two real text rows, with two timestep tables. It is the smallest existing
accepted native-preprocess H3 fixture that exercises all three modalities,
real timestep/AdaLN routing, and rotary construction. It is preferable to the
older sequence-18 block gate for repeated search, and it does not reuse the
sequence-1024 structural example from the readiness note.

All model dimensions remain released dimensions: hidden 5376, 56 heads,
head dimension 128, FFN 14336, rotary width 96, timestep embedding width 2688,
and BF16 transformer/AdaLN semantics. The source fixture is
`/home/alex/diffusion-fixtures/data/minimax-h3-denoiser-native-preprocess-l1-source`.
It was produced by the pinned Diffusers frontend from raw video/audio/text,
raw timesteps, and position IDs, then captured the packed block input,
`temb_silu`, AdaLN indices, rotary tables, and exact BF16 block-0 output. The
depth-3 and depth-5 references were independently regenerated with
`export_minimax_h3_denoiser.py`; neither reference was reconstructed from
DiffIR.

## Baseline seal and source parity

- unoptimized one-block DiffIR fingerprint:
  `6273aba3a3e944eb76aaa20abf401dea89fdd2b6a2223e51f55648c582e654fe`
- sealed bundle program fingerprint: same as above
- base program-plus-binding fingerprint:
  `9fab1d35378d25731f18f69f7405a5239d6674ed08b16454bb0643d1110fe429`
- block: 0
- dtype: BF16 block inputs, constants, and output
- planned bytes: 1,523,155,712
- actual backend resident bytes: 1,590,264,576

The unoptimized CUDA result passed the existing H3 bars unchanged
(cosine >= 0.999, relative L2 <= 0.02, norm ratio in [0.98, 1.02], no
nonfinites):

| Metric | Baseline block 0 |
|---|---:|
| cosine | 0.99999954879679664 |
| relative L2 | 0.00094995145874250285 |
| max absolute | 8 |
| mean absolute | 0.093246313565898506 |
| norm ratio | 0.99999829545230723 |
| nonfinite | 0 |
| BF16 exact mismatches / 21,504 | 7,515 |

The baseline was therefore admitted before search. It is close under the
accepted H3 contract but not bit-exact; no exactness claim is made.

## Automatic real-CUDA search

The latency-objective beam search used 5 warmups, 20 CUDA-event iterations,
beam 4, depth 3, a 96-candidate ceiling, a 20 GiB planned-memory ceiling, and a
4 GiB free-device-memory pressure guard. Structural, schedule, numeric,
memory/residency, prefetch, attention implementation, precision, INT4, and
INT5 candidates remained enabled.

- legal transforms discovered: 122
- transformed candidates generated/measured: 95
- total measurements including baseline: 96
- accepted: 91 including baseline (90 transformed)
- rejected numerical: 5
- rejected verify/execution/nonfinite/memory: 0

All five numerical rejections were INT4 strategies. Their relative L2 was
about 0.0325--0.0335, above the fixed 0.02 bar. INT5 cleared the numerical gate
(relative L2 about 0.0154) but ran at about 5.0--5.6 ms and did not win.
Streamed-weight candidates reduced planned memory to as little as 521,232,384
bytes but ran at about 166--185 ms, so the latency objective correctly left
them unselected.

The automatically selected candidate was index 25:

```text
fuse_qkv_projection ops= tensors= params=
set_block_size ops=1,2,3,18,19,8,9,10,11,12,13,14,15,16,17 tensors= params=256
```

This is a generic structural QKV fusion plus generic scheduling choice; no
H3-specific optimizer transform was added.

Winner identities and source metrics:

- program fingerprint:
  `3b65ae106950ec700590dd7231bf446a7f96709afb833712497ae2668f43f67c`
- candidate fingerprint:
  `e9afa80261b7e104968018647a46582778a8aaa5d746d2faef4ff00f8deb4df6`
- cosine: 0.99999922393081997
- relative L2: 0.0012468770927852044
- max absolute: 4
- norm ratio: 1.0000498670616347
- nonfinite: 0
- BF16 exact mismatches: 10,493 / 21,504

Search-time minimums were 2.0531 ms baseline and 1.5012 ms winner, a
1.3677x ratio. The end-of-search baseline recheck was 2.0388 ms, drift ratio
0.9930175. The conservative multiplicative noise bound was 1.0226644
(2.266%), which became the effective winner margin; the selected difference is
well outside it.

An order-reversed, three-session, 10-warmup/100-iteration check measured:

| Order | Baseline session mean | Winner session mean |
|---|---:|---:|
| baseline then winner | 2.14513 ms | 1.56700 ms |
| winner then baseline | 2.13983 ms | 1.56705 ms |

The pair averages are 2.14248 ms and 1.567025 ms, a controlled **1.3672x**
ratio. The baseline order spread was 0.247%; the winner spread was 0.0032%.
The search's larger 2.266% bound remains the reported decision bound.

Planned memory fell from 1,523,155,712 to 1,292,174,080 bytes, a reduction of
230,981,632 bytes (15.16%). Actual backend resident bytes fell by the same
230,981,632 bytes, from 1,590,264,576 to 1,359,282,944 (14.52%).

## Cold preparation and profile attribution

Preparation is outside the objective. The very first uncached baseline process
reported 577.573 ms preparation. With the generated-code cache populated, the
profile-only split was:

| Component | Baseline | Winner |
|---|---:|---:|
| total preparation | 306.454 ms | 317.206 ms |
| resident checkpoint bytes | 1,291,145,216 | 1,291,145,216 |
| explicit host prefault | 69.7450 ms | 72.1621 ms |
| minor / major page faults in prefault | 19,702 / 0 | 19,702 / 0 |
| post-prefault resident staging + H2D wall | 83.9080 ms | 84.7464 ms |
| total resident upload boundary | 153.658 ms | 156.914 ms |
| streamed host stage / wait / H2D | 0 / 0 / 0 ms | 0 / 0 / 0 ms |

The post-prefault upload is a wall boundary after checkpoint page residency is
established; it still includes CUDA driver's pageable-memory staging as well as
H2D. It is not presented as a pure PCIe event duration.

Across 100 hot iterations, baseline operation events summed to 210.842 ms
(2.10842 ms/iteration), attention to 0.650112 ms
(0.00650112 ms/iteration), and non-kernel device timeline to 2.60669 ms
(0.0260669 ms/iteration). Baseline Linear/GEMM events sum to about 1.5220 ms per
iteration; generated non-Linear work is about 0.5799 ms, dominated by the
0.5163 ms checkpoint QKV-weight deinterleave.

Winner operation events summed to 154.074 ms (1.54074 ms/iteration), attention
to 0.763872 ms (0.00763872 ms/iteration), and non-kernel device timeline to
2.26294 ms (0.0226294 ms/iteration). Winner Linear/GEMM events sum to about
1.4794 ms; generated non-Linear work is about 0.0537 ms. The fused packed QKV
GEMM is 0.27285 ms versus baseline weight deinterleave plus three Q/K/V GEMMs
at about 0.8098 ms. Attention is not the source of this sequence-4 win.

## Clean replay and recurrence

Exact clean replay rebuilt candidate
`e9afa80261b7e104968018647a46582778a8aaa5d746d2faef4ff00f8deb4df6`;
the replayed and search-emitted DiffIR files compare byte for byte. The plan's
all-site strategy was then generalized with target-side legality checks. New
program fingerprints were
`b92174c99986acae03864ad0497c019798fdff025cf05c8f90b60c0480f45c6c`
at depth 3 and
`1c86241315cd5f40779810042c6bedff5ffbb6f7e7b5f1a4a5b7d284b7bed875`
at depth 5; these are intentionally new target identities, not claimed to be
exact replays of the one-block candidate.

| Depth | Path | Cosine | Relative L2 | Max abs | Norm ratio | Nonfinite | Exact mismatches |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 0.99999954879679664 | 0.000949951458742503 | 8 | 0.999998295452307 | 0 | 7,515 |
| 1 | winner | 0.99999922393081997 | 0.00124687709278520 | 4 | 1.00004986706163 | 0 | 10,493 |
| 3 | baseline | 0.99999874235280217 | 0.00158599111214330 | 32 | 1.00000740225333 | 0 | 11,269 |
| 3 | winner | 0.99999802018088213 | 0.00199217896978123 | 32 | 1.00009363781311 | 0 | 13,572 |
| 5 | baseline | 0.99999862794183381 | 0.00165653694988097 | 32 | 0.999999093114863 | 0 | 12,990 |
| 5 | winner | 0.99999791375562896 | 0.00204512555124249 | 32 | 1.00009818404354 | 0 | 14,636 |

The winner remains far inside the fixed H3 gate at depth 5 and its relative-L2
growth flattens from depth 3 to 5 rather than accelerating nonlinearly. Hot
diagnostic chain means were 6.19674 vs 4.58110 ms at depth 3 and 10.3854 vs
7.70612 ms at depth 5 (baseline vs winner); these were not separate optimizer
objectives.

## Exact reproduction commands

All commands below run from `/home/alex/diffusion-compiler-phase2`.

```sh
cmake -S . -B build-phase2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DDIF_CUDNN_ROOT=/home/alex/mojodiffusion/.pixi/envs/default
cmake --build build-phase2 --parallel 2
ctest --test-dir build-phase2 --output-on-failure

INDEX=/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer/model.safetensors.index.json
FIXTURE=/home/alex/diffusion-fixtures/data/minimax-h3-denoiser-native-preprocess-l1-source
ROOT=artifacts/phase2-h3-block

build-phase2/difcast "$FIXTURE/stage_packed_bf16.diftensor" \
  "$ROOT/source-oracle/input-1-hidden.diftensor" bf16 4 5376
cp "$FIXTURE/stage_temb_silu_bf16.diftensor" "$ROOT/source-oracle/input-2-temb-silu.diftensor"
cp "$FIXTURE/input_08_adaln_indices.diftensor" "$ROOT/source-oracle/input-3-adaln-indices.diftensor"
cp "$FIXTURE/stage_rotary_cos_bf16.diftensor" "$ROOT/source-oracle/input-4-cos.diftensor"
cp "$FIXTURE/stage_rotary_sin_bf16.diftensor" "$ROOT/source-oracle/input-5-sin.diftensor"
cp "$FIXTURE/block_stage_126_block_output.diftensor" "$ROOT/source-oracle/reference-depth-1.diftensor"

build-phase2/difc make-h3-transformer-bf16 "$ROOT/programs/block1.difir" 4 5376 56 128 14336 96 1 2 2688 128 resident split
build-phase2/difc make-h3-transformer-bf16 "$ROOT/programs/block3.difir" 4 5376 56 128 14336 96 3 2 2688 128 resident split
build-phase2/difc make-h3-transformer-bf16 "$ROOT/programs/block5.difir" 4 5376 56 128 14336 96 5 2 2688 128 resident split

build-phase2/difweights make-h3-bundle "$INDEX" "$ROOT/programs/block5.difir" "$ROOT/bundles/block5.difbind"
build-phase2/difweights verify-bundle "$ROOT/bundles/block5.difbind" "$ROOT/programs/block5.difir"
build-phase2/difweights rebind-h3-bundle "$ROOT/bundles/block5.difbind" "$INDEX" "$ROOT/programs/block1.difir" "$ROOT/bundles/block1.difbind"
build-phase2/difweights rebind-h3-bundle "$ROOT/bundles/block5.difbind" "$INDEX" "$ROOT/programs/block3.difir" "$ROOT/bundles/block3.difbind"
```

The common input arguments below are expanded explicitly in the actual run:

```sh
INPUTS="--bind 1=$ROOT/source-oracle/input-1-hidden.diftensor --bind 2=$ROOT/source-oracle/input-2-temb-silu.diftensor --bind 3=$ROOT/source-oracle/input-3-adaln-indices.diftensor --bind 4=$ROOT/source-oracle/input-4-cos.diftensor --bind 5=$ROOT/source-oracle/input-5-sin.diftensor"

build-phase2/difopt --program "$ROOT/programs/block1.difir" \
  --weight-bundle "$ROOT/bundles/block1.difbind" $INPUTS \
  --reference 39="$ROOT/source-oracle/reference-depth-1.diftensor" \
  --objective latency --backend cuda --warmups 5 --iterations 20 \
  --beam 4 --depth 3 --max-candidates 96 --margin 0.02 \
  --min-free-mib 4096 --blocks 64,128,256,512 --prefetch 1,2,4 \
  --quant-bits 4,5 --quant-groups 64 --max-abs 1e30 \
  --min-cos 0.999 --max-rel-l2 0.02 --min-norm-ratio 0.98 \
  --max-norm-ratio 1.02 --memory-budget-mib 20480 \
  --plan "$ROOT/search/winner-plan.json" \
  --journal "$ROOT/search/journal.json" \
  --out "$ROOT/search/winner.difir" \
  --db "$ROOT/search/measurements.diftune"

build-phase2/difopt --program "$ROOT/programs/block1.difir" \
  --weight-bundle "$ROOT/bundles/block1.difbind" $INPUTS \
  --replay "$ROOT/search/winner-plan.json" \
  --out "$ROOT/search/replayed-winner.difir"
cmp "$ROOT/search/winner.difir" "$ROOT/search/replayed-winner.difir"

build-phase2/difopt --program "$ROOT/programs/block3.difir" \
  --weight-bundle "$ROOT/bundles/block3.difbind" $INPUTS \
  --replay-global-strategy "$ROOT/search/winner-plan.json" \
  --out "$ROOT/search/winner-block3.difir"
build-phase2/difopt --program "$ROOT/programs/block5.difir" \
  --weight-bundle "$ROOT/bundles/block5.difbind" $INPUTS \
  --replay-global-strategy "$ROOT/search/winner-plan.json" \
  --out "$ROOT/search/winner-block5.difir"

build-phase2/difweights rebind-program "$ROOT/bundles/block1.difbind" \
  "$ROOT/search/replayed-winner.difir" "$ROOT/bundles/winner-block1.difbind"
build-phase2/difweights rebind-program "$ROOT/bundles/block3.difbind" \
  "$ROOT/search/winner-block3.difir" "$ROOT/bundles/winner-block3.difbind"
build-phase2/difweights rebind-program "$ROOT/bundles/block5.difbind" \
  "$ROOT/search/winner-block5.difir" "$ROOT/bundles/winner-block5.difbind"
```

Source recurrence regeneration uses fresh output directories:

```sh
python3 /home/alex/diffusion-fixtures/oracles/export_minimax_h3_denoiser.py \
  --output "$ROOT/source-depth3" --layers 3 --dump-block-stages 3
python3 /home/alex/diffusion-fixtures/oracles/export_minimax_h3_denoiser.py \
  --output "$ROOT/source-depth5" --layers 5 --dump-block-stages 5
cp "$ROOT/source-depth3/block_stage_194_block_output.diftensor" \
  "$ROOT/source-oracle/reference-depth-3.diftensor"
cp "$ROOT/source-depth5/block_stage_262_block_output.diftensor" \
  "$ROOT/source-oracle/reference-depth-5.diftensor"
```

The exact recurrence runner and comparison commands are:

```sh
run_block() {
  PHASE2_PROGRAM=$1
  PHASE2_BUNDLE=$2
  PHASE2_OUTPUT_ID=$3
  PHASE2_OUTPUT=$4
  build-phase2/difrun --backend cuda --program "$PHASE2_PROGRAM" \
    --weight-bundle "$PHASE2_BUNDLE" \
    --input 1="$ROOT/source-oracle/input-1-hidden.diftensor" \
    --input 2="$ROOT/source-oracle/input-2-temb-silu.diftensor" \
    --input 3="$ROOT/source-oracle/input-3-adaln-indices.diftensor" \
    --input 4="$ROOT/source-oracle/input-4-cos.diftensor" \
    --input 5="$ROOT/source-oracle/input-5-sin.diftensor" \
    --output "$PHASE2_OUTPUT_ID=$PHASE2_OUTPUT" \
    --warmups 5 --iterations 20 --session-runs 2 --min-free-mib 4096 \
    --cache-dir "$ROOT/cache"
}

run_block "$ROOT/programs/block1.difir" "$ROOT/bundles/block1.difbind" 39 "$ROOT/runs/baseline-depth1.diftensor"
run_block "$ROOT/search/replayed-winner.difir" "$ROOT/bundles/winner-block1.difbind" 39 "$ROOT/runs/winner-depth1.diftensor"
run_block "$ROOT/programs/block3.difir" "$ROOT/bundles/block3.difbind" 107 "$ROOT/runs/baseline-depth3.diftensor"
run_block "$ROOT/search/winner-block3.difir" "$ROOT/bundles/winner-block3.difbind" 107 "$ROOT/runs/winner-depth3.diftensor"
run_block "$ROOT/programs/block5.difir" "$ROOT/bundles/block5.difbind" 175 "$ROOT/runs/baseline-depth5.diftensor"
run_block "$ROOT/search/winner-block5.difir" "$ROOT/bundles/winner-block5.difbind" 175 "$ROOT/runs/winner-depth5.diftensor"

for PHASE2_DEPTH in 1 3 5; do
  for PHASE2_PATH in baseline winner; do
    build-phase2/difcompare \
      "$ROOT/source-oracle/reference-depth-$PHASE2_DEPTH.diftensor" \
      "$ROOT/runs/$PHASE2_PATH-depth$PHASE2_DEPTH.diftensor" \
      --min-cos 0.999 --max-rel-l2 0.02 --min-norm-ratio 0.98 \
      --max-norm-ratio 1.02 --max-abs 1e30
  done
done
```

For the controlled one-block A/B, call `run_block` in baseline/winner then
winner/baseline order with `--warmups 10 --iterations 100 --session-runs 3`.
Add `--profile-pipeline` to one run of each program for the attribution above.
Every candidate record is in
`artifacts/phase2-h3-block/search/journal.json`.

## Promotion boundary

A 50-block experiment is still blocked on review of this Phase 2 result. Even
after review, sequence 4 is a source-faithful synthetic packing gate, not native
production video geometry. A future 50-block pass must retain the sealed
checkpoint identity, use an independently captured 50-block source result,
carry the same pressure and timing controls, and reassess recurrence error at
the larger depth. No 50-block optimization was started here.
