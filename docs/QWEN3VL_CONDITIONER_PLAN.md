# Qwen3-VL conditioner plan: native prompt -> [439,5120] BF16 conditioning

Date: 2026-08-31. Branch `w3-conditioner`. Companion to the landed native
tokenizer (`src/text/`, `tools/diftokenize.cpp`, `dif_tokenizer_tests`) —
together they are removal-order item 5 (the XL long pole) in
`docs/FLAME_CPP_RUNTIME_PORT.md` §5: replace the Serenity Mojo
tokenizer+conditioner dependency with a streamed DiffIR program over the
checkpoint's own `text_encoder/` shards.

This document is an expressibility audit and build plan. It implements
nothing. Every claim is marked VERIFIED (with file:line or tool-result
evidence) or HYPOTHESIS (with the check that will settle it).

---

## 0. Status of the tokenizer half (DONE, gated)

The conditioner's input contract is already satisfied natively:

- `dif::text::QwenBpeTokenizer` reproduces `transformers`
  `Qwen2TokenizerFast` (`add_special_tokens=False`) including the seven
  tokens H3 declares only in `tokenizer_config.json`
  (`<d>`,`</d>`,`<|cutoff|>`, lyrics/caption pairs; ids 151669..151675).
- VERIFIED: golden prompt -> exactly 439 ids. Stripped-of-trailing-newline
  variant (the byte string Serenity actually encoded, per the recorded
  `serenity/reference_bf16/result.json` prompt field) has id-sequence
  SHA-256 `3e8fe983c658547cd112df4feadc166642cc9ff535d742e217b7d95900671173`
  (last id 13); the raw file bytes give 439 ids ending in 624. Both match
  the transformers oracle byte-for-byte.
- VERIFIED: 64 embedded parity fixtures + 3,500-case random fuzz vs the
  oracle: 0 mismatches. 7/7 ctest suites green including the new
  `dif_tokenizer_tests` (skip-code 77 when the checkpoint is absent).
- `diftokenize --diftensor-out` already emits the ids as an I32 `.diftensor`
  — the natural input artifact for the future conditioner program.

## 1. Encoder architecture (from the checkpoint, not from docs)

Checkpoint: `/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder/`
— index SHA-256 `06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a`
(matches `artifacts/h3-quality-natural-language-2026-08-30/input-manifest.json`
`conditioning.encoder_index_sha256`). 14 shards, 1058 tensors,
66,714,780,128 bytes, all BF16 (shard-0 header read directly).

`config.json` (`Qwen3VLForConditionalGeneration`, `text_config`) — VERIFIED
by direct read:

| field | value |
|---|---|
| num_hidden_layers | 64 (only 0..49 executed — see extraction rule) |
| hidden_size | 5120 |
| num_attention_heads / num_key_value_heads | 64 / 8 (GQA group 8) |
| head_dim | 128 |
| intermediate_size | 25600, hidden_act silu (SwiGLU) |
| rms_norm_eps | 1e-6 |
| rope_theta | 5,000,000; rope_scaling mrope_section [24,20,20], mrope_interleaved true |
| vocab_size | 151936; attention_bias false; dtype bfloat16 |

Weight map (VERIFIED from the safetensors index + shard-0 header shapes):

```
model.language_model.embed_tokens.weight            BF16 [151936, 5120]
model.language_model.layers.N.input_layernorm.weight          [5120]
model.language_model.layers.N.self_attn.q_proj.weight         [8192, 5120]
model.language_model.layers.N.self_attn.k_proj.weight         [1024, 5120]
model.language_model.layers.N.self_attn.v_proj.weight         [1024, 5120]
model.language_model.layers.N.self_attn.q_norm.weight         [128]
model.language_model.layers.N.self_attn.k_norm.weight         [128]
model.language_model.layers.N.self_attn.o_proj.weight         [5120, 8192]
model.language_model.layers.N.post_attention_layernorm.weight [5120]
model.language_model.layers.N.mlp.gate_proj.weight            [25600, 5120]
model.language_model.layers.N.mlp.up_proj.weight              [25600, 5120]
model.language_model.layers.N.mlp.down_proj.weight            [5120, 25600]
model.language_model.norm.weight  [5120]   (NOT used — pre-norm extraction)
lm_head.weight                             (NOT used)
model.visual.* (351 keys, vision tower)    (NOT used — t2va has no image)
```

No biases anywhere in the text tower (`attention_bias: false`; none in the
index).

### Extraction rule (the subtle part) — VERIFIED

Output = `hidden_states[50]`: the RAW residual stream after layers 0..49
have run, BEFORE `model.norm` and before layer 50 executes. Serenity
verified transformers' indexing empirically (index k = raw state after k
layers; only the LAST tuple entry is overwritten post-norm):
`serenitymojo/models/text_encoder/minimax_h3_qwen3vl_streamed.mojo:12-27`
and the oracle script
`serenitymojo/models/text_encoder/parity/minimax_h3_conditioner_real_weight_oracle.py`
(docstring "INDEXING", which sidesteps `output_hidden_states` entirely by
manually looping `model.layers[i]`). Output shape [439, 5120] BF16;
recorded payload SHA-256
`9b1609bdc8c02365d386844cbeae988aaa66d72847528f0a6d3e5b24b1d89585`.

### Per-layer math — VERIFIED against the working Serenity reference

`serenitymojo/models/text_encoder/qwen3_encoder.mojo:720-820` (`_layer`),
the exact code path of the accepted 2026-08-30 conditioning run
(`minimax_h3_conditioning.mojo:103-167` wires it):

```
x1 = RMSNorm(x, input_layernorm, eps=1e-6)
q = x1 @ q_proj^T          # [S, 8192] -> [S, 64, 128]
k = x1 @ k_proj^T          # [S, 1024] -> [S, 8, 128]
v = x1 @ v_proj^T          # [S, 1024] -> [S, 8, 128]
q = RMSNorm_perhead(q, q_norm[128], eps) ; k likewise      # over head_dim
q = RoPE_halfsplit(q, cos, sin) ; k likewise               # full 128 dims
k,v repeated 8 -> 64 heads (GQA)                            # qwen3_encoder.mojo:788-789
attn = causal SDPA(q, k, v, scale = 1/sqrt(128))
x = x + attn @ o_proj^T
x2 = RMSNorm(x, post_attention_layernorm, eps)
x = x + (silu(x2 @ gate^T) * (x2 @ up^T)) @ down^T
```

### MRoPE for text-only prompts — VERIFIED mechanism, one inherited flag

`minimax_h3_conditioning.mojo:44-53`: the reference builds
`mm_token_type_ids` (0=text,1=image,2=video) and derives 3-axis rotary
positions from modality runs; an all-text prompt is all zeros, so all three
axes carry identical sequential positions and MRoPE collapses to ordinary
1-D RoPE regardless of `mrope_section`/`mrope_interleaved` — position ids
0..S-1, `inv_freq[i] = theta^(-2i/128)`, rotate-half pairing.
HYPOTHESIS (inherited, flagged in the oracle script's own docstring): the
`Qwen3VLTextModel` sequential-position fallback equals
`Qwen3VLModel.get_rope_index` for text-only input. The accepted H3 run used
exactly this collapse and passed its visual gate, and the per-layer parity
ladder (§5) verifies it numerically end-to-end; it needs re-derivation only
if a vision/keyframe path is ever added.

### Serenity padding policy — relevant difference

Serenity pads ids to a power of two (439 -> 512, pad id 151643) because its
Mojo SDPA dispatch is comptime-enumerated, then slices rows back
(`minimax_h3_conditioning.mojo:128-166`); causality makes right-pad rows
inert for rows < 439. The DC runtime has no comptime shape table, so the
program can be built at exact S=439. Padding is a frontend policy choice,
not a requirement (see §5 padding gate).

## 2. Op-by-op mapping to EXISTING DiffIR opcodes

Opcode inventory: `include/dif/ir/ir.hpp:34-73`; verifier contracts in
`src/ir/verify.cpp`; CPU reference `src/runtime/cpu_executor.cpp`.

| Encoder step | DiffIR opcode | Evidence / fit |
|---|---|---|
| token ids input | I32 `.diftensor` from `diftokenize` | landed this branch |
| embedding lookup | `GatherRows` (19) | verify.cpp:339 — float [V,...] + i32 [M] -> [M,...]; embed table [151936,5120] is row-major by token. EXACT fit |
| input/post layernorm | `RmsNorm` (17) | existing, used by H3 denoiser |
| q/k/v/o/gate/up/down projections | `Linear` (6) | cuBLASLt LinearPlan; bias-less (attention_bias false) |
| cos/sin table build | `RotaryPosition` (24) | verify.cpp:420 — positions f32 [S,1], inv_freq f32 [64] -> cos/sin [S,128]. EXACT fit |
| q_norm/k_norm + RoPE | `QkNormPartialRope` (7) | verify.cpp:601 — input [S,H,128], weight [128], RotaryDim=128 (full-dim is the legal rotary==head_dim case). CPU reference cpu_executor.cpp:726-772 does RMSNorm over head_dim (weight [D], F32 accumulate) then EXACTLY rotate-half: out[d]=x[d]cos−x[d+half]sin, out[d+half]=x[d+half]cos+x[d]sin — the HF convention. VERIFIED no new rope op or layout attr needed |
| GQA head repeat | **GAP — see §3** | `Attention` (8) verifier requires q,k,v same shape (verify.cpp:634-636) |
| causal SDPA | `Attention` (8), `Causal` attr (AttrKey 7) | cpu_executor.cpp:784 (key_end=query+1); `CudnnAttentionPlan(query, scale, causal)` include/dif/runtime/cudnn_attention.hpp:13. Implementation 1 (generated, S<=4096 OK for 439/512) or 2 (cuDNN BF16) |
| attn reshape [S,64,128]<->[S,8192] | free | DiffIR tensors are dense row-major; [S,H,D] and [S,H*D] are the same bytes — the frontend declares the desc shape each consumer needs (same trick the H3 denoiser uses between deinterleave and linear) |
| SwiGLU | `SwiGlu` (4) + `GateFirst` attr | matches silu(gate)*up |
| residual adds | `Add` (1) | trivial |
| row slice if padded | `SelectRowChunks` (22) | existing |
| dtype boundaries | `Cast` (21) | BF16 storage / F32 accumulate contract already repo-wide |

Everything except GQA maps onto existing opcodes with no semantic
stretching.

## 3. Gap list (hand to the IR owner — NOT implemented here)

1. **GQA in `Attention` — the only hard gap.** The verifier pins q,k,v to
   identical [S,H,D] shapes. Two options, in preference order:
   - **(a) `KvHeads` attribute on `Attention`** (new `AttrKey`): k/v become
     [S,Hkv,D] with H % Hkv == 0; head h reads kv head h/(H/Hkv).
     Generated-kernel change is an index-map tweak; cudnn_frontend SDPA
     supports GQA natively via differing head counts on the K/V descriptors.
     No materialized 8x copy of K/V (for this model: saves 2x
     [S,64,128]−[S,8,128] BF16 ≈ 12.9 MB per layer at S=439 — small, but
     the attr also keeps candidate identity clean for difopt).
   - **(b) new `RepeatHeads` opcode** ([S,Hkv,D] -> [S,H,D], attr Heads):
     dumb-copy semantics, matches Serenity's `_repeat_kv` precedent
     (qwen3_encoder.mojo:788-789), zero attention-kernel changes. Fallback
     if (a) is unwelcome.
   Per difopt discipline (FLAME_CPP_RUNTIME_PORT.md §1), whichever lands
   must enter candidate identity, not ad-hoc runtime state.
2. **Nothing else.** Notably NOT needed: reshape/view op (dense row-major
   makes the [S,H,D]/[S,H*D] boundary a descriptor choice), a new rope
   arm (rotate-half already exact, §2), an mrope op (collapses to 1-D for
   text), embedding op (GatherRows), or any attention-mask tensor (Causal
   attr suffices; right-padding is inert under causal masking).

## 4. Memory / streaming plan (RTX 3090 Ti, 24 GiB)

Per-layer weights (BF16, from §1 shapes): attention 94.4 M params
(q 8192x5120, k/v 2x 1024x5120, o 5120x8192) + MLP 393.2 M
(3x 25600x5120) + 4 norm vectors ≈ 487.6 M params ≈ **0.98 GiB/layer**
(matches Serenity's measured "~0.95 GiB",
minimax_h3_qwen3vl_streamed.mojo STREAMING STRATEGY header). Embed table
1.556 GiB, used once. 50 layers streamed = ~49 GiB disk reads **once per
prompt** (conditioning runs before, not inside, the denoise loop — the same
one-time 15-30 s NVMe tax Serenity states plainly).

Existing machinery to reuse as-is (no new code class):

- `.difbind` SHA-sealed bundles + per-constant **Streamed** role bit
  (docs/RUNTIME_MODULES.md "Weights"; src/weights/bundle.cpp).
- Plan-slot realization + liveness intervals + **depth-1 dual-stream weight
  prefetcher** (FLAME_CPP_RUNTIME_PORT.md §1) — the H3 denoiser already
  streams 50 blocks of a 26 GiB transformer per evaluation; the conditioner
  is the same problem shape at ~2x the per-layer size and 1/19th the
  execution count.
- VRAM: ~2 layers resident (prefetch depth 1) ≈ 2.0 GiB + activations.
  Activations are trivial: residual [439,5120] BF16 ≈ 4.3 MiB; the largest
  intermediate is MLP gate/up [439,25600] ≈ 21.5 MiB each. Peak well under
  4 GiB — no contest with anything else since the conditioner runs
  standalone.
- Host: run under `scripts/mem_safe_runtime.sh` (MemoryMax=24G) like all
  accepted runs; the mmap-page-residency lesson is already encoded there
  (the first H3 launch's host OOM,
  docs/H3_NATURAL_LANGUAGE_QUALITY_GATE_2026-08-30.md "Runtime results").
  Serenity additionally MADV_DONTNEEDs consumed shard pages
  (streamed.mojo header); verify the bundle reader's residency is bounded
  the same way at 49 GiB scan scale — measurement item, not new design.
- Program size: 50 layers x ~13 ops ≈ 700 ops — comparable to the
  denoiser; no codec/verifier concern.

## 5. Gate design

Precedent to reuse, not reinvent: the proven oracle method in
`minimax_h3_conditioner_real_weight_oracle.py` — instantiate transformers'
OWN `Qwen3VLTextModel`, load REAL checkpoint shards, run GPU BF16 (never
CPU/F32: a wrong-dtype reference diverges more than the port under test),
manually loop layers to read raw per-depth hidden states. Torch stays a dev
oracle per policy (FLAME_CPP_RUNTIME_PORT.md §5 tail).

Gate ladder (each is a tool-run with recorded numbers, per repo testing
discipline §8):

1. **Ids gate — DONE** (§0): 439 ids, oracle-identical sha.
2. **Embed gate**: GatherRows output vs oracle `embed_tokens(ids)` —
   bit-exact required (pure row copy, no arithmetic).
3. **Per-op fixtures** for the two nontrivial ops at encoder shapes:
   QkNormPartialRope with RotaryDim=128=head_dim vs torch
   (q_norm+rotate_half), and GQA Attention (once the §3 opcode lands) vs
   torch SDPA with enable_gqa/repeat_kv. Torch-oracle fixture files, same
   pattern as the existing training gates.
4. **Depth ladder**: compiler program sliced at depth 1, 23, 50 (mirror the
   oracle's depths; 23 was chosen when shard 6 was missing — with all 14
   shards on disk, extend the oracle to depth 50) vs oracle raw hidden
   states. Report cosine / rel-L2 / max-abs / bit-mismatch per boundary,
   first-divergence localization — the same table format as the denoiser
   trajectory in H3_NATURAL_LANGUAGE_QUALITY_GATE_2026-08-30.md.
5. **Recorded-payload comparison**: full 50-layer output [439,5120] BF16 vs
   payload sha `9b1609bd...`. Report bit mismatches, cosine, max-abs.
   HONESTY CLAUSE: bit-identity with Serenity's payload is the diagnostic
   target, not the acceptance bar — Serenity's numbers come from its own
   Mojo kernel stack (cshim cuDNN SDPA, its own rms/rope kernels), and the
   denoiser side already shows the two stacks differ within-BF16 (block-1
   cosine 0.999979 with 48.9 M bit mismatches, gate doc). ACCEPTANCE =
   depth-50 parity vs the transformers-own-math oracle at BF16 noise floor
   (per-boundary cosine/rel-L2 in line with the oracle's own
   run-to-run BF16 spread), with the payload comparison published alongside.
   If the compiler output happens to be bit-identical to either reference,
   record it; do not engineer for it.
6. **Padding A/B**: S=439 exact vs padded-512-then-`SelectRowChunks` must
   agree bit-for-bit under Causal (cheap; pins the frontend policy).
7. **End-to-end proof**: feed the native conditioning into the existing
   accepted denoise path in place of the imported Serenity payload and
   re-run the visual gate — the final Mojo-dependency-removal claim for
   point (1) of the honest-claim list in FLAME_CPP_RUNTIME_PORT.md §5.

## 6. Effort estimate (implementable chunks, honest)

| # | Chunk | Est. | Depends on |
|---|---|---|---|
| 1 | GQA in Attention (§3, IR owner: attr + verifier + CPU ref + generated CUDA + cuDNN plan + fixtures) | 1-2 days | IR owner |
| 2 | Frontend builder: embed + single layer + extraction at fixed S; per-op fixtures (gate 2-3) | 1-2 days | 1 |
| 3 | 50-layer assembly + bundle/Streamed wiring + memory-plan validation under mem-safe scope | 1-2 days | 2 |
| 4 | Oracle extension to depth 50 + depth-ladder parity run + payload comparison (gates 4-6) | 1-2 days | 3, GPU time |
| 5 | `difcondition` tool (ids .diftensor -> conditioning .diftensor + manifest, difimport-style) + docs | 0.5-1 day | 3 |
| 6 | End-to-end swap into the accepted denoise chain + visual re-gate (gate 7) | 1 day | 4, 5 |

Total ≈ **6-9 focused days**, of which the only external dependency is the
GQA opcode decision (chunk 1). Everything else composes from EXISTING,
already-proven machinery: GatherRows/RmsNorm/Linear/QkNormPartialRope/
RotaryPosition/SwiGlu/Attention+Causal, cuBLASLt plans, streamed bundles,
the depth-1 prefetcher, and the established oracle method.

## Open hypotheses (all with a settling check in §5)

- H1: `get_rope_index` == sequential positions for text-only (inherited
  flag; settled by gate 4 at depth 1).
- H2: cuDNN GQA path bit-stability vs generated fallback at S=439 (settled
  by gate 3 fixture A/B; fallback is admitted for S<=4096 regardless).
- H3: bundle-reader host-page residency bounded at 49 GiB streaming scale
  (settled by the mem-safe scope counters during gate 4's full run).
