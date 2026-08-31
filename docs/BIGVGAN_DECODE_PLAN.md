# BigVGAN Native Audio Decode — Implementation Plan (DiffIR + difaudiodecode)

Read-only scout report, 2026-08-31. Replaces removal-order step 4 of
`docs/FLAME_CPP_RUNTIME_PORT.md` ("L — native BigVGAN audio decode as a DiffIR
program"), i.e. removes the last-but-one Mojo dependency: the fresh-process
call to `/home/alex/mojodiffusion/output/bin/minimax_h3_decode_768x768x124
decode <dir> ... decode_audio_only` made by
`artifacts/h3-quality-natural-language-2026-08-30/compiler/decode_audio_and_mux_full_exact_bf16.sh`.
Everything below marked (V) is verified with file:line; (H) is hypothesis.

## 1. What the decoder is (V)

Sources of truth, in trust order:

1. Pinned diffusers reference: `/home/alex/minimax_h3_ref/diffusers/autoencoder_kl_minimax_h3_audio.py`
   (diffusers PR huggingface/diffusers#14355 @ e1b518df per the Mojo port
   header). NOTE: it is NOT under `diffusers-src/` — that tree has no H3 audio
   VAE; the file lives one level up, plus vendor copies at
   `/home/alex/minimax_h3_ref/creator-MiniMax-H3/FL2VA/audio_vae/dac_bigvgan.py`
   and `/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/audio_vae/dac_bigvgan.py`.
2. Gated Mojo host oracle: `serenitymojo/models/minimax_h3/audio_decoder.mojo`
   (557 lines, op-for-op port; parity gates in
   `serenitymojo/models/minimax_h3/parity/` — 12 staged checks at ~1e-6 vs the
   vendor's own `dac_bigvgan.py`, per the device port's header).
3. Device port `serenitymojo/models/minimax_h3_device/audio_decoder_device.mojo`
   (gate 11/11 vs host oracle; e2e waveform cos 0.999999999994677, max_abs
   5.7e-6 — cited at `pipeline/minimax_h3_i2va.mojo:790-792`).

No divergence found between the Mojo port and the diffusers reference; the
Mojo file documents itself as op-for-op and its parity gate enforces it. One
device-side shortcut must NOT be inherited: `audio_decoder_device.mojo:27-38`
implements the 2x upsampler's transposed conv as zero-insert + UNFLIPPED
conv1d, valid only because all 254 Kaiser filters are exactly symmetric
(measured, F32-exact). The C++ port should implement a real transposed conv
and not depend on filter symmetry.

### Architecture (V, from audio_decoder.mojo + reference :395-491 + checkpoint)

Waveform out directly — no mel, no separate vocoder. Config (V,
`audio_vae/metadata.json` + `pipeline/minimax_h3_i2va.mojo:664-678`):
latent_channels=32, latent_dim=2048, decoder_dim=1024 (upsample_initial_channel),
rates (5,5,2,2,2,2,2) — total x800, up-kernels (9,9,4,4,4,4,4),
resblock kernels (3,7,11), resblock dilations ((1,3,5))x3, sample_rate 32000.

Layer list, input `[B, 32, T]` (B=2 = stereo as batch items,
reference :624-640):

1. `dec_in_proj`: plain Conv1d 32→2048, K=1 — the ONLY conv that is NOT
   weight-normed (audio_decoder.mojo:388-397).
2. `decoder.conv_pre`: wn-Conv1d 2048→1024, K=7, pad 3.
3. 7 upsample stages i=0..6, channels 1024→512→…→8:
   a. `decoder.ups.i.0`: wn-ConvTranspose1d C→C/2, K=k_i, stride=r_i,
      padding=(k_i−r_i)//2 ⇒ L_out = L·r_i exactly.
   b. 3 AMP blocks (`decoder.resblocks.{3i..3i+2}`), one per resblock kernel
      K∈{3,7,11}; each block: for each dilation d∈{1,3,5}:
      `x += conv2(act2(conv1(act1(x))))` where conv1 = wn-Conv1d C→C, K,
      dilation d, pad (K·d−d)//2; conv2 = same K, dilation 1, pad (K−1)//2;
      act = alias-free SnakeBeta (below). Activations interleave: index 2d
      feeds convs1, 2d+1 feeds convs2 (reference :421-426).
   c. The 3 block outputs are AVERAGED (sum / 3, reference :484-489). Trap:
      dropping the divide stays plausible, 3x too loud.
4. `decoder.activation_post` (alias-free SnakeBeta on 8 ch).
5. `decoder.conv_post`: wn-Conv1d 8→1, K=7, pad 3, NO BIAS (the only bias-less
   conv).
6. clamp to [−1,1] — part of the model (reference :491).

**SnakeBeta** (reference :140-155): `x + (exp(β)+1e-9)^−1 · sin(exp(α)·x)²`,
α, β per-channel `[C]` vectors stored in LOG space. Trap: linear treatment is
a near-identity that looks plausible.

**Alias-free activation** (reference :158-226): upsample 2x → SnakeBeta →
downsample 2x, both with the depthwise K=12 Kaiser-sinc filter that ships in
the checkpoint as buffers (`…upsample.filter`, `…downsample.lowpass.filter`,
all `[1,1,12]`). Filters are LOADED, never recomputed (audio_decoder.mojo:33-38).
Upsample: replicate-pad by pad=K//r−1 each side, depthwise ConvTranspose1d
stride r pad 0, multiply by r, then trim pad_left=pad·r+(K−r)//2 /
pad_right=pad·r+(K−r+1)//2. Downsample: replicate-pad ASYMMETRIC
(left K//2−even=5, right K//2=6 for K=12), depthwise strided conv.

**Weight norm** (audio_decoder.mojo:20-31, 56-88): every conv except
dec_in_proj ships as `weight_g`/`weight_v`; effective w = g·v/‖v‖ with the
norm per dim-0 slice. MUST be folded at load with FLOAT64 accumulation of the
sum of squares — sequential F32 put folded weights 2.5-3.4e-6 relative off on
the widest folds (measured, documented in the Mojo file). For ConvTranspose
weights `[C_in, C_out, K]`, dim 0 is the INPUT channel
(audio_decoder.mojo:448-450).

**Latent denormalization** (V, `pipeline/minimax_h3_i2va.mojo:693-788`):
x·std_c + mean_c per channel, 32 mean/std constants — canonical copy in
`audio_vae/config.json` (`latents_mean`/`latents_std`), duplicated as
literals in both Mojo pipelines. Applied to latents BEFORE decode (the
diffusers `decode()` takes already-denormalized latents, :624-633).

**Geometry of the accepted run** (V,
docs/H3_NATURAL_LANGUAGE_QUALITY_GATE_2026-08-30.md:57,71): audio state rows
`[584,32]` F32 (sha 20a45cdd…) = channel-major rows, stereo channel 0's 292
rows then channel 1's 292 (unpack semantics at
`serenitymojo/models/minimax_h3/rearrange.mojo:121-139` → `[2,32,292]`).
292 × 800 = 233,600 samples/ch at 32 kHz stereo = 7.3 s, matching the 175
frame / 24 fps video. Output artifact: 16-bit PCM stereo WAV
(`serenitymojo/audio/wav.mojo`: 44-byte RIFF header, interleaved L/R,
clamp then i16 = round(x·32767)).

## 2. Weights (V, safetensors header read)

`/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/audio_vae/model.safetensors`
(605 MB file, encoder+decoder, 1087 tensors, all F32). Decoder side = 914
tensors, 64,939,353 params (259.8 MB F32). Name patterns:

- `dec_in_proj.{weight [2048,32,1], bias}`
- `decoder.conv_pre.{weight_g [1024,1,1], weight_v [1024,2048,7], bias}`
- `decoder.ups.N.0.{weight_g, weight_v, bias}` ×7 (weight_v `[C_in, C_out, K]`)
- `decoder.resblocks.N.convs{1,2}.M.{weight_g, weight_v, bias}` ×126
- `decoder.resblocks.N.activations.M.act.{alpha,beta}` `[C]` ×126 pairs
- `decoder.resblocks.N.activations.M.{upsample.filter, downsample.lowpass.filter}` `[1,1,12]` ×126 pairs
- `decoder.activation_post.*` (alpha/beta `[8]`, 2 filters), `decoder.conv_post.{weight_g [1,1,1], weight_v [1,8,7]}` (no bias)

After folding, resident F32 decoder weights ≈ 235 MB. Trivial next to 24 GB.

## 3. DiffIR gap analysis

Op budget today (V, `include/dif/ir/ir.hpp:34-73`): 38 opcodes; relevant
existing pieces: `Add` (same-shape elementwise, verify.cpp:93), `Multiply`,
`Fill` (constant tensor, verify.cpp:331), `Clamp` (Lower/Upper attrs,
verify.cpp:149-161), `Cast`, `AffineLastDim` (last-dim only — NOT usable for
per-channel work on `[B,C,L]`). There is NO conv of any kind, no sin/exp
elementwise, no transpose (`docs/RUNTIME_MODULES.md` "Missing for FLUX/WAN
class: … Conv/GroupNorm/upsample"). Verdict: the whole decoder is expressible
with exactly TWO new opcodes.

### New opcode 1: `Conv1d` (Opcode 39)

One general op covering plain / dilated / grouped / transposed / depthwise,
zero- or replicate-padded, with output trim — no model-specific opcodes,
matching OPTIMIZER.md's generality rule. Also used for FLUX/WAN-class conv
work later (1d now; the attr scheme extends to a future Conv2d).

- Inputs: `input [B, C_in, L]` (rank 3), `weight`, optional `bias [C_out]`.
- Output: `[B, C_out, L_out]`.
- Attrs (new AttrKeys, continuing at 30): `Stride`(u64 ≥1, default 1),
  `Dilation`(u64 ≥1, default 1; must be 1 when Transposed),
  `Groups`(u64 ≥1, divides C_in and C_out; NOT reusing quant `GroupSize` —
  different semantics), `PadLeft`/`PadRight`(u64 ≥0),
  `PadMode`(u64: 0=zero, 1=replicate), `Transposed`(bool),
  `TrimLeft`/`TrimRight`(u64 ≥0, Transposed only).
- Semantics, non-transposed: pad input per PadMode, then standard correlation.
  `weight [C_out, C_in/g, K]`;
  `L_out = (L + PadL + PadR − (Dilation·(K−1)+1))/Stride + 1` (verifier
  requires exact divisibility is NOT required — floor, but require L_out ≥ 1).
- Semantics, transposed (matches audio_decoder.mojo order: pad input in
  sample space → scatter with stride → trim output):
  `weight [C_in, C_out/g, K]`; `full = (L + PadL + PadR − 1)·Stride + K`;
  `L_out = full − TrimL − TrimR ≥ 1`. Torch `conv_transpose1d(padding=P)` ≡
  PadL=PadR=0, TrimL=TrimR=P. BigVGAN upsampler ≡ replicate PadL=PadR=pad,
  Trim as computed by the builder. The ×ratio scale is folded into the filter
  constant at import (×2 is exact in F32).
- Dtype rules: input/weight/bias/output same dtype ∈ {F32,BF16,F16}; F32
  accumulate (the existing `dif_load/dif_store` contract,
  compiler.cpp:172-256). This model runs all-F32.
- Verifier rule style: mirror `verify.cpp:105-161` (expect counts 2-or-3/1,
  shape/dtype checks, attr range checks, dedicated failure strings; both
  weight layouts checked under the Transposed flag — the layout swap is the
  classic port bug).
- CPU reference sketch: direct loops, exactly the proven ordering of
  `audio_decoder.mojo` `conv1d` (:104-160) and `conv_transpose1d` (:163-215)
  with the pad materialized or index-mapped; F32 accumulator per output
  element (double accumulate NOT needed at runtime — only the weight-norm
  fold needs float64, and that happens at import).

### New opcode 2: `SnakeBeta` (Opcode 40)

BigVGAN/DAC-family activation, not model-specific (also LTX-2's vocoder —
serenitymojo has it as a standalone gated op, `ops/snake.mojo` precedent).

- Inputs: `input [B, C, L]`, `alpha [C]`, `beta [C]` (log-space, straight from
  checkpoint). Output: same shape/dtype as input.
- Attr: `Epsilon` (F64, default 1e-9).
- Semantics: `y = x + (exp(β_c)+ε)^−1 · sin(exp(α_c)·x)²`. exp of the
  per-channel params may be precomputed per kernel launch (C ≤ 1024) but the
  IR contract stores log-space so the checkpoint binds unmodified.
- Dtype: float set, F32 compute. Verifier: counts 3/1, rank-3 input, vector
  params matching dim 1.

Deliberately NOT added: Pad op (folded into Conv1d as PadMode — replicate pad
only ever feeds a conv here), Sin/Exp elementwise generics (would need
per-channel broadcast machinery the IR lacks; SnakeBeta is the honest
primitive), an Upsample1d macro-op (it is exactly Conv1d-transposed +
composition).

### Everything else maps to existing ops (V)

- Residual adds and block accumulation → `Add`.
- ÷3 block average → `Fill`(value 1/3, same shape) + `Multiply` (Fill exists,
  verify.cpp:331; slot reuse makes the extra tensor free).
- Final clamp → `Clamp` (Lower −1 / Upper 1).
- Latent denorm → the SAME `Conv1d`: depthwise K=1, groups=32, weight=std
  `[32,1,1]`, bias=mean — the program becomes self-contained, constants baked
  by the builder from `audio_vae/config.json`.
- Row unpack `[584,32] → [2,32,292]` → host-side in the tool (mirror
  rearrange.mojo:121-139); it's 18,688 floats, not worth IR support.

### Program shape (V, counted from the layer list)

One program, shapes baked for (B=2, T=292): ≈570 ops — ≈391 Conv1d, 127
SnakeBeta (126 checkpoint activations + activation_post — matches the tensor
census exactly), ≈52 Add/Fill/Multiply/Clamp. ~600 ops is small next to the
H3 denoiser programs. (H) NVRTC one-kernel-per-op on a ~570-op TU compiles
acceptably and is amortized by the existing SHA-keyed PTX cache; if compile
time bites, dedupe entrypoints by (opcode, shape, attrs) signature — many AMP
convs repeat per stage.

## 4. Memory / perf (V sizes; H timings)

- Folded weights ≈ 235 MB F32 resident (no streaming needed). Peak activation
  ≈ 30 MB per tensor (largest: stage-1 alias-free up-buffer `[2,256,14600]`
  and tail `[2,8,467200]`); with liveness slot reuse total device footprint
  well under 1 GB. Fits trivially in 24 GB — but KEEP the separate-process
  boundary: it is deliberate policy (t2va.mojo:2632-2636: audio and video
  decoders must not share one CUDA process on long generations), and the
  current bridge already runs audio decode as its own process.
- Compute ≈ 130 GMAC ≈ 260 GFLOP F32 for the accepted geometry (dominated by
  stage-0/1 AMP convs and conv_pre). (H) Naive per-element CUDA kernels land
  ~0.3-1.5 s device time on the 3090 Ti; fresh-process wall (mmap + verify +
  PTX-cache hit + 235 MB H2D ≈ 0.1 s) ≈ 1-2 s — beating both the Serenity
  3.270 s and the current 5.25 s bridge
  (H3_NATURAL_LANGUAGE_QUALITY_GATE_2026-08-30.md:87). CPU reference ≈
  30-90 s — fine for gates, not the product path.

## 5. Tool: `difaudiodecode` (mirror difvaedecode.cpp conventions)

```
difaudiodecode --backend cpu|cuda --program audio-bigvgan.difir
  --weight-bundle audio-bigvgan.difbind
  --input final-audio.diftensor            # [2T,32] F32 state rows
  --output-wav audio.wav
  [--output-waveform FILE.diftensor]       # [2,L] F32, for gates
  [--backend-plugin FILE.so] [--verify-shards] [--cache-dir DIR] [--min-free-mib N]
```

Host steps: load rows (`.diftensor`; difh3infer already emits
`--output-audio` = final-audio.diftensor and the safetensors handoff at
src/frontend/h3_latents.cpp:173-177), unpack to `[2,32,T]`, run program
(denorm is in-program), receive `[2,1,L]`, write WAV. New tiny module
`src/support/wav.cpp`: 44-byte RIFF/WAVE PCM header + interleaved int16,
EXACTLY wav.mojo's quantization (clamp to [−1,1], i16 = round(x·32767)) —
re-read wav.mojo:120-150 during implementation for the precise
rounding/negative-clamp expression; byte parity of the header+quantizer is
required for the artifact gate. `difh3media --audio-wav` then muxes unchanged.
The bridge script's replacement drops the Mojo LD_LIBRARY_PATH, the Serenity
binary, and its `latents.safetensors` re-read; `mem_safe_runtime.sh` wrapper
stays.

Frontend builder: new `src/frontend/h3_audio_vae.cpp` —
`build_audio_bigvgan_program(B, T, config)` + name→tensor-id map for the
bundle, plus a `--stages N` truncated-program mode replicating the Mojo
staged seams (pre / stage-i / tail) for parity isolation.

Weight import: extend `difimport` (or a `difweights` mode) with
`--fold-weight-norm`: reads `audio_vae/model.safetensors`, folds g/v in
FLOAT64, multiplies the 254 filters by their ratio (×2), emits folded
safetensors + `.difbind` sealed to the program fingerprint. Must NOT fold
`dec_in_proj` (plain weight) and must use the transposed dim-0 convention for
`ups.*` (both traps documented in audio_decoder.mojo).

## 6. Gate design

Oracle: pinned torch/diffusers (`autoencoder_kl_minimax_h3_audio.py`) run on
the REAL checkpoint — allowed as dev oracle by policy
(FLAME_CPP_RUNTIME_PORT.md §5); fixture generator script lives with the other
exporters in `tools/`. The Mojo binary is the secondary cross-check, not the
oracle (it is itself a port; its gates ran vs the vendor python too).

1. **Opcode unit gates** (CPU): Conv1d vs `torch.nn.functional.conv1d` /
   `conv_transpose1d` fixtures covering {plain, dilated 3/5, grouped,
   depthwise, transposed, replicate-pad, asymmetric pad 5/6, trim}, K∈{1,3,4,7,9,11,12};
   SnakeBeta vs the reference formula. Non-degenerate random data (the
   H=30-silent-zero lesson). Bit-level bars vs a same-order C++ re-execution;
   vs torch: max_abs ≤ 1e-6 F32. Negative verifier tests per rule (fail-closed
   convention).
2. **Import gate**: folded-weight tensors vs a torch
   `weight_norm`-removed dump — relative ≤ 1e-7 (float64 fold); shape census
   == the 914-tensor header inventory.
3. **Stage parity** (CPU exec, then CUDA-vs-CPU): boundaries exactly as the
   Mojo oracle staged them — after pre (dec_in_proj+conv_pre), after each of
   the 7 stages, after tail — vs torch on the accepted `[584,32]` rows.
   Record all six metrics (cosine, rel-L2, max-abs, norm ratio, nonfinite,
   bit mismatches). Expected F32-vs-F32 ~1e-6 rel; bar: per-stage rel-L2
   ≤ 5e-5, cosine ≥ 0.9999999, zero nonfinite. Per-stage isolation matters:
   stages differ (rates/kernels), "works at stage 0" ⇏ stage 2.
4. **Accepted-artifact final gate**: decode the recorded audio rows (sha
   20a45cdd…) and compare with the accepted `audio.wav`
   (artifacts/h3-quality-natural-language-2026-08-30/compiler/full_exact_bf16/).
   Honest bar reasoning: our F32 waveform vs the Mojo GPU's differs only by
   sum-order noise (Mojo device-vs-host was max_abs 5.7e-6); int16 LSB =
   3.05e-5, so most samples quantize identically and the rest flip at most
   1 LSB. Gate: header byte-identical; per-sample int16 |Δ| ≤ 1; differing
   samples ≤ 1%; SNR vs recorded wav ≥ 60 dB; zero nonfinite floats before
   quantization. Byte-identity of the whole file is NOT the bar (different
   executor, different reduction order) — claiming it would be dishonest;
   near-parity + the torch-anchored stage gates are the correctness argument.
5. **Perf gate**: fresh-process wall via `mem_safe_runtime.sh` + `time -v`,
   same discipline as the quality-gate table; bar: ≤ 3.27 s (beat Serenity),
   report vs the 5.25 s bridge. VRAM sampled peak recorded.
6. **E2e**: swap the tool into the decode+mux script, produce media/video.mp4,
   run the existing mux; document in a new `docs/H3_AUDIO_DECODE_GATE_<date>.md`
   and update RUNTIME_MODULES.md / FLAME_CPP_RUNTIME_PORT.md removal ledger.

## 7. Effort estimate (chunks, each with its verification)

| # | Chunk | Files | Effort | Verified by |
|---|---|---|---|---|
| 1 | Opcode spec: ir.hpp enum+names, AttrKeys, verifier rules, codec passthrough | ir.hpp, src/ir/verify.cpp, opcode_name | 0.5 d | negative verifier tests |
| 2 | CPU reference Conv1d + SnakeBeta | src/runtime/cpu_executor.cpp | 1 d | gate 1 (torch fixtures) |
| 3 | Weight-norm fold importer + bundle | tools/difimport.cpp or difweights.cpp | 0.5-1 d | gate 2 |
| 4 | Frontend builder + staged/truncated programs | src/frontend/h3_audio_vae.cpp (new) | 1-1.5 d | difc verify + op/tensor census vs §3 counts |
| 5 | CPU stage parity vs torch oracle | tools/ fixture exporter (dev python) | 1 d | gate 3 (CPU) |
| 6 | CUDA emitters (emit_conv1d incl. transposed/replicate/trim; emit_snake_beta) | src/compiler/compiler.cpp | 1-2 d | gate 1 on CUDA + gate 3 CUDA-vs-CPU |
| 7 | difaudiodecode tool + wav.cpp + artifact/perf/e2e gates + docs | tools/difaudiodecode.cpp (new), src/support/wav.cpp | 1 d | gates 4-6 |
| 8 | (optional) OpenCL conformance for the 2 opcodes | backends/opencl/opencl_backend.cpp | 0.5-1 d | conformance suite |

Total ≈ 6.5-8 focused days — consistent with the ledger's "L" rating. Chunk 8
is an IR-owner decision: RUNTIME_MODULES.md claims OpenCL covers the "full
current op set"; landing new opcodes without it either breaks that claim or
needs an explicit recorded gap.

## 8. Risk register (all sourced)

- Transposed weight layout `[C_in, C_out/g, K]` and weight-norm over the
  INPUT-channel dim for ups.* (audio_decoder.mojo:167,448-450).
- float64 fold accumulation (audio_decoder.mojo:64-76, measured defect class).
- `dec_in_proj` unfolded / `conv_post` bias-less — break unconditional loops.
- Log-space snake params; ÷3 average; clamp-in-model — three "plausible
  output" traps explicitly documented in the sources.
- Asymmetric even-kernel downsample pad (5/6) — encode as explicit
  PadLeft/PadRight, never derived inside the emitter.
- Do not inherit the device port's no-filter-flip shortcut; real transposed
  conv only.
- (H) NVRTC TU size ~570 kernels — mitigation: entrypoint dedupe.
- W8A8-nondeterminism (gate-w1r/) is unrelated: this path is plain F32, but
  keep it off the W8A8 route regardless.

## 9. Decision points for the integrator

1. Opcode numbers/attr-key numbers final assignment (IR owner).
2. Rank-3 `[B,C,L]` contract with B=2 baked (recommended — one run for stereo,
   matches diffusers decode) vs rank-2 run-twice.
3. OpenCL now vs recorded gap (chunk 8).
4. In-program denorm-as-depthwise-conv (recommended) vs host-side denorm in
   the tool.

## Integrator decisions (2026-08-31)

1. Opcode/attr-key numbering: DEFERRED until the w2-backward branch (which is
   actively appending opcodes) merges; the IR owner assigns the next free
   numbers then. Nothing in this plan depends on specific values.
2. Rank-3 [B,C,L] with B=2 baked: ACCEPTED.
3. OpenCL: implement the two opcodes for CUDA+CPU first and RECORD the OpenCL
   gap explicitly in RUNTIME_MODULES.md / STATUS docs (the "full current op
   set" OpenCL claim must carry the exception until chunk 8 lands).
4. In-program denorm as depthwise K=1 conv: ACCEPTED (self-contained program).
