# Native BigVGAN audio decode gate — 2026-08-31

## Verdict

The MiniMax-H3 audio decoder runs natively through DiffIR and the C++ CUDA
runtime. Decoding the accepted run's real audio latent rows reproduces the
accepted `audio.wav` to within one int16 LSB on 99.92% of samples at
86.87 dB SNR, in 0.63 s — 5.2x faster than the Serenity Mojo decoder
(3.270 s) and 8.3x faster than the retained bridge (5.25 s). This removes
the Mojo runtime from the output half of the H3 pipeline
(docs/FLAME_CPP_RUNTIME_PORT.md §5, removal-order step 4).

Implementation plan and source-of-truth analysis: docs/BIGVGAN_DECODE_PLAN.md.

## Program

| Property | Value |
|---|---|
| Fingerprint | `759f16c98cbe07fe27ad78e71ad0109dc93d7b5ca94f93682ab6ee71cd00262b` |
| Operations | 603 (391 `Conv1d`, 127 `SnakeBeta`, 85 misc) — matches the plan's derivation exactly |
| Geometry | rank-3 `[B=2, C, L]`, stereo as batch; 292 latent frames -> 233,600 samples/channel, 32 kHz |
| Weights | folded bundle, 788 bindings, `0904085bfcba6aefe7ba275f80b636dee8f010f4655c2a9bb49cfdfed139cc75` |
| Denormalization | in-program depthwise K=1 convolution over baked `config.json` constants |

## Weight-norm fold (gate 2)

`AUDIO_FOLD PASS input_census=914 output_census=779 folded=135
filters_expanded=254 passthrough=390` — float64 sum-of-squares fold,
`dec_in_proj` passthrough, `conv_post` bias-less, 254 shared Kaiser filters
materialized depthwise with the x2 ratio folded into the 127 upsample
filters only. Bit-exact against a float64 torch reference (779/779).

## Opcode gate (gates 1, CPU + CUDA)

`AUDIO_OPCODE_GATE PASS backends=cpu cuda cases=14 mode=enforce` — the
Conv1d variant matrix (plain, dilated, grouped, depthwise, strided,
transposed, replicate pad, asymmetric pad, trim; K in {1,3,4,7,9,11,12})
plus SnakeBeta, against PyTorch fixtures under bars frozen from measurement
(max_abs 6e-6, rel_l2 8e-7, cos 0.9999999).

## Stage parity (gate 3, CPU vs the pinned torch oracle, real rows)

Nine boundaries (pre, seven upsample stages, tail), all six metrics, zero
nonfinite: worst relative L2 1.97e-5 at stage 3, cosine >= 0.9999999998,
norm ratio within 1 +/- 1.9e-7 — inside the plan's frozen bars
(rel_l2 <= 5e-5, cos >= 0.9999999) with >2.5x margin.

## Artifact gate (gate 4, CUDA, accepted rows vs accepted audio.wav)

```
AUDIO_DECODE PASS backend=cuda-nvrtc frames=292 samples_per_channel=233600
             channels=2 nonfinite=0
AUDIO_WAV_GATE PASS header_identical=True samples=467200 max_int16_delta=1
             differing=370 differing_fraction=0.000792 snr_db=86.87
             bars={header:identical,|d|<=1,frac<=1%,snr>=60dB}
```

Whole-file byte identity is deliberately NOT the bar: a different executor
and reduction order cannot reproduce another implementation's F32 summation
order. The correctness argument is the torch-anchored stage parity above
plus this int16-level agreement (every differing sample differs by exactly
one LSB).

## Perf gate (gate 5)

`AUDIO_PERF wall=0:00.63 max_rss_kib=498688 bar=3.27s bridge=5.25s` —
fresh process under `scripts/mem_safe_runtime.sh`, host RSS 487 MiB. The
separate-process decode boundary is retained deliberately (the 24 GiB
policy); this measurement is the whole process, not a kernel-only number.

## Not claimed

VAE *encode*; OpenCL execution of the two new opcodes (recorded gap —
the backend carries fail-closed arms); swapping the accepted artifact's
decode script (left untouched by design); any claim about audio quality
beyond sample-level agreement with the accepted artifact.
