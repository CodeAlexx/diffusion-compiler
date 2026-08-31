# Phase 3: complete MiniMax-H3 inference path

Date: 2026-08-30

Phase 3 is complete for the compiler's precomputed-conditioning H3 product
boundary. The retained path runs the released 50-block transformer at production
geometry, advances the video and audio states through the released schedule,
executes both final output heads, writes the Serenity-compatible latent handoff,
decodes video with the compiled VAE, decodes audio with the retained Serenity
BigVGAN bridge, and produces the final H.264/AAC media artifact.

This phase did not add an optimizer search, redesign model semantics, or create a
fourth H3 stack. It integrated the already-admitted Serenity W8A8 and CK
implementations around the existing DiffIR graph.

## Source contracts

The pinned Diffusers checkout remains the semantic source of truth:

- The H3 schedule is a float32 `linspace(1, 0)` followed by the exponential
  shift, consecutive deduplication, and `t = 1 - sigma`:
  `/home/alex/minimax_h3_ref/diffusers-src/src/diffusers/schedulers/scheduling_minimax_h3.py:127-170`.
- The scheduler recovers one sigma from the rounded timestep but uses the sigma
  grid for the Euler ratio:
  `/home/alex/minimax_h3_ref/diffusers-src/src/diffusers/schedulers/scheduling_minimax_h3.py:255-282`.
- The final shared AdaLN norm and separate video/audio projections are declared
  at `/home/alex/minimax_h3_ref/diffusers-src/src/diffusers/models/transformers/transformer_minimax_h3.py:519-525`
  and executed before modality row selection at the same file's lines 635-640.
- Video rows use channel-slowest `(C, pt, ph, pw)` packing, while audio rows are
  stereo-major:
  `/home/alex/minimax_h3_ref/diffusers-src/src/diffusers/modular_pipelines/minimax_h3/packing.py:278-328`.
- Only generated video/audio rows are stepped:
  `/home/alex/minimax_h3_ref/diffusers-src/src/diffusers/modular_pipelines/minimax_h3/denoise.py:258-274`.
- The released video and audio latent denormalization and decode boundaries are
  `/home/alex/minimax_h3_ref/diffusers-src/src/diffusers/modular_pipelines/minimax_h3/decoders.py:95-114`
  and lines 183-195, respectively.

Serenity supplied the performance and process-boundary map:

- It precomputes the schedule-wide modulation cache once at
  `/home/alex/mojodiffusion/serenitymojo/pipeline/minimax_h3_t2va.mojo:2910-2937`.
- It loads a strict resident prefix and refills one reusable W8A8 tail block at
  `/home/alex/mojodiffusion/serenitymojo/pipeline/minimax_h3_t2va.mojo:2969-3040`;
  the hot tail's persistent mapping and bounded two-half pinned staging are
  implemented at
  `/home/alex/mojodiffusion/serenitymojo/models/dit/minimax_h3_runtime_cache.mojo:450-570`.
- Each block consumes its cached modulation and the final head consumes the
  final cached modulation at
  `/home/alex/mojodiffusion/serenitymojo/pipeline/minimax_h3_t2va.mojo:1816-1907`
  and lines 1968-1980.
- Audio-only and video-only decode are deliberate process boundaries because
  the decoders must not share a near-full 24 GiB process:
  `/home/alex/mojodiffusion/serenitymojo/pipeline/minimax_h3_t2va.mojo:2578-2633`.
- The released Serenity media handoff writes RGB24 and invokes NVENC/AAC at
  `/home/alex/mojodiffusion/serenitymojo/pipeline/minimax_h3_t2va.mojo:930-943`
  and lines 1121-1161.

## Integrated behavior

The compiler now provides:

- source-faithful float32 H3 sigma construction and data-ward Euler updates;
- dynamic per-evaluation video/audio/conditioning timesteps;
- schedule-wide BF16 block and final-head AdaLN cache selection;
- metadata validation against cache kind, version, schedule size, block count,
  source index path, source size, and source modification time;
- a strict W8A8 resident prefix plus one reusable device tail store and bounded
  two-half pinned host staging;
- W8A8 recognition restricted to creator transformer blocks, excluding the
  context refiner;
- optional CK dispatch restricted to the maximum-sequence transformer
  attentions, excluding the refiner attentions;
- exact video/audio row unpacking and audio latent denormalization;
- a SafeTensors handoff with `video_state_rows` and `audio_state_rows` keys;
- exact-hash VAE bundle sealing that reuses the existing converted checkpoint
  shard and adds only the necessary shape/float32 exception shard;
- sequential temporal/spatial VAE decode with bounded device residency;
- RGB24 validation, NVENC or libx264 mux, AAC audio, overwrite refusal, and the
  `serenity.minimax_h3.result.v1` result schema.

The production path intentionally uses separate denoise, video-decode,
audio-decode, and mux processes. This releases the transformer's 19.41 GB
allocation before VAE/BigVGAN decode and is the safe 24 GiB execution policy,
not an incomplete join.

## Exact and approximate classification

| Stage | Classification | Evidence |
|---|---|---|
| Sigma schedule and Euler update | Exact F32 semantics | 30-point shift-12/shift-3 oracle: max abs 0, relative L2 0, no bit mismatches; pinned trajectory also runs in `dif_tests` |
| Conditioning row selection and final heads | Exact graph semantics | DiffIR retains released heads; schedule cache is the creator-produced BF16 output and is metadata-bound |
| Video/audio row unpack and handoff | Exact | Unit gates plus two complete-run hashes are bit-identical |
| Transformer projections | Approximate | Established Serenity W8A8 route; explicitly reported as `approximate_w8a8_established_h3_gate` |
| Full denoise attention used for the final media | Exact | Program-declared cuDNN; CK count is zero |
| Optional SM86 CK route | Approximate | Explicit `approximate_ck_int8_established_h3_gate`; never labeled exact |
| Compiler video VAE | Approximate to source F32 weights, high parity to Serenity BF16 | Existing FP16 converted shard, exact source/derived hashes, full decoded comparison below |
| Audio decode | Existing Serenity implementation | The compiler handoff is consumed directly by the retained BigVGAN runtime |
| RGB24 and media handoff | Exact structural contract | Finite/range gate, ffprobe metadata, repeat byte hash |

## Matched production check

Hardware and geometry:

- NVIDIA GeForce RTX 3090 Ti, SM86, 24,564 MiB;
- one checkpoint: local MiniMax-H3 FL2VA release;
- text 241 rows, video 8,580 rows, audio 244 rows, total `S = 9,065`;
- 20 schedule points / 19 model evaluations;
- latent `[1,24,22,30,52]` and decoded video `[1,3,73,480,832]`;
- 24 fps, stereo 32 kHz audio;
- exact cuDNN attention for the complete final-media run;
- W8A8 projection quality mode, 42 resident blocks plus 8 streamed blocks;
- 20-step, 50-block BF16 modulation cache.

### Complete compiler path

The final cold-path run recorded:

| Stage | Hot/device work | Stage wall | Device residency / free VRAM |
|---|---:|---:|---:|
| 19-evaluation denoise | 156,703.389 ms device timeline; 5,091.1 ms mean after the cold first evaluation | 217,590.608 ms | 19,411,205,120 B resident; 24,011,800,576 B free before, 4,247,715,840 B free after |
| Scheduler updates | 31.548 ms | included above | host F32 |
| Latent unpack/handoff | 1.096 ms | included above | 3.2 MB handoff |
| Video VAE, 48 sequential tile executions | 37,012.5 ms kernel sum | 44,054.6 ms | 373,473,792 B resident; 23,541,710,848 B free after |
| BigVGAN audio bridge | retained Serenity decoder | 8.07 s cold process wall | 2,867,332 KiB host RSS; no OOM |
| RGB24 + NVENC/AAC mux | 979.956 ms mux | 2.92 s process wall | 686,720 KiB host RSS |

The cold denoise wall includes a 58,532.493 ms initial resident H2D upload and
2,441 major host page faults. The per-evaluation steady-state figures are the
meaningful hot transformer timing. An earlier warm-page complete run took
114,898.381 ms wall with the same final tensors.

The final artifact is 73 H.264 frames at 832x480 and 24 fps plus two-channel AAC
at 32 kHz. `ffprobe` reports 3.041667 s video and 3.040000 s audio. The second
complete run reproduced the video latent, audio rows, denormalized audio latent,
decoded video, WAV, and MP4 byte-for-byte.

### Matched Serenity comparisons

The comparison is decomposed at authenticated tensor boundaries because the
compiler begins with precomputed conditioning embeddings; Phase 3 did not add a
prompt encoder. No unmatched prompt run is presented as a whole-pipeline A/B.

| Boundary | Compiler | Mature Serenity comparator | Parity / speed |
|---|---:|---:|---|
| 50-block W8A8 + CK transformer, same source tensor fixture and SM86 DSO | 3,894.79 ms mean across three sessions | 4,725.674559 ms hot | 48,733,440 BF16 outputs bit-identical; compiler 1.213x faster |
| One-block W8A8 + exact cuDNN, final three-order gate | 95.882 ms mean | 141.852 ms exact BF16 baseline | 1.479x faster; bit-identical to Serenity W8A8 output and inside the source gate |
| Full 73-frame video decode from the same final latent | 44.0546 s | 58.29 s retained low-memory Serenity decoder | compiler 1.323x faster; mean PSNR 51.359480 dB and all-frame SSIM 0.999028 |
| Audio decode | compiler handoff into Serenity BigVGAN | same decoder | direct handoff; 97,600 samples/channel, 3.05 s |
| Media | compiler RGB24/NVENC handoff | Serenity RGB24/NVENC contract | matching geometry, fps, audio, and result schema |

The full 73-frame VAE comparison used the same checkpoint, GPU, final latent,
geometry, tiling policy, and decoded RGB range. Per-channel mean PSNR was
52.079 dB red, 51.334 dB green, and 50.764 dB blue.

The decoded fixture was visually inspected through a contact sheet. It is finite
and temporally stable, but it is an analytic tensor fixture with repeated
diagonal color texture, not a natural-language generation. The report therefore
makes numerical parity and plumbing claims, not a prompt-semantic quality claim.

## Rejected configurations

Rejected measurements are retained as negative evidence, not accepted paths:

- The first all-resident full graph used only a 512 MiB reserve. Its ledger was
  22,164,595,200 bytes required from 23,894,949,888 bytes free and it exhausted
  VRAM during preparation/execution. It wrote no partial output. This
  configuration is not used again.
- A 42-resident/8-tail run without the modulation cache reread about 17.7 GB of
  AdaLN weights per evaluation and took 72,452.9 ms for one evaluation. It is
  semantically valid but not a retained performance path.
- The current Serenity batched-spatial VAE attempted to retain all 12 decoded
  tiles, allocated about 20.5 GB in its MAX cache, and failed a subsequent 32 MB
  allocation after 51 partial frames. Partial output was removed. The retained
  mature comparator is the earlier sequential low-memory decoder, which
  completed all 73 frames.

## Reproducible command surfaces

The production sequence is:

```sh
build/difh3infer --backend cuda ... \
  --schedule-points 20 --all-text-tokens 241 \
  --h3-w8a8-cache resident_w8a8_row_blocks_50.safetensors \
  --h3-w8a8-resident-layers 42 \
  --h3-modulation-cache modcache_steps_20_blocks_50.safetensors \
  --h3-modulation-source-index model.safetensors.index.json \
  --h3-modulation-steps 20 --denoise-only --min-free-mib 4096

build/difvaedecode --backend cuda ... --verify-shards --min-free-mib 4096

# Decode audio from latents.safetensors with the retained Serenity
# decode_audio_only entry point, then:
build/difh3media --video video-decoded.diftensor --audio-wav audio.wav \
  --output-dir media --input-fps 24 --output-fps 24 --encoder h264_nvenc
```

Complete local evidence is under the ignored directory
`artifacts/phase3-h3-complete/`; checkpoint data and generated artifacts are not
committed.
