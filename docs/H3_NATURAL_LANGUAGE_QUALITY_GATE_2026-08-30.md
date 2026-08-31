# MiniMax-H3 natural-language quality gate

Date: 2026-08-30

## Verdict

The exact-BF16 Diffusion Compiler arm passes the first natural-language visual
quality gate. It produces a genuinely coherent, cinematic 175-frame H3 video
with native dialogue and sound from the same real Qwen3-VL conditioning,
starting states, schedule, geometry, and checkpoint as the mature Serenity
reference. No optimizer search, W8A8, groupwise INT8, Sage attention, CK
attention, approximate step reuse, synthetic embedding, or analytic fixture was
used.

This is a visual-quality pass, not a numerical-parity pass. The compiler video
is good on inspection, but the final latent trajectory differs materially from
Serenity. The first measured internal difference is already present after block
1 in the first denoise evaluation, and accumulation reaches video-latent cosine
0.886187 at the end of evaluation 19. That discrepancy remains open semantic
evidence and must not be hidden by the successful visual result.

## Prompt and conditioner identity

The exact official-contract T2VA prompt is preserved at
`artifacts/h3-quality-natural-language-2026-08-30/prompt.txt`.

- prompt SHA256: `6f6ec2cf781c58bc8630fd93fad1d4205fd313ceee06180217b04810e18f434a`
- tokenizer: checkpoint `processor/tokenizer.json`, no chat template, no added
  special tokens, 439 tokens
- tokenizer SHA256: `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7`
- tokenizer-config SHA256: `a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68`
- conditioner: accepted Serenity Qwen3-VL path
  `/home/alex/mojodiffusion/serenitymojo/models/text_encoder/minimax_h3_conditioning.mojo`,
  hidden layer 50, streamed BF16 checkpoint weights
- text-encoder index SHA256:
  `06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a`
- output: BF16 `[439,5120]`, payload SHA256
  `9b1609bdc8c02365d386844cbeae988aaa66d72847528f0a6d3e5b24b1d89585`
- conditioning SafeTensors SHA256:
  `17b3517ac131f662d26da620fe91b7571c3fb7e4763ac9c371bbfecc71c4a6ee`
- accepted negative-prompt policy: none; one conditional forward

`tools/import_serenity_h3_inputs.py` copies the SafeTensors payload into the
compiler tensor container without arithmetic, synthesis, cast, or quantization.
The import gate reports zero bit mismatches across all conditioning and starting
state elements.

## Matched settings and starting state

- task: T2VA, one continuous shot
- checkpoint: `/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA`
- transformer index SHA256:
  `fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb`
- seed: video 4242, derived audio 4243
- output: 832x480, 175 frames, 24 fps, 7.291667 seconds
- latent geometry: `[1,24,52,30,52]`; video rows `[20280,96]`; audio
  rows `[584,32]`
- schedule: 20 points / 19 evaluations, released data-ward float32 Euler,
  video shift 12, audio shift 3
- denoiser: 50 blocks, streamed checkpoint BF16 projections, exact cuDNN
  attention, exact step cache, exact schedule-wide BF16 AdaLN modulation cache
- modulation-cache SHA256:
  `2f47ac3a8505d17cbb594a10f3bda4ed59ae5558cb942de2692d275cc92697b7`

Starting tensors:

| Tensor | Shape / dtype | Payload SHA256 | Bit mismatches |
|---|---|---|---:|
| text conditioning | `[439,5120]` BF16 | `9b1609bdc8c02365d386844cbeae988aaa66d72847528f0a6d3e5b24b1d89585` | 0 / 2,247,680 |
| video state rows | `[20280,96]` F32 | `d6bcf2b1871bab3fd8480a82e02a6cd35d4587c5ed08ef52f78f93fa8e7ca9a1` | 0 / 1,946,880 |
| audio state rows | `[584,32]` F32 | `20a45cddc717aef9c21cf0bf5b03c53df1194d09473a9da26dbc455258b0c180` | 0 / 18,688 |

The source initial-state SafeTensors SHA256 is
`74cee455dc36514eb675cbf3e3d0ffa12e6fa6cf33c7d15fcadc6a3b4e6de818`.

## Runtime results

Hardware: NVIDIA GeForce RTX 3090 Ti, SM86, 24 GiB.

| Stage | Serenity | Diffusion Compiler |
|---|---:|---:|
| real conditioning | 182.119 s evidence run; 287.896 s resumed reference run | exact imported BF16 payload, no second encoder execution |
| evaluation 1 | 274.535 s with seven boundary saves | 21.328 s in the complete run |
| complete 19-evaluation trajectory | 3,031.291 s across evidence eval 1 plus resumed evals 2-19 | 426.478 s wall, 421.803 s device timeline, 22.446 s/evaluation |
| denoiser sampled VRAM peak | 8,418 MiB | 7,403 MiB |
| video decode | 70.71 s, 22,892 MiB sampled peak | 38.462 s execution / 55.43 s command, 6,422 MiB sampled peak |
| audio decode | 3.270 s | 5.25 s fresh process |
| H264/AAC mux | 0.77 s | 0.963 s |

The first compiler launch caused a host, not GPU, OOM: an uncapped long-lived
terminal cgroup accumulated 40.2 GiB while hashing and mapping checkpoint
shards; `systemd-oomd` killed desktop processes while GPU usage peaked at only
7,769 MiB. All accepted reruns use a fresh
`scripts/mem_safe_runtime.sh` scope with `MemoryMax=24G`, `MemoryHigh=infinity`,
`MemorySwapMax=2G`, and a 16 GiB desktop reserve. The accepted full denoiser
scope peaked at 1,310,810,112 bytes and the VAE scope at 11,872,407,552 bytes;
both reported zero low/high/max/OOM events.

## First-evaluation trajectory

All comparisons contain zero nonfinite elements. Hidden tensors are BF16
`[21303,5376]`; heads and post-Euler states are float32 modality rows.

| Boundary | Cosine | Relative L2 | Max abs | Norm ratio | Bit mismatches |
|---|---:|---:|---:|---:|---:|
| after block 1 | 0.999978711 | 0.006543314 | 768 | 0.999491314 | 48,938,031 |
| after block 3 | 0.999988382 | 0.004820362 | 768 | 0.999994442 | 89,391,217 |
| after block 5 | 0.999988309 | 0.004836311 | 1,024 | 1.000083092 | 93,102,738 |
| after block 10 | 0.999988188 | 0.004862553 | 2,048 | 1.000129172 | 95,346,521 |
| after block 20 | 0.999958794 | 0.011015659 | 65,536 | 0.993719036 | 96,214,411 |
| after block 50 | 0.999488523 | 0.031986984 | 83,968 | 1.000177756 | 106,184,109 |
| video head | 0.999833004 | 0.018343154 | 0.501277 | 1.001416632 | 1,946,878 |
| audio head | 0.999924855 | 0.012264835 | 0.095432 | 0.999548510 | 18,687 |
| video after first Euler update | 0.999999986 | 0.000165331 | 0.002310 | 1.000001325 | 1,946,537 |
| audio after first Euler update | 0.999999972 | 0.000237463 | 0.001735 | 1.000007067 | 18,683 |
| final video state | 0.886187316 | 0.478426059 | 5.037430 | 1.005432660 | 1,946,879 |
| final audio state | 0.958390643 | 0.289195763 | 1.402290 | 1.004724367 | 18,688 |

The first meaningful numerical difference is after block 1. The small first
Euler delta masks much of the head difference at evaluation 1, but repeated
evaluations accumulate it into the final latent discrepancy. Because the final
Compiler media remains visibly good, this run does not trigger the user's
bad-video precision ladder or optimizer work; it does retain the discrepancy
for the next semantic-parity investigation.

## Decoded and visual review

Both arms produced 175-frame H.264/AAC MP4s. Paired decoded RGB metrics are
PSNR 17.3483 dB and SSIM 0.741833. These low paired-image metrics are consistent
with the final latent divergence and show that the two arms are no longer the
same sample, not that either image is intrinsically poor.

Direct inspection of both evenly spaced sheets and the Compiler dialogue-motion
sheet found:

- stable engineer identity, face, off-white jacket, body proportions, and tool
  case;
- natural skin and restrained cinematic exposure with no color/exposure reset;
- coherent spacecraft, ramp, hangar, reflections, steam, and background
  workers;
- a continuous side-truck to rear-tracking move with natural walking progress;
- no frozen or repeated frame, loop, scene reset, identity substitution, or
  severe VAE texture artifact;
- 175 unique consecutive decoded frames; Compiler median consecutive-frame MAD
  7.0469/255 and first-to-last MAD 37.1159/255;
- stereo 32 kHz Compiler audio at -34.46 dBFS RMS and -17.91 dBFS peak, with
  zero clipped samples;
- intelligible dialogue. Whisper-tiny transcribes both arms as “Five minutes,
  let's get our flying,” confusing “her” with “our.” Both place the phrase at
  approximately 0-3 seconds, slightly earlier than the prompt's literal
  midpoint; this is shared generation behavior rather than a Compiler-only
  regression.

## Artifacts and reproduction

Reference:

- `artifacts/h3-quality-natural-language-2026-08-30/serenity/reference_bf16/video.mp4`
  (`2b97157c0cccf02559da5e1f880035e767a3f5c7c87bd2be1b2ca76430363757`)
- `artifacts/h3-quality-natural-language-2026-08-30/serenity/reference_bf16/contact-sheet.png`
- `artifacts/h3-quality-natural-language-2026-08-30/serenity/reference_bf16/contact-sheet-dialogue-motion.png`

Compiler:

- `artifacts/h3-quality-natural-language-2026-08-30/compiler/full_exact_bf16/media/video.mp4`
  (`285d9a6baa5ef94b8381a0f4204f1df75207e05373b0613445743ceecdf8302b`)
- `artifacts/h3-quality-natural-language-2026-08-30/compiler/full_exact_bf16/contact-sheet.png`
- `artifacts/h3-quality-natural-language-2026-08-30/compiler/full_exact_bf16/contact-sheet-dialogue-motion.png`
- `artifacts/h3-quality-natural-language-2026-08-30/compiler/full_exact_bf16/final-sha256.txt`

Reproducible compiler commands are captured in:

- `artifacts/h3-quality-natural-language-2026-08-30/compiler/run_full_exact_bf16.sh`
- `artifacts/h3-quality-natural-language-2026-08-30/compiler/decode_full_exact_bf16.sh`
- `artifacts/h3-quality-natural-language-2026-08-30/compiler/decode_audio_and_mux_full_exact_bf16.sh`
- `artifacts/h3-quality-natural-language-2026-08-30/compiler/first_eval_trace/run_first_eval_trace.sh`

The accepted outputs and generated checkpoint bundles remain ignored artifacts;
they are not repository source.
