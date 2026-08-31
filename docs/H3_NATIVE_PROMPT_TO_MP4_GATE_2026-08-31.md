# MiniMax-H3 native prompt-to-MP4 gate — 2026-08-31

## Verdict

The accepted natural-language H3 generation now runs from **literal prompt
text** with no PyTorch, Python, Serenity, Mojo/MAX, Rust FFI, ComfyUI, or
model-specific worker executing any part of the neural graph. The last
non-native stage — the Qwen3-VL-32B text conditioner — is a DiffIR program
over generic opcodes, executed by the shared C++ runtime.

The produced video preserves the accepted quality gate: the same scene the
prompt describes, at the same fidelity as the accepted artifact, with no
artifacts. It is a different *sample* (the conditioning differs from
Serenity's by ~3.4e-3 relative, and 19 Euler steps amplify that), which is
expected and is not a quality regression — the accepted artifact itself
diverged from its own Serenity reference at final-latent cosine 0.886.

## The chain (every stage native)

| Stage | Tool | Evidence |
|---|---|---|
| prompt text -> 439 token ids | `diftokenize` | ids sha `3e8fe983…`, byte-identical to the accepted encoding |
| ids -> `[439,5120]` BF16 conditioning | `difcondition` | tensor gate below |
| seeds 4242/4243 -> initial states | `difh3noise` | byte-identical to the accepted states |
| conditioning + states -> latents | `difh3infer` | `H3_DENOISE PASS`, 19/19 evaluations |
| video latent -> frames | `difvaedecode` | `[1,3,175,480,832]`, range [0,1] |
| audio rows -> waveform | `difaudiodecode` | native BigVGAN, 233,600 samples/ch |
| frames + wav -> MP4 | `difh3media` + ffmpeg | h264 832x480x175 @24fps + AAC, 7.29 s |

## Conditioner tensor gate

Oracle: transformers' own `Qwen3VLTextModel` on the real checkpoint, GPU
BF16, raw per-depth hidden states (`tools/export_qwen3vl_conditioner_reference.py`).
Durable fixtures: `/home/alex/dif-fixtures/qwen3vl-conditioner-2026-08-31/`.

The oracle's run-to-run spread is EXACTLY ZERO (bit-identical across
processes), so it cannot serve as the noise floor. The bar is instead the
same-framework kernel-choice envelope: swapping only transformers' attention
implementation (eager vs sdpa) with identical weights and inputs.

| Comparison (depth 50, `[439,5120]` BF16) | relative L2 | cosine |
|---|---:|---:|
| **native vs oracle (canonical sdpa)** | **3.449e-03** | **0.99999405** |
| envelope: oracle eager vs oracle sdpa | 3.805e-03 | 0.99999276 |
| oracle vs Serenity's accepted payload | 3.485e-03 | 0.99999393 |
| native vs Serenity's accepted payload | 3.135e-03 | 0.99999509 |
| control: oracle depth 49 vs payload | 8.038e-02 | 0.99679 |

The port sits inside the framework's own kernel-choice envelope, agrees with
Serenity's independent Mojo stack *more closely than the torch oracle does*,
and does not compound with depth (3.357e-03 at depth 1 -> 3.449e-03 at depth
50). Zero nonfinite everywhere. The depth-49 control is 23x worse, which
both confirms the extraction rule (raw residual after 50 layers, before
`model.norm`) and shows the metric discriminates.

Bit-mismatch fraction is deliberately NOT a bar: it is 87.5% between two
correct torch kernels.

## Program and resources

Program `4767e7ae…`: 802 operations — 350 Linear, 50 grouped-query
Attention, 100 QkNormPartialRope, 100 RmsNorm, 50 SiLU, 50 Multiply, 100
Add, 1 GatherRows, 1 RotaryPosition. 551 streamed weights bound directly to
the checkpoint's own 14 shards (index `06c952c5…`, matching the accepted
artifact's recorded encoder index); no derived copy, no Python loader.

Conditioning: 204.2 s wall (prepare 6.2 s), 2.27 GiB VRAM, 5.08 GiB host
RSS, zero OOM events under `scripts/mem_safe_runtime.sh`. Denoise 15:41 for
19 evaluations. Video decode 54.5 s. Audio decode <1 s. Mux 3.1 s.

## Torch-free proof (process level, not source grep)

Test environment: `env -i PYTHONNOUSERSITE=1` — in it,
`python3 -c "import torch"` fails with `ModuleNotFoundError: No module named
'torch'`. Every stage below ran successfully in that same environment, under
`strace -f -e trace=execve`:

| Stage | Binaries exec'd |
|---|---|
| tokenize | `diftokenize`, `env` |
| condition | `difcondition`, `env`, `flock`, `bash` |
| audio decode | `difaudiodecode`, `env`, `flock`, `bash` |
| mux | `difh3media`, `env`, **`ffmpeg`** |

`ffmpeg` is the declared, documented native mux boundary — not a model
runtime. No Python, torch, Mojo, MAX, Serenity, or worker process appears
anywhere.

Link level (`ldd`, vendored-cuDNN build): `diftokenize`, `difcondition`,
`difh3infer`, `difvaedecode`, `difaudiodecode`, `difh3media`, `difh3noise`
all link only CUDA (cuda/cudart/cublas/cublasLt/nvrtc), cuDNN 9, and
glibc/libstdc++ — no libtorch, no libpython, and no path resolving into a
Python site-packages tree.

## Final artifact

`832x480`, 175 frames, 24 fps, 7.291667 s, h264 + AAC. Visual inspection of
a 10-frame contact sheet against the accepted artifact's sheet: the same
female engineer with short dark hair, weathered off-white flight jacket over
a dark utility suit, carrying a compact metal tool case in her right hand,
walking through a night launch hangar with a large spacecraft under rows of
cool white work lights, steam from floor vents, distant technicians, yellow
floor markings, camera trucking right at slow speed — every element the
prompt specifies. Coherent motion, stable identity across frames, no
artifacts.

## Not claimed

Byte-identity with the accepted latents (impossible: the conditioning
legitimately differs from Serenity's within BF16, and the sampler amplifies
it). VAE *encode*. A speed claim for the conditioner against Serenity's
(unmeasured under matched conditions). Padding A/B (gate 6 of the plan) —
the program is built at exact S=439, so the padded path is untested.
Vision/keyframe conditioning: the MRoPE-collapse-to-1-D argument is measured
for text-only input (rebuilt tables match transformers' 3-axis output
exactly, max delta 0.0) but `get_rope_index` was not re-derived for a
vision path.
