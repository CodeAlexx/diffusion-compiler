# Diffusion Compiler usage

Every tool lives in `tools/` and builds into the CMake build directory
(`build/` below). Run any tool without arguments for its current usage; the
forms here are transcribed from that output. Binary `.difir`, `.diftensor`,
`.difbind`, `.difplan`, and `.diftrain` files are canonical. CLI text is an
operations surface, not a second IR format.

Conventions shared by most tools:

- `--backend cpu|cuda` selects the executor.
- `--cache-dir DIR` holds the PTX cache and persisted cuBLASLt heuristics.
- `--min-free-mib N` is the CUDA pressure guard; preparation fails closed
  below it.
- `--weight-bundle FILE.difbind` / `--bundle FILE.difbind` binds a sealed
  checkpoint; `--verify-shards` re-digests every shard on load.
- `ID=FILE` inputs may be `FILE.diftensor` or `SHARD.safetensors::TENSOR`.
- `--json` prints the versioned `diffusion-compiler-telemetry` document;
  `--report FILE` writes it.

## Compile and inspect

### difc

Build, edit, verify, and fingerprint DiffIR programs.

```text
difc make-rms OUT.difir ROWS COLS BLOCK
difc make-linear-blend OUT.difir ROWS COLS f32|bf16|f16
difc make-flow-euler-trajectory OUT.difir ROWS COLS STEPS f32|bf16|f16
difc make-patchify3d OUT.difir B C T H W PT PH PW f32|bf16|f16
difc make-unpatchify3d OUT.difir B C T H W PT PH PW f32|bf16|f16
difc make-row-pack OUT.difir SEQ TEXT VIDEO AUDIO WIDTH f32|bf16|f16
difc make-h3-attention OUT.difir S HEADS DIM ROTARY BLOCK
difc make-h3-block OUT.difir S HIDDEN HEADS DIM FFN ROTARY BLOCK
difc make-h3-block-bf16 OUT.difir S HIDDEN HEADS DIM FFN ROTARY BLOCK
difc make-h3-block-raw-bf16 OUT.difir S HIDDEN HEADS DIM FFN ROTARY BLOCK resident|streamed
difc make-h3-stack-bf16 OUT.difir S HIDDEN HEADS DIM FFN ROTARY LAYERS BLOCK resident|streamed
difc make-h3-transformer-bf16 OUT.difir S HIDDEN HEADS DIM FFN ROTARY LAYERS TABLES TIME_EMBED BLOCK resident|streamed [split|packed] [generated|cudnn]
difc make-h3-token-refiner-bf16 OUT.difir S HIDDEN HEADS DIM FFN LAYERS BLOCK resident|streamed
difc make-h3-denoiser OUT.difir VIDEO_TOKENS AUDIO_TOKENS TEXT_TOKENS TIMESTEP_TABLES resident|streamed generated|cudnn [LAYERS REFINER_LAYERS]
difc make-h3-video-vae OUT.difir LATENT_T LATENT_H LATENT_W LAYERS resident|streamed generated|cudnn [BATCH]
difc make-h3-video-encoder OUT.difir FRAMES HEIGHT WIDTH resident|streamed
difc make-mlp-training OUT.difir ROWS INPUT_WIDTH HIDDEN_WIDTH OUTPUT_WIDTH [LR BETA1 BETA2 EPS WEIGHT_DECAY] [f32|bf16]
difc make-rectified-flow-training OUT.difir ROWS LATENT_WIDTH TIMESTEP_WIDTH HIDDEN_WIDTH ACCUMULATION_STEPS [LR BETA1 BETA2 EPS WEIGHT_DECAY]
difc set-linear-math IN.difir OUT.difir strict|tf32|direct-int5
difc set-attention-implementation IN.difir OUT.difir generated|cudnn
difc set-elementwise-fusion IN.difir OUT.difir on|off
difc set-constant-residency IN.difir OUT.difir resident|streamed
difc expose-tensors IN.difir OUT.difir ID [ID ...]
difc verify FILE.difir
difc fingerprint FILE.difir
```

`make-*` emits a verified program for a primitive, block, stack, denoiser,
VAE, encoder, or training graph. `set-*` rewrites one schedule attribute.
`expose-tensors` marks internal tensors as observable outputs. `verify` runs
structural verification; `fingerprint` prints the program identity.

### difinspect

```text
difinspect FILE.difir [--prefetch-distance N] [--resident-plan-mib N] [--fixed-runtime-mib N]
           [--resident-order first|largest] [--alias-reshapes] [--assignments] [--json]
difinspect FILE.difir --source [--provenance FILE.provenance.json] [--bundle FILE.difbind]
           [--plan FILE.difplan] [--trace FILE.jsonl] [--op ID] [--json]
```

The first form prints operations, tensors, and the residency plan for the
given budget. `--source` joins creator provenance (the frontend sidecar,
default `FILE.difir.provenance.json`), checkpoint weight identity from the
bundle, compiler transforms and decisions from the plan, and the backend
implementation a runtime trace actually observed, per operation. Nothing is
inferred from tensor names; an absent link is reported as absent.

### difslice

```text
difslice IN.difir OUT.difir FIRST_OP LAST_OP
```

Cuts an operation range into a standalone program for diagnosis.

### difplan

```text
difplan show PLAN.difplan [--json]
difplan diff A.difplan B.difplan [--json]
difplan residency PLAN.difplan [--program FILE.difir] [--json]
difplan residency --program FILE.difir --budget-mib N [--fixed-runtime-mib N]
        [--order first|largest] [--prefetch-distance N] [--json]
difplan explain tensor ID PLAN.difplan [--program FILE.difir] [--json]
difplan explain op ID PLAN.difplan [--program FILE.difir] [--json]
```

Reads the decisions the compiler recorded while planning: streamed-residency
admission arithmetic, every measured candidate's verdict, the precision
policy, and the target requirements. A subject without a recorded decision is
reported as such, never explained by inference.

### difprobe

```text
difprobe [--backend auto|cpu|cuda] [--device N] [--json] [--reserve-mib N] [--host-budget-mib N]
         [--pinned-budget-mib N] [--workspace-budget-mib N] [--staging-budget-mib N]
```

Prints the static target profile (vendor, architecture, compute capability,
SM count, tensor-core dtypes, driver, runtime, cuBLASLt, cuDNN versions, and
the target fingerprint) and the dynamic runtime budget (free and usable VRAM,
host, pinned, workspace, and staging budgets). The `--*-budget-mib` and
`--reserve-mib` flags change only the budget class, never the target.

### difweights

```text
difweights stats FILE.safetensors|FILE.index.json [--top N] [--json]
difweights inspect-shard FILE.safetensors
difweights inspect-index FILE.index.json
difweights inspect-bundle FILE.difbind
difweights verify-bundle FILE.difbind PROGRAM.difir
difweights rebind-program SEALED.difbind PROGRAM.difir OUT.difbind
difweights subset-bundle SOURCE.difbind PROGRAM.difir OUT.difbind
difweights make-h3-bundle INDEX PROGRAM.difir OUT.difbind
difweights rebind-h3-bundle SEALED.difbind INDEX PROGRAM.difir OUT.difbind
difweights make-h3-token-refiner-bundle INDEX PROGRAM.difir OUT.difbind
difweights rebind-h3-token-refiner-bundle SEALED.difbind INDEX PROGRAM.difir OUT.difbind
difweights check-h3-denoiser-bindings INDEX PROGRAM.difir
difweights make-h3-denoiser-bundle INDEX PROGRAM.difir OUT.difbind
difweights rebind-h3-denoiser-bundle SEALED.difbind INDEX PROGRAM.difir OUT.difbind
difweights make-h3-video-vae-bundle SOURCE.safetensors PROGRAM.difir OUT.safetensors OUT.difbind LATENT_T LATENT_H LATENT_W LAYERS resident|streamed generated|cudnn
difweights seal-h3-video-vae-bundle DERIVED.safetensors SOURCE.safetensors PROGRAM.difir OUT.difbind LATENT_T LATENT_H LATENT_W LAYERS resident|streamed generated|cudnn
difweights reuse-h3-video-vae-bundle SEALED.difbind PROGRAM.difir GEOMETRY.safetensors OUT.difbind LATENT_T LATENT_H LATENT_W LAYERS resident|streamed generated|cudnn
difweights make-h3-video-encoder-bundle SOURCE.safetensors PROGRAM.difir OUT.difbind FRAMES HEIGHT WIDTH resident|streamed
difweights make-krea2-bf16-bundle SOURCE.safetensors PROGRAM.difir DERIVED.safetensors OUT.difbind
difweights make-krea2-text-bf16-bundle SOURCE.safetensors PROGRAM.difir DERIVED.safetensors OUT.difbind
difweights make-krea2-vae-bf16-bundle SOURCE.safetensors PROGRAM.difir DERIVED.safetensors OUT.difbind
```

`stats` reports checkpoint storage statistics (counts, dtypes, bytes, ranks,
repeated shape patterns, largest tensors) without inferring model semantics.
`inspect-*` and `verify-bundle` read shards, indexes, and sealed bundles.
`make-*` seals a checkpoint to a program; `rebind-*` and `reuse-*` re-seal an
existing bundle to a new program or geometry; `subset-bundle` keeps only the
tensors a program binds.

### difcast, difcompare, difimage

```text
difcast INPUT.diftensor OUTPUT.diftensor f32|bf16|f16 [DIM ...]
difcompare REFERENCE ACTUAL [--min-cos N] [--max-rel-l2 N] [--min-norm-ratio N]
           [--max-norm-ratio N] [--max-abs N] [--flatten]
difimage INPUT.diftensor OUTPUT_DIRECTORY
```

`difcast` converts a tensor's dtype (optionally reshaping). `difcompare`
judges one tensor pair against explicit bars; `REFERENCE`/`ACTUAL` may be
`FILE.diftensor` or `FILE.safetensors::TENSOR_NAME`. `difimage` writes a
decoded tensor as BMP files, one per frame, for inspection.

## Run and profile

### difrun

```text
difrun --backend cpu|cuda --program FILE.difir [--backend-plugin FILE.so]
       [--weight-bundle FILE.difbind] [--verify-shards]
       --input ID=FILE [--input ...] --output ID=FILE [--output ...]
       [--warmups N] [--iterations N] [--session-runs N] [--min-free-mib N] [--cache-dir DIR]
       [--trace-ops] [--profile-pipeline] [--map-inputs] [--serial-streaming]
       [--tune-linear OP_ID] [--linear-tune-warmups N] [--linear-tune-iterations N] [--linear-tune-sessions N]
       [--expand-linear-algorithms] [--select-linear-algorithm OP_ID:HEURISTIC_RANK] [--persist-linear-heuristics]
       [--fuse-linear-swiglu OP_ID] [--absorb-linear-bias OP_ID] [--cutlass-linear OP_ID:SCHEDULE_ID]
       [--resident-streamed-constant TENSOR_ID] [--cudnn-attention-heuristic a|b|fallback|autotune]
       [--streamed-keep-pages] [--streamed-staging-buffers N] [--streamed-prefetch-depth N]
       [--streamed-stage-threads N] [--streamed-pinned-budget-mib N] [--pinned-io]
       [--h3-w8a8-cache FILE.safetensors] [--h3-w8a8-layer N] [--h3-w8a8-resident-layers N]
       [--h3-convrot-int8-checkpoint FILE.safetensors] [--h3-convrot-int8-layer N] [--h3-convrot-int8-resident-layers N]
       [--h3-int8-mlp-chunk-rows N] [--h3-int8-cublaslt] [--h3-int8-cublaslt-rank N] [--h3-int8-cublaslt-tune]
       [--h3-int8-cutlass-scaled-fc1] [--h3-int8-cutlass-scaled-all] [--h3-int8-compact-adaln]
       [--h3-groupwise-cache FILE.safetensors] [--h3-groupwise-layer N]
       [--h3-modulation-cache FILE.safetensors] [--h3-modulation-input FILE.diftensor] [--h3-modulation-layer N]
       [--h3-modulation-source-index FILE.index.json] [--h3-modulation-steps N] [--h3-modulation-total-layers N]
       [--h3-ck-attention-dso FILE.so] [--h3-int8-attention-layers N] [--w8a8-tail-copy-stream]
```

Prepares and executes one program with explicit inputs and outputs, timing
`--iterations` runs after `--warmups`. `--trace-ops` prints per-operation
timings; `--profile-pipeline` reports staging and pipeline phases. The
`--tune-linear`, `--select-linear-algorithm`, `--fuse-linear-swiglu`,
`--absorb-linear-bias`, and `--cutlass-linear` flags apply explicit backend
execution policy to named operations without changing DiffIR semantics. The
`--h3-*` flags select the H3 precision routes described under
[difh3infer](#difh3infer).

### diftrace

```text
diftrace program --backend cpu|cuda --program FILE.difir [--weight-bundle FILE.difbind]
         [--input ID=FILE ...] [--warmups N] [--iterations N] [--nvtx]
         [--profile-pipeline] [--trace-ops] [--json] [--report FILE]
diftrace recipe RECIPE.json --workdir DIR [--build DIR] [--prompt-file FILE] [--set VAR=VALUE ...]
         [--nvtx] [--no-ffprobe] [--stage-cache DIR] [--json] [--report FILE]
diftrace merge TRACE.jsonl [--json]
```

`program` runs one program with attributed tracing and reports where wait
goes: GEMM, attention, convolution, generated kernels, H2D/D2H/D2D bytes,
staging, waits, synchronization, layout, allocation, each tied to the DiffIR
operation that submitted it. `recipe` runs a benchmark recipe with
`DIF_TRACE_FILE` set per stage and attributes the complete wall to stages and
each stage to those classes. `merge` folds an existing trace file into one
report. A prepared execution that runs many times emits one document per run;
only the first carries `execution.preparation_reported = true`, so one-time
preparation is counted once. A timing carries a `plan` label when a fused
backend plan executed at that operation's slot. Run the same command under
`nsys` with `--nvtx` for Nsight Systems correlation.

## Optimize and tune

### difopt

```text
difopt [program source] [bindings] [options]

program source (exactly one):
  --program FILE               verified DiffIR program to optimize
  --h3-denoiser                build the MiniMax-H3 denoiser frontend
H3 geometry (with --h3-denoiser):
  --video-tokens N --audio-tokens N --text-tokens N --timestep-tables N
  --hidden N --heads N --head-dim N --ffn N --rotary N
  --layers N --refiner-layers N --block-size N
  --video-input-dim N --audio-input-dim N --text-input-dim N
  --time-input-dim N --time-hidden-dim N --time-embed-dim N
  --attention-implementation 1|2
  --streamed-constants
bindings:
  --weight-bundle FILE.difbind  sealed checkpoint bindings
  --verify-shards               re-digest every bundle shard on load
  --bind ID=FILE [--bind ...]   bind inputs and constants from tensors
  --reference ID=FILE           trusted source output for the gate
  --synthetic-bindings SEED     deterministic experiment fixture
search:
  --objective latency|memory    default latency
  --backend cpu|cuda            default cpu
  --precision-policy NAME       plan compatibility policy identity
  --warmups N --iterations N    measurement shape
  --min-free-mib N              CUDA pressure guard
  --beam N --depth N --max-candidates N --margin F
  --latency-tolerance F         memory objective latency ceiling
  --no-structural --no-schedule --no-numeric --no-memory
  --formats LIST                bounded physical-format competition
  --formats-table [--json]      print every format's legality/availability and exit
  --arithmetic-folding          fold arithmetic constants (backend rounding)
  --blocks 64,128,256 --prefetch 1,2,4 --quant-bits 4,5 --quant-groups 64
acceptance bars (trusted; the search cannot change them):
  --max-abs F --min-cos F --min-norm-ratio F --max-norm-ratio F
  --max-rel-l2 F --memory-budget-mib N
outputs:
  --plan FILE --journal FILE --out FILE --db FILE
replay:
  --replay FILE                 rebuild a recorded plan and verify it
  --replay-global-strategy FILE apply each transform to all legal sites
```

`difopt` searches structural, schedule, numeric, and memory candidates and
accepts each one in a fixed order: verify, execute, numerical gate, memory,
then timing. `--formats` takes a comma-separated list from `fp32`, `bf16`,
`fp16`, `fp8-e4m3`, `int8-convrot`, `int4-group`, `int5-group`,
`squareq-w8`, `squareq-w4`, `squareq-nvfp4`; only formats legal on the
probed target and implemented as search candidates produce transforms, and
every requested format is recorded in the plan as a `physical-format`
decision with the capability facts that admitted or excluded it. Plans
serialize with the compiler revision, target fingerprint, budget class,
precision policy, and minimum VRAM/workspace; a target-bound `--replay` fails
closed when any of those conditions changes.

### diftune

```text
diftune --program FILE --db FILE --input ID=FILE [--input ...] [--reference ID=FILE ...]
        [--blocks keep|64,128,256,512] [--linear-math strict,tf32]
        [--warmups N] [--iterations N] [--min-free-mib N]
        [--max-abs N] [--min-cos N] [--min-norm-ratio N] [--max-norm-ratio N] [--json] [--report FILE]
```

Measures block-size and Linear-math candidates for one program against a
reference. Order per candidate: verify, execute, numerical admission, timing;
only admitted candidates rank on time. Results are recorded in `--db`.

### difquant

```text
difquant plan IN.difir BITS GROUP
difquant int4 IN.difir IN.difbind OUT.difir OUT.safetensors OUT.difbind GROUP [--skip-source-digest]
difquant int4-outlier IN.difir IN.difbind OUT.difir OUT.safetensors OUT.difbind GROUP [--skip-source-digest]
difquant int5 IN.difir IN.difbind OUT.difir OUT.safetensors OUT.difbind GROUP [--skip-source-digest]
difquant int5-awq IN.difir IN.difbind OUT.difir OUT.safetensors OUT.difbind GROUP CALIBRATION_DIR ADALN_INPUT
         [--alpha N] [--tensor-alpha ID=N] [--direct-linear] [--skip-source-digest]
```

`plan` rewrites the program for the given bit width and group size and
prints the planned versus naive memory bytes (`LOWBIT_PLAN`) without writing
anything. The other commands rewrite the program and seal a quantized bundle
for group-wise INT4, INT4 with outliers, INT5, or AWQ-calibrated INT5.

### difschedule

```text
difschedule make-exponential-shifted SIGMAS.diftensor TIMESTEPS.diftensor POINTS SHIFT
difschedule make-explicit SIGMAS.diftensor SIGMA SIGMA [SIGMA ...]
```

Writes authenticated flow schedules as tensors for the samplers.

## Benchmark and trace

### difbench

```text
difbench run RECIPE.json --workdir DIR [--build DIR] [--prompt-file FILE] [--set VAR=VALUE ...]
         [--repeat N] [--cooldown-seconds S] [--drop-file-cache] [--digest-model-files]
         [--no-ffprobe] [--gpu-lock FILE] [--label TEXT] [--stage-cache DIR] [--json] [--report FILE]
difbench show RECIPE.json [--workdir DIR] [--build DIR] [--prompt-file FILE] [--set VAR=VALUE ...] [--json]
difbench inspect OUTPUT.png|OUTPUT.mp4 [--no-ffprobe] [--json]
```

`run` is the canonical literal-prompt-to-saved-output boundary. The timer
starts before the first stage process is created and stops when the last
stage exits; the saved output is then verified natively (PNG) or with
ffprobe (MP4). Each repetition uses `DIR/run-N`. Defaults: one run, 10 s
cooldown, ffprobe when installed. The tool records dependency-ordered process
stages with per-stage rusage, page-cache residency of the declared model
files (cold/warm/mixed), NVML peak VRAM and power under the configured cap,
and the comparator from the recipe. Total complete wall is the only
acceptance metric; stage timings are diagnostics. Keep repetitions few: the
GPU power cap is deliberate. `show` resolves and prints the recipe without
running it. `inspect` verifies a saved artifact on its own.

### Recipe format

A recipe is JSON with `"kind": "diffusion-compiler-benchmark-recipe"` and
`"version": 1`. The two committed recipes in `perf/recipes/` are validation
fixtures for the frozen H3 FL2VA and Krea 2 Turbo chains, not part of the
tool.

```json
{
  "kind": "diffusion-compiler-benchmark-recipe",
  "version": 1,
  "name": "krea2-turbo-bf16-1024",
  "output_kind": "image",
  "description": "...",
  "workload": {"model_family": "...", "task": "...", "geometry": "...", "sampler": "...",
               "precision_class": "...", "quality_status": "..."},
  "comparator": {"name": "...", "wall_seconds": 59.14, "target_ratio": 2.0},
  "variables": {"DC_ART": "${recipe_dir}/../../artifacts/...", "CACHE": "${recipe_dir}/../../.difcache"},
  "prompt": {"file": "${DC_COND}/prompt.txt"},
  "inputs": [{"name": "creator_chain_fixture", "path": "${DC_ART}/....safetensors"}],
  "model_files": ["${DC_ART}/krea2-turbo-prepared-bf16.difbind"],
  "required_files": ["${DC_COND}/krea2-qwen3vl-4b.difir"],
  "required_tools": ["ffmpeg", "ffprobe"],
  "stages": [
    {"name": "tokenize", "after": [], "argv": ["${build}/diftokenize", "--processor", "..."]},
    {"name": "condition", "after": ["tokenize"],
     "cache": {"key": ["${prompt_file}"], "outputs": ["${workdir}/taps.safetensors"]},
     "argv": ["${build}/difcondition", "run", "--krea2", "..."]}
  ],
  "output": "${workdir}/image.png"
}
```

- `workload`, `comparator`: descriptive; `comparator.wall_seconds` and
  `target_ratio` are reported next to the measured wall.
- `variables`: expanded in every other string. Built-ins are `${recipe_dir}`,
  `${build}`, `${workdir}`, `${prompt_file}`, `${output}`, and
  `${input:NAME}` for each entry in `inputs`. `--set VAR=VALUE` overrides a
  variable; `--prompt-file` overrides `prompt.file`.
- `model_files`: files whose page-cache residency is sampled before the run
  and, with `--digest-model-files`, digested. `required_files` and
  `required_tools` are preflight checks; a missing one blocks the run
  (exit status 2).
- `stages`: each has `name`, `argv`, and `after` (dependency names; a stage
  without `after` follows the previous stage). Every stage is a fresh
  process. `output_kind` is `image` or `video` and `output` names the
  artifact to verify.

### Stage cache

A stage may declare `"cache": {"key": [files...], "outputs": [files...]}`.
With `--stage-cache DIR` (on `difbench run` and `diftrace recipe`) a stage
whose key is already in `DIR` has its outputs restored and its process never
started. The key includes the stage's argv and effective child environment
(inherited values, overridden by recipe/stage values and then callback values),
with the work directory normalized, plus
every key file's path, size, and mtime outside the work directory and its
SHA-256 up to 64 MiB. Environment values enter the digest only; they are not
written to the cache manifest. Version 2 keys invalidate older entries that
did not include the environment. Every stage record carries `cache_status` (`disabled`,
`none`, `miss`, `hit`, `store-failed`) and `cache_key`, and the run
conditions list the hits, so a cached wall is never mistaken for a cold one.
Without `--stage-cache` the declarations are inert. The H3 recipe caches its
presentation, vision, and conditioner stages on the prompt, keyframes,
program files, and conditioner ConvRot cache.

## Bisect, quality, regress

### difbisect

```text
difbisect pairs --native FILE.safetensors --oracle FILE.safetensors --order NAME[,NAME...] [bars] [--json] [--report FILE]
difbisect manifest MANIFEST.json [bars] [--json] [--report FILE]
difbisect validate-oracle MANIFEST.json [--json]
difbisect revert-check --repo DIR --commit SHA (--build-dir DIR | --no-build) [--targets T,...] [--keep] [--json] [--report FILE] -- GATE [ARG...]
difbisect program --backend cpu|cuda --program FILE.difir [--weight-bundle FILE.difbind] [--input ID=FILE ...]
          --oracle FILE.safetensors --map TENSOR_ID=NAME [--map ...] [--oracle-manifest FILE.json]
          [bars] [--json] [--report FILE]
bars: [--min-cos F] [--max-rel-l2 F] [--min-norm-ratio F] [--max-norm-ratio F] [--max-abs F]
      (defaults 0.999, 0.02, 0.98, 1.02, unbounded)
```

Finds the first divergence between native captures and an oracle fixture.
`pairs` compares same-named boundaries in the order given. `manifest` takes
an explicit ordered boundary list with native/oracle tensor specs
(`FILE.diftensor` or `FILE.safetensors::NAME`). `program` executes the
program, captures the mapped tensors at their producers, and compares them in
program order. The report names the last boundary that passed and the first
that failed, lists the uncaptured operations between them as an unobserved
span, and never asserts a divergence at a boundary nobody observed.
`validate-oracle` rejects any oracle boundary tensor that is non-finite or
constant, since a degenerate reference cannot convict. Oracle manifests are
written by `scripts/oracle_fixture_manifest.py`.

### difquality

```text
difquality CANDIDATE [--reference FILE] [--kind image|video|audio]
           [--min-psnr DB] [--min-ssim F] [--min-snr DB] [--frames N] [--no-ffmpeg]
           [--reviewer NAME --review accept|reject [--review-note TEXT]] [--json] [--report FILE]
```

Generic image, video, and audio gate. It checks decodability and sanity
(constant image, silent audio) and, against a reference, PSNR, an
8x8-window SSIM, SNR, and for MP4s ffmpeg-sampled frame and audio-track
comparisons. Verdicts: `FAIL` when the artifact is missing, undecodable, of
the wrong kind, geometrically mismatched, constant or silent, below a numeric
bar, or rejected by a recorded review; `PASS` only when numeric admission
holds (or sanity holds with no reference) and a human review is recorded as
accept; otherwise `MANUAL REVIEW REQUIRED`. Defaults: PSNR 30 dB, SSIM 0.90,
SNR 20 dB, 8 sampled frames. Scalar metrics never replace perceptual
inspection.

### difregress

```text
difregress run SUITE.json --tier smoke|full|model NAME [--build DIR] [--workdir DIR]
           [--baseline FILE] [--samples N] [--json] [--report FILE]
difregress record SUITE.json --tier smoke|full|model NAME --baseline FILE [--build DIR] [--workdir DIR] [--samples N] [--json]
difregress show SUITE.json [--tier ...] [--json]
```

Runs strict correctness checks (exit status, JSON assertions) and
noise-aware performance checks against a recorded baseline. Check verdicts:
`PASS`; `FAIL` (exit status or JSON assertion); `BLOCKED` (a declared
blocked exit status, or the program cannot start); `REGRESSED` (median above
the baseline median by more than `max(tolerance, baseline noise)`). Tier
verdict: `FAIL` > `REGRESSED` > `BLOCKED` > `PASS`. Exit status: 0 PASS,
1 FAIL/REGRESSED, 3 BLOCKED. `record` writes a baseline; `show` lists the
checks.

### Regression suite format

`perf/regress/suite.json` is this repository's suite. Its smoke tier runs the
target, telemetry, bisect, and format test binaries, checks `difprobe --json`
and `difopt --formats-table`, and runs `diftrace program` on a committed
64x128 RMSNorm fixture (`perf/regress/fixtures/`) gating
`run_launch_telemetry.host_stream_synchronizes` and `device_mem_allocs`
against zero-tolerance baselines. Its model tiers wrap the two difbench
recipes.

```json
{
  "kind": "diffusion-compiler-regression-suite",
  "version": 1,
  "name": "diffusion-compiler",
  "variables": {"REPO": "${suite_dir}/../.."},
  "checks": [
    {"name": "target-tests", "tier": "smoke", "argv": ["${build}/dif_target_tests"]},
    {"name": "probe-json", "tier": "smoke", "argv": ["${build}/difprobe", "--json"],
     "expect_json": {"kind": "device-probe"}},
    {"name": "formats-table-cuda", "tier": "smoke",
     "argv": ["${build}/difopt", "--formats-table", "--backend", "cuda", "--json"],
     "expect_json": {"kind": "physical-formats"},
     "performance": {"metric": "wall_seconds", "samples": 3, "tolerance": 0.25}},
    {"name": "rms-run-host-stream-syncs", "tier": "smoke",
     "argv": ["${build}/diftrace", "program", "--backend", "cuda", "--program", "${REPO}/perf/regress/fixtures/rms-64x128.difir", "..."],
     "expect_json": {"kind": "trace"},
     "performance": {"metric": "json:run_launch_telemetry.host_stream_synchronizes", "samples": 1, "tolerance": 0.0}},
    {"name": "krea2-prompt-to-png", "tier": "model", "model": "krea2",
     "argv": ["${build}/difbench", "run", "${REPO}/perf/recipes/krea2-turbo-bf16-1024.json",
              "--workdir", "${workdir}", "--json", "--cooldown-seconds", "0"],
     "blocked_exit": [2],
     "expect_json": {"status": "completed"},
     "performance": {"metric": "json:summary.complete_wall_seconds.minimum", "samples": 1, "tolerance": 0.05}}
  ]
}
```

- `tier`: `smoke` or `model`; a `model` check also names its `model` and is
  selected with `--tier model NAME`. `--tier full` runs everything.
- `argv`: the process to run; `${build}`, `${workdir}`, `${suite_dir}`, and
  suite `variables` expand.
- `expect_json`: key/value assertions on the process's JSON output.
- `blocked_exit`: exit statuses that mean `BLOCKED` rather than `FAIL`.
- `performance`: `metric` is `wall_seconds` or `json:PATH` into the output
  document, `samples` is how many runs feed the median, and `tolerance` is
  the fractional regression allowance (0.0 means any increase regresses).

A baseline file (`perf/regress/baselines.json`, written by `difregress
record`) carries the telemetry schema head, `"kind": "regression-baselines"`,
provenance (compiler version, git revision, timestamp), the suite name, and
per-check `samples`, `median`, `min`, `max`, `recorded_at`, and `revision`.

## Model tools

### MiniMax-H3 FL2VA

The complete chain, in order, is transcribed in
`perf/recipes/h3-fl2va-convrot-int8-ck.json`: `difh3vision inputs` and `run`,
`difh3encode` for each keyframe, `difcondition run`, `difh3noise`,
`difh3state`, `difh3infer`, `difvaedecode`, `difaudiodecode`, and ffmpeg
muxing (or `difh3media`).

`revert-check` suspects the harness before convicting a commit: it runs the
gate on a detached worktree of HEAD (expected to fail), reverts the commit
onto it, rebuilds with the given build directory's `DIF_*` configuration
(or `--no-build`), and runs the gate again. `{repo}` and `{build}` in the
gate expand to the worktree and its build directory. Verdicts: `CONFIRMED`
(exit 0) when HEAD fails and HEAD minus the commit passes; `NOT_ISOLATED`
(exit 1) when both fail; `HEAD_PASSES` (exit 1) when the premise is wrong;
`BLOCKED` (exit 3) on a revert conflict, a build failure, or a gate that
cannot start. The worktree is removed unless `--keep` is given.

#### diftokenize

```text
diftokenize --processor <dir> | (--tokenizer-json <f> [--tokenizer-config <f>])
            --prompt-file <path> | --prompt <text> | --battery <json>
            [--strip-trailing-newline] [--ids-out <path>] [--diftensor-out <path>]
            [--krea2-inputs-out <path>] [--flux2-inputs-out <path>] [--quiet]
```

Native tokenizer for the Qwen3-VL and Qwen3 conditioners. `--krea2-inputs-out`
and `--flux2-inputs-out` write the conditioner input bundle for those
frontends; `--battery` runs a JSON parity battery.

#### difh3vision

```text
difh3vision program --grid-h N --grid-w N --output V.difir [--grid-t N] [--trace]
difh3vision bundle --checkpoint TEXT_ENCODER_DIR --program V.difir --grid-h N --grid-w N --output V.difbind [--grid-t N] [--trace]
difh3vision inputs --checkpoint TEXT_ENCODER_DIR --processor PROCESSOR_DIR --image FIRST.png [--image LAST.png]
            --prompt-file PROMPT --grid-h N --grid-w N --output INPUTS.safetensors [--grid-t N]
            [--strip-trailing-newline] [--ids-out IDS.diftensor] [--tags-out TAGS.diftensor]
difh3vision run --program V.difir --bundle V.difbind --inputs INPUTS.safetensors --grid-h N --grid-w N
            --output VISION.safetensors [--grid-t N] [--backend cuda|cpu] [--trace] [--cache-dir DIR] [--min-free-mib N]
            [--resident-streamed] [--no-overlap-streaming] [--profile-pipeline] [--warmups N]
            [--attention generated|cudnn] [--deterministic-linear] [--verify-repeat]
            [--cudnn-attention-heuristic a|b|fallback|autotune|deterministic]
            [--capture-tensor ID --capture-dir DIR] [--report FILE.json]
difh3vision combine --vision-input FIRST.safetensors --vision-input LAST.safetensors --output BOTH.safetensors
```

Qwen3-VL vision tower. `inputs` builds the presentation (pixel patches,
token ids, and token tags) from the keyframes and prompt; `run` produces the
vision embeddings; `combine` merges two single-image runs.

#### difcondition

```text
difcondition program --checkpoint DIR --sequence N --output FILE.difir [--layers N] [--attention 1|2] [--krea2]
difcondition bundle --checkpoint DIR --program FILE.difir --output FILE.difbind [--sealed-bundle FILE.difbind]
difcondition run --program FILE.difir --bundle FILE.difbind --ids FILE.diftensor --output FILE.diftensor
             [--backend cuda|cpu] [--cache-dir DIR] [--min-free-mib N]
             [--convrot-int8-checkpoint FILE] [--convrot-int8-linear-count N] [--convrot-int8-weight-only-quality]
             [--streamed-stage-threads N]
difcondition run --program FILE.difir --bundle FILE.difbind --vision-inputs PRESENTATION.safetensors
             --vision-outputs VISION.safetensors --vision-tokens N --output FILE.diftensor [--cache-dir DIR] [--min-free-mib N]
difcondition run --krea2 --program FILE.difir --bundle FILE.difbind --inputs creator.safetensors --output native.safetensors
             [--cache-dir DIR] [--min-free-mib N] [--convrot-int8-checkpoint FILE] [--convrot-int8-linear-count N]
             [--convrot-int8-weight-only-quality] [--streamed-stage-threads N]
```

Qwen3-VL text/multimodal conditioner. `program` and `bundle` build and seal
it for a sequence length; `run` produces the conditioning tensor from token
ids, from a presentation plus vision outputs, or (with `--krea2`) from the
Krea 2 tokenizer bundle. Add `--flux2` for the FLUX.2 [klein] 9B Qwen3-8B
frontend, using `--inputs` from `diftokenize --flux2-inputs-out`. The
`--convrot-int8-*` flags lower the first N eligible Linears through a
generic ConvRot INT8 cache; `--convrot-int8-weight-only-quality` keeps
activations in BF16.

#### difh3convrot

```text
difh3convrot OFFICIAL.index.json OUT.safetensors [--layers N]
             [--convrot-scale-chunk N | --groupwise-quality] [--quality-groups QKV OUT FC1 FC2] [--quality-scale-f32]
difconvrot --program P.difir --bundle B.difbind --output OUT.safetensors
difconvrot --program P.difir --bundle B.difbind --rebind-cache CACHE.safetensors
```

Builds the ConvRot INT8 cache: projection weights rotated offline in 256-wide
Hadamard groups with per-output-channel scales. The first form derives it
from the official H3 transformer index; the `--program`/`--bundle` forms
derive or rebind a generic cache for any program's Linears.

#### difmodcache

```text
difmodcache --checkpoint-index INDEX.json (--video-sigmas F.diftensor --audio-sigmas F.diftensor | --schedule-points N)
            [--steps N] --output FILE.safetensors [--condition-video-floor F32] [--condition-audio-timestep F32]
            [--engine cuda|cpu] [--cublas LIB.so] [--source-index-string STR] [--verify-against FILE.safetensors]
```

Precomputes the per-block AdaLN modulation cache for a schedule so the
denoiser does not re-run the timestep MLP each evaluation.

#### difh3layout, difh3noise, difh3state

```text
difh3layout t2va OUT_DIR TEXT_TAGS.diftensor FRAMES HEIGHT WIDTH AUDIO_LATENTS PATCH_T PATCH_H PATCH_W
            VIDEO_T AUDIO_T CONDITION_VIDEO_T CONDITION_AUDIO_T [first|last ...]
difh3layout ref2va OUT_DIR TEXT_TAGS.diftensor FRAMES HEIGHT WIDTH AUDIO_LATENTS PATCH_T PATCH_H PATCH_W
            VIDEO_T AUDIO_T CONDITION_VIDEO_T CONDITION_AUDIO_T KIND:T:H:W:A [...]
difh3noise --seed U64 --output FILE.diftensor [--rng serenity|torch-cpu] [--layout flat|h3-video|h3-audio]
           [--rows N --cols N] [--latent-frames N --latent-height N --latent-width N] [--audio-latents N]
           [--skip-normal-values N] [--verify-against FILE.diftensor]
difh3state --condition CLEAN.diftensor [--condition CLEAN2] --condition-noise NOISE.diftensor [--condition-noise NOISE2]
           --target-noise NOISE.diftensor --condition-timestep F32 --output STATE.diftensor
```

`difh3layout` writes the packed row layout tables for text-to-video/audio or
reference-conditioned generation into `OUT_DIR`: position ids, token tags,
per-modality indices and maps, timestep and AdaLN indices, and timesteps. `difh3noise` generates the initial noise
with either the torch-parity CPU generator or the GPU generator, in the H3
video or audio layout. `difh3state` assembles the initial denoiser state from
keyframe latents, their condition noise, the target noise, and the condition
timestep.

#### difh3encode

```text
difh3encode --backend cpu|cuda --program TILE.difir --weight-bundle FILE.difbind --image FILE.png
            --pixels-id ID --moments-id ID --output-moments FILE.diftensor --output-latent FILE.diftensor
            --output-rows FILE.diftensor [--backend-plugin FILE.so] [--verify-shards]
            [--tile-size N] [--tile-overlap N] [--posterior-seed N] [--cache-dir DIR] [--min-free-mib N]
```

Encodes a keyframe PNG through the tiled video encoder into moments, a
sampled latent, and packed rows.

#### difh3infer

```text
difh3infer --backend cpu|cuda --sampler euler|res_multistep
           --denoiser-program FILE.difir --denoiser-bundle FILE.difbind
           (--text-tags FILE.diftensor | --all-text-tokens N) --text FILE.diftensor --video FILE.diftensor --audio FILE.diftensor
           (--simple-steps N | --schedule-points N | --video-sigmas FILE.diftensor --audio-sigmas FILE.diftensor)
           --latent-t N --latent-h N --latent-w N --audio-latents N
           [--keyframes none|first|last|first-last | --reference-geometry KIND:T:H:W:A ...]
           --output-latent FILE.diftensor [--output-video-rows FILE.diftensor] --output-audio FILE.diftensor
           [--output-audio-latent FILE.diftensor] [--output-handoff latents.safetensors]
           [--h3-w8a8-cache FILE.safetensors --h3-w8a8-resident-layers N
            | --h3-convrot-int8-checkpoint FILE.safetensors
              [--h3-convrot-int8-layers N | --h3-convrot-int8-attention-layers N --h3-convrot-int8-mlp-layers N]
              --h3-convrot-int8-resident-layers N [--h3-convrot-bf16-audio-rows]
            | --h3-groupwise-cache FILE.safetensors --h3-groupwise-layers N]
           [--h3-int8-mlp-chunk-rows N] [--h3-int8-cublaslt --h3-int8-cublaslt-rank N --h3-int8-cublaslt-tune]
           [--h3-int8-cutlass-scaled-fc1] [--h3-int8-cutlass-scaled-all | --h3-int8-convrot-scale-chunk N]
           [--h3-int8-compact-adaln] [--h3-cache-text-refiner]
           [--resident-streamed-constant TENSOR_ID ...] [--cudnn-attention-heuristic a|b|fallback|autotune]
           [--h3-modulation-cache FILE.safetensors --h3-modulation-source-index FILE.index.json [--h3-modulation-steps N]]
           [(--h3-ck-attention-dso FILE.so | --h3-owned-attention [--h3-owned-attention-center-k]) --h3-int8-attention-first-layer N --h3-int8-attention-layers N] [--h3-int8-attention-first-step N]
           [--denoise-only | --vae-program FILE.difir --vae-bundle FILE.difbind --output-raw FILE.diftensor --output-decoded FILE.diftensor]
           [--first-eval-input-dir DIR] [--capture-denoiser-dir DIR --capture-denoiser-tensor ID ...]
           [--max-evaluations N] [--patch-h N] [--patch-w N] [--backend-plugin FILE.so] [--verify-shards] [--profile-pipeline]
           [--streamed-keep-pages] [--pipelined-resident-upload | --lazy-resident-upload] [--h3-resident-readahead-mib N] [--h3-resident-mapped-copy] [--keep-resident-host-pages]
           [--streamed-staging-buffers N] [--streamed-prefetch-depth N] [--streamed-stage-threads N]
           [--streamed-pinned-budget-mib N] [--pinned-io] [--cache-dir DIR] [--min-free-mib N] [--serve SOCKET | --connect SOCKET]
```

Runs the H3 sampler over the prepared denoiser and writes the video latent,
video rows, audio rows, and audio latent, optionally decoding video in the
same process. Precision routes, one of:

- `--h3-convrot-int8-checkpoint`: ConvRot INT8 projections for the first N
  blocks (`--h3-convrot-int8-layers`, or attention and MLP counts
  separately), with `--h3-convrot-int8-resident-layers` blocks kept on the
  device. `--h3-int8-cutlass-scaled-all` runs QKV, output, FC1, and FC2
  through the CUTLASS INT8 scaled-epilogue GEMMs; `--h3-int8-cutlass-scaled-fc1`
  does FC1 only; `--h3-int8-convrot-scale-chunk N` enables per-K-chunk
  activation scales instead. `--h3-int8-cublaslt` uses shape-prepared
  cuBLASLt IMMA plans. `--h3-int8-mlp-chunk-rows` sets the row tile for
  direct INT8 MLP projections. `--h3-int8-compact-adaln` consumes compact
  AdaLN tables inside the ConvRot epilogues. `--h3-convrot-bf16-audio-rows`
  recomputes the audio rows in BF16.
- `--h3-w8a8-cache`: the row-scaled W8A8 MLP route with a resident block
  prefix.
- `--h3-groupwise-cache`: groupwise INT8 weight-only projections.

`--h3-ck-attention-dso` replaces attention in `--h3-int8-attention-layers`
blocks from `--h3-int8-attention-first-layer` with the project-owned INT8
attention library; blocks outside the range stay on exact cuDNN.
`--h3-modulation-cache` uses a `difmodcache` file instead of running the
timestep MLP. `--h3-cache-text-refiner` runs the text refiner once and
reuses it across evaluations.
`--max-evaluations` truncates the sampler for diagnostics; `--capture-denoiser-*`
copies named tensors during the first evaluation for `difbisect`.

Persistent denoiser: `--serve SOCKET` runs the request on the command line,
keeps the prepared denoiser (plans, scratch, resident weights) alive, prints
`H3_SERVE READY socket=... pid=...`, and then serves further requests on the
Unix socket. `difh3infer --connect SOCKET <the same difh3infer arguments>`
sends one request and prints its output; only inputs, schedule, geometry
(validated against the prepared program), outputs, and diagnostics may differ
between requests. A request whose prepare-affecting flags differ from the
prepared ones is refused; a single `--shutdown` token stops the server. Served
outputs are bit-identical to a fresh process; the receipt carries
`persistent_reuse=1`. The server holds the denoiser's resident bytes for its
lifetime (22.6 GB on the H3 recipe), so other GPU stages cannot run alongside
it on a 24 GB device.

#### difvaedecode

```text
difvaedecode --backend cpu|cuda --program TILE.difir --weight-bundle FILE.difbind --input LATENT.diftensor
             --latent-id ID --raw-id ID (--output-raw FILE.diftensor --output-decoded FILE.diftensor | --output-rgb FILE.rgb)
             [--backend-plugin FILE.so] [--verify-shards] [--clip-length N] [--token-drop N]
             [--tile-size N] [--tile-overlap N] [--warmups N] [--iterations N] [--min-free-mib N] [--cache-dir DIR]
             [--convrot-int8-checkpoint FILE] [--convrot-int8-linear-count N] [--convrot-int8-resident]
             [--deterministic-conv] [--trace-ops] [--workers N] [--tile-digests FILE] [--digest-tensor ID ...]
```

Tiled video VAE decode from the video latent to decoded frames
(`--output-decoded`) or a raw RGB stream ready for ffmpeg (`--output-rgb`).
`--clip-length` and `--token-drop` set the temporal clip and dropped leading
tokens. The `--convrot-int8-*` flags lower the decoder's Linears through a
generic ConvRot cache kept resident with `--convrot-int8-resident`.
`--trace-ops` aggregates per-operation device timings over every tile
execution and prints them by opcode as `H3_VAE_DECODE_OP` lines.
`--tile-digests FILE` appends the SHA-256 of every raw tile output exactly as
the executor returned it, before stitching, so GPU-side and host-side
differences can be told apart; `--digest-tensor ID` (repeatable) captures an
intermediate tensor into the same file. `--workers N` caps the host stitching
threads. `--deterministic-conv` restricts cuDNN convolutions to algorithms
reported deterministic and fails when none fits the workspace limit.

#### difaudiodecode

```text
difaudiodecode --backend cpu|cuda --program FILE.difir --weight-bundle FILE.difbind --input ROWS.diftensor
               --output-wav FILE.wav [--output-waveform FILE.diftensor] [--sample-rate N] [--verify-shards]
               [--cache-dir DIR] [--min-free-mib N]
```

Audio VAE decode from audio rows to a WAV file.

#### difh3media

```text
difh3media --video DECODED.diftensor --audio-wav audio.wav --output-dir DIR --input-fps N
           [--output-fps N] [--ffmpeg FILE] [--encoder h264_nvenc|libx264]
```

Muxes decoded frames and audio into an MP4 through FFmpeg. The default
encoder is portable `libx264`; use `h264_nvenc` only when the selected FFmpeg
build exposes it.

#### difimport

```text
difimport --conditioning FILE.safetensors --initial-state FILE.safetensors --output-dir DIR --manifest FILE.json
          --prompt FILE --tokenizer FILE --tokenizer-config FILE --encoder-index FILE --checkpoint-index FILE
          --source-encoder FILE --width N --height N --frames N --fps N --steps N --seed N
```

Byte-preserving container conversion of recorded H3 conditioner and
initial-state SafeTensors into DiffTensor files plus a hashing manifest. It
does not tokenize, encode, sample noise, cast, or quantize; payload bytes are
copied verbatim.

### Krea 2

The chain is transcribed in `perf/recipes/krea2-turbo-bf16-1024.json`:
`diftokenize --krea2-inputs-out`, `difcondition run --krea2`, `difkrea2text`,
`difkrea2sample`, `difkrea2vae`. These tools print only an error on a bad
command line; their option names are listed here from the source.

```text
difkrea2block --checkpoint RAW.safetensors --fixture creator.safetensors --output native.safetensors
              --report report.json --diffir block.diffir [--block N] [--sequence-override FILE.safetensors::NAME] [--final-only]
difkrea2denoise --checkpoint FILE --fixture FILE --output FILE --report FILE --diffir FILE
difkrea2text (--checkpoint FILE | --bundle FILE) --output FILE --report FILE --diffir FILE
             (--fixture FILE | --taps FILE --mask-inputs FILE) [--no-compare] [--capture-first-block]
difkrea2sample (--checkpoint FILE | --bundle FILE) --positive-conditioning FILE --positive-tokenizer FILE
               --initial-fixture FILE --reference FILE --output FILE --report FILE --diffir FILE
               --steps N --guidance F [--mu F] [--seed N] [--stop-after N] [--text-token-cap N]
               [--negative-conditioning FILE --negative-tokenizer FILE]
               [--resident-plan-mib N] [--resident-order first|largest] [--whole-plan] [--alias-reshapes]
               [--lazy-resident-upload | --pipelined-resident-upload]
               [--streamed-prefetch-depth N] [--streamed-staging-buffers N] [--streamed-stage-threads N]
               [--tune-linear ID] [--linear-tune-warmups N] [--linear-tune-iterations N] [--linear-tune-sessions N]
               [--expand-linear-algorithms] [--select-linear-algorithm ID:RANK] [--persist-linear-heuristics]
               [--cutlass-linear ID:SCHEDULE] [--parallel-linears IDS] [--fuse-parallel-linears IDS]
               [--fuse-parallel-swiglu IDS] [--lower-fused-swiglu] [--profile-pipeline] [--cache-dir DIR]
difkrea2vae (--checkpoint FILE | --bundle FILE) --output FILE --report FILE --diffir FILE
            (--fixture FILE | --sampler FILE --reference FILE --config FILE --png FILE)
```

`difkrea2block` and `difkrea2denoise` are the source-faithful block and
denoiser gates against creator fixtures. `difkrea2text` runs TextFusion from
the conditioner taps and tokenizer mask. `difkrea2sample` runs the 8-step
Euler denoiser (CFG when `--guidance` is nonzero and negative inputs are
given); `--whole-plan` requires an explicit `--resident-plan-mib`.
`difkrea2vae` decodes with the Qwen-Image VAE in tile mode (fixture) or full
mode (sampler output, reference, config, PNG).

### FLUX.2 [klein]

```text
difflux2block --checkpoint MODEL.safetensors --fixture creator.safetensors --output native.safetensors --report report.json
              --diffir block.diffir --image-tokens N --text-tokens N [--batch-size N] [--block N]
              [--single | --transformer --double-depth N --single-depth N | --vae --latent-height N --latent-width N]
              [--attention cudnn|flash] [--select-linear-algorithm OP_ID:HEURISTIC_RANK] [--capture-boundary NAME ...]
              [--expand-linear-algorithms]
difflux2sample --model-dir DIR --vae-checkpoint ae.safetensors --prompt TEXT --output image.png --report report.json
               [--state-output state.safetensors] [--transformer-checkpoint model.safetensors]
               [--positive-conditioning positive.safetensors --negative-conditioning empty.safetensors]
               [--initial-latent initial.safetensors] [--initial-latent-tensor NAME]
               [--seed N] [--steps N] [--start-step N] [--stop-after N] [--capture-every N] [--width N --height N] [--guidance F]
               [--transformer-attention cudnn|flash] [--cudnn-attention-heuristic a|b|fallback|autotune]
               [--streamed-prefetch-depth N --streamed-staging-buffers N --streamed-stage-threads N] [--streamed-release-pages]
               [--resident-plan-mib N] [--resident-order largest|first] [--lazy-resident-upload]
               [--tune-linear-operation ID ...] [--expand-linear-algorithms] [--select-linear-algorithm OP_ID:HEURISTIC_RANK ...]
               [--persist-linear-heuristics] [--linear-tuning-warmups N --linear-tuning-iterations N --linear-tuning-sessions N]
               [--w8a8-single-linear1-blocks N | --w8a8-single-linear1-block ID ...]
               [--w8a8-convrot | --w8a8-f32-convrot | --w8a8-f32-signed-convrot | --w8a8-f32-signed-convrot-4096
                | --w8a8-signed-convrot | --w8a8-signed-convrot-4096]
               [--w8a8-weight-equalization] [--w8a8-mse-weight-scale]
               [--w8a8-activation-residual2] [--w8a8-activation-residual2-single-linear1]
               [--w8a8-activation-residual2-single-linear2] [--w8a8-activation-residual2-double]
               [--w8a8-activation-clip-ratio F] [--w8a8-activation-clip-single-linear1 F]
               [--w8a8-activation-clip-single-linear2 F] [--w8a8-activation-clip-double F]
               [--w8a8-activation-clip-switch-step N --w8a8-activation-clip-after-ratio F]
               [--w8a8-single-mlp-blocks N | --w8a8-single-mlp-block ID ...] [--w8a8-single-mlp-h256-convrot]
               [--w8a8-single-qk-blocks N | --w8a8-single-qk-block ID ...]
               [--w8a8-single-linear2-blocks N | --w8a8-single-linear2-block ID ...]
               [--w8a8-double-image-mlp-block ID ...] [--w8a8-double-mlp-block ID ...] [--w8a8-double-image-block ID ...]
               [--w8a8-double-text-block ID ...] [--w8a8-double-block ID ...]
               [--fp8-single-linear1-blocks N | --fp8-single-linear1-block ID ...]
               [--fp8-single-mlp-blocks N | --fp8-single-mlp-block ID ...]
               [--fp8-single-linear2-blocks N | --fp8-single-linear2-block ID ...]
               [--fp8-double-image-mlp-block ID ...] [--fp8-double-image-block ID ...] [--fp8-double-text-block ID ...]
               [--fp8-double-block ID ...] [--fp8-row-scaled]
               [--int8-weight-only-all-linears | --int8-weight-only-row-scaled-all-linears]
               [--int8-weight-only-group-size 16|32|64] [--int8-weight-only-exclude CHECKPOINT_NAME ...]
               [--int8-weight-only-group32 CHECKPOINT_NAME ...] [--cache-dir DIR] [--profile-pipeline]
```

`difflux2block` is the source-faithful gate for one double block, single
block, the whole transformer, or the VAE against a creator fixture.
`difflux2sample` is the complete native prompt-to-PNG path: it tokenizes and
conditions from `--model-dir`, runs the generalized-time Euler schedule, and
decodes with the F32 VAE. The `--w8a8-*` flags select the ConvRot W8A8
variant and which block linears it applies to (counts of leading blocks, or
explicit block ids), with activation clip ratios and residual-2 policies per
linear class. The `--fp8-*` flags select the FP8 (Blackwell) linears the
same way. The `--int8-weight-only-*` flags apply group-wise INT8 weight-only
storage to all remaining linears with per-tensor exclusions or group-32
overrides. `--stop-after`, `--start-step`, `--capture-every`, and
`--state-output` support step-bounded diagnostics and bisecting.

### Training

```text
diftrain run-mlp --backend cpu|cuda --program FILE.difir --features FILE.diftensor --targets FILE.diftensor
         (--w1 FILE --b1 FILE --w2 FILE --b2 FILE | --resume FILE.diftrain) --steps N
         --checkpoint OUT.diftrain --losses OUT.diftensor [--prediction OUT.diftensor] [--gradients-dir DIR]
         [--backend-plugin FILE.so] [--cache-dir DIR] [--min-free-mib N]
diftrain run-flow --backend cpu|cuda --program FILE.difir --fixture DIR [--resume FILE.diftrain] --steps N
         --checkpoint OUT.diftrain --losses OUT.diftensor [--predictions-dir DIR] [--gradients-dir DIR]
         [--backend-plugin FILE.so] [--cache-dir DIR] [--min-free-mib N]
diftrain make-lora OUT.difir ROWS LATENT_WIDTH TIMESTEP_WIDTH HIDDEN_WIDTH RANK ALPHA [LR BETA1 BETA2 EPS WEIGHT_DECAY]
diftrain run-lora --backend cpu|cuda --program FILE.difir --fixture DIR [--resume FILE.diftrain | --init-seed N] --steps N
         --checkpoint OUT.diftrain --losses OUT.diftensor [--prediction OUT.diftensor] [--gradients-dir DIR]
         [--backend-plugin FILE.so] [--cache-dir DIR] [--min-free-mib N]
diftrain export-lora --program FILE.difir --checkpoint FILE.diftrain --output OUT.safetensors
diftrain inspect FILE.diftrain
diftrain export FILE.diftrain OUT_DIR
difdittrain --backend cpu|cuda --fixture DIR --output DIR [--steps N] [--cache-dir DIR]
            [--checkpoint OUT.diftrain] [--resume IN.diftrain] [--export-adapters OUT.safetensors]
```

`diftrain` runs the compiled forward/backward/AdamW graphs built by `difc
make-mlp-training`, `difc make-rectified-flow-training`, and `diftrain
make-lora`, writing losses, gradients, and a checksummed `.diftrain`
checkpoint that `--resume` continues. `export-lora` writes the trained
adapters as safetensors. `difdittrain` trains a DiT block fixture; fixtures
whose `config.json` sets `lora_rank` accept the checkpoint, resume, and
adapter-export flags.

### Operator fixture gates

```text
difditops CASE --fixture DIR --output DIR --backend cpu|cuda
difaudioops CASE --fixture DIR --output DIR --backend cpu|cuda
```

`difditops` cases: `rms_norm`, `rms_norm_modulate_weighted`,
`rms_norm_modulate_plain`, `swiglu_gatefirst`, `swiglu_valuefirst`,
`residual_gate`, `layer_norm`, `qk_norm_rope_fulltable`,
`qk_norm_rope_halftable`, `attention_full`, `attention_causal`,
`attention_gqa<H>x<KvH>[_causal]`; semantic attributes come from
`FIXTURE/attrs.json`. `difaudioops` cases: `conv_k1_pointwise`,
`conv_k3_dilated3`, `conv_k7_plain`, `conv_k11_dilated5`, `conv_k9_grouped3`,
`conv_k3_stride2_nopad`, `conv_k12_depthwise_replicate_asym`,
`conv_k4_transposed_stride2`, `conv_k9_transposed_stride5`,
`conv_k12_transposed_depthwise_replicate`, `conv_k7_transposed_grouped`,
`snake_beta_c7`, `snake_beta_c1`, `snake_beta_c64`. Each runs one operator on
an exported fixture and writes the native output for comparison.

## Environment variables

| Variable | Effect |
|---|---|
| `DIF_TRACE_FILE=path` | Any tool appends one `runtime-trace` JSON document per prepared execution to `path`. `diftrace recipe` sets it per stage; `diftrace merge` reads the file. |
| `DIF_NVTX=1` | Push NVTX ranges (`dif::prepare`, `dif::run`, `op<id> <opcode>`) for Nsight Systems correlation. Requires the NVTX headers at build time; silently unavailable otherwise. |

## Runtime knobs that matter

These are execution policy set by the compiler or an explicit flag. The
runtime never picks them heuristically, and any non-default choice enters
`difopt` candidate identity. Outputs are byte-identical across host-paging
choices; only the deterministic-algorithm policies can change bytes, and only
by removing run-to-run variance.

| Knob | Flag | Behavior |
|---|---|---|
| Deterministic Linear | `--deterministic-linear` (`difh3vision run`) | Require a cuBLASLt algorithm with no cross-CTA split-K reduction for ordinary Linears. Fails closed when the heuristics expose none. |
| Direct-IO weight staging | on by default; `--h3-resident-mapped-copy` (`difh3infer`) turns it off | Resident H3 INT8 checkpoint weights and streamed constants whose mapped pages are cold (under 90% resident by `mincore`) are staged into pinned memory with O_DIRECT reads, sixteen MiB chunks, sixteen in flight, bypassing the page cache; warm pages take the mapping copy. Measured on the H3 checkpoint drive (1909 extents): cold first evaluation 18.1 s -> 8.9 s, page-cache paths cap at about 1 GB/s there. After a direct read the same range is re-read in the background with buffered reads so the next process or evaluation finds warm pages (fadvise/readahead advice is a no-op on ext4 with kernel 6.8), gated on the cgroup being able to hold the charge. Reported as `denoiser_resident_direct_read_bytes` / `denoiser_streamed_direct_read_bytes`. |
| Resident read-ahead window | `--h3-resident-readahead-mib N` (`difh3infer`), default 0 | Page-cache read-ahead issued ahead of the resident checkpoint upload in file order. Measured no gain on this host (buffered reads cap at about 1 GB/s); kept as an experiment knob. Reported as `denoiser_resident_readahead_bytes`. |
| cuDNN attention backward | `AttentionBackward` with `Implementation=2` (program attribute) | cuDNN SDPA backward over the program's saved logsumexp (transposed into cuDNN's packed stats layout per call) instead of the generated math kernel: 11x faster at S=1536 (59 vs 655 ms), dK/dV bit-repeatable, dQ not (a few dozen bf16 flips per million on cuDNN's flash backward). `--cudnn-attention-heuristic deterministic` (heuristic 4) requests cuDNN's deterministic algorithms and fails closed when none exists. Gated by `dif_cudnn_attention_backward_tests` against the CPU F32 reference: every element within one bf16 quantum, cosine >= 0.99999. |
| Exact-then-INT8 attention by evaluation | `--h3-int8-attention-first-step N` (`difh3infer`); `RunOptions::h3_int8_attention_hybrid` at prepare, `h3_int8_attention_active` per run | Evaluations before N run exact cuDNN attention, later ones the selected INT8 route, from one prepared denoiser (both plans are built; the INT8 branch fails closed for a run that asks for exact without the hybrid prepared). Measured on the H3 t37 contract against the exact seven-evaluation decode: N=3 worst frame 32.3 dB / SSIM 0.958 / audio SNR 12.9 dB, N=4 37.5 dB / 0.978 / 14.7 dB, versus 24.5 dB / 0.850 / 1.7 dB for INT8 on every evaluation; each exact evaluation costs about 3.5 s more. |
| Owned INT8 attention | `--h3-owned-attention` (`difh3infer`) | In-tree owned H3 dense INT8 attention (sm_86, head dim 128, noncausal BF16 [S,H,128]) bound in place of the attention DSO for the selected transformer layer range; approximate class, reported in the run receipt as `owned_h3_dense_int8_v4_in_tree`. Mutually exclusive with `--h3-ck-attention-dso`; fails closed off sm_86. |
| Owned attention K centering | `--h3-owned-attention-center-k` (`difh3infer`) | Subtracts the per-head K mean over the sequence before INT8 quantization (softmax-invariant; deterministic two-stage mean). Separate receipt identity `owned_h3_dense_int8_v4_in_tree_center_k`; requires `--h3-owned-attention`. |
| Deterministic convolution | `--deterministic-conv` (`difvaedecode`) | Require cuDNN convolution algorithms reported `CUDNN_DETERMINISTIC`. Fails closed when none fits the workspace limit. Motivated by the H3 video decode not being bit-reproducible from an identical latent. |
| Deterministic attention heuristic | `--cudnn-attention-heuristic deterministic` (`difh3vision run`) | Restrict cuDNN SDPA engine discovery to deterministic engines. Elsewhere `a`, `b`, `fallback`, `autotune` select the discovery heuristic only. |
| Resident host page cache | `--keep-resident-host-pages` (`difh3infer`) | Default evicts each resident weight's mapped range with `posix_fadvise(DONTNEED)` after upload, so every fresh process rereads from storage. The flag keeps the clean, reclaimable file-cache pages so a later process stages them from cache. The runtime reads the process's cgroup-v2 memory limit and charge and fails closed to eviction, printing a `RESIDENT_HOST_PAGES` line, when the limit cannot hold the resident bytes. |
| Streamed page release | `--streamed-keep-pages`, `--streamed-release-pages` | Default releases mapped streamed-weight ranges with `madvise(MADV_DONTNEED)` after each staging copy. `--streamed-keep-pages` keeps them populated across runs of one prepared execution (host RSS grows by up to the streamed payload). `--streamed-release-pages` (`difflux2sample`) forces release. |
| Staging threads | `--streamed-stage-threads N` | Worker threads for the mmap-to-pinned staging copy. Default 1 is the historical single-threaded memcpy on the submitting thread; the frozen recipes use 4. |
| Staging ring | `--streamed-staging-buffers N`, `--streamed-prefetch-depth N`, `--streamed-pinned-budget-mib N` | Pinned staging ring size and how many operations ahead the overlapped scheduler prefetches (defaults 2 and 1). Buffers must be at least 2 and at least depth + 1. The ring fails closed above the pinned budget (default 2048 MiB). |
| Resident upload | `--lazy-resident-upload`, `--pipelined-resident-upload` | Lazy: allocate persistent constants at prepare and populate at first use, overlapping the one-time upload with compute. Pipelined: upload through a bounded two-slot pinned pipeline during preparation. Mutually exclusive. |
| Pinned I/O | `--pinned-io` | Stage dynamic inputs and outputs through a pinned bounce buffer sized at prepare, counted against the pinned budget. |
| Residency plan | `--resident-plan-mib N`, `--resident-order first\|largest`, `--fixed-runtime-mib N`, `--prefetch-distance N` | Device budget for resident weights, admission order, fixed runtime reservation, and streamed prefetch distance. `difplan residency` shows the admission arithmetic. |
| Linear heuristics | `--persist-linear-heuristics`, `--expand-linear-algorithms`, `--select-linear-algorithm OP:RANK` | Persist cuBLASLt algorithm selections under the cache directory keyed on the full problem identity (validated on restore, fail open to fresh heuristics); expand the heuristic candidate list; pin one operation to a heuristic rank. |
| Pressure guard | `--min-free-mib N` | Preparation fails closed when free VRAM is below N MiB. |

Build-time and media knobs: `DIF_CUDA_ARCHITECTURES` (default `native`;
`120` for Blackwell), `DIF_ENABLE_CUDA`, `DIF_ENABLE_CUDNN`,
`DIF_ENABLE_OPENCL`. H3 media output defaults to `libx264`; set
`--encoder h264_nvenc` on `difh3media` only when the selected FFmpeg exposes
that encoder.
