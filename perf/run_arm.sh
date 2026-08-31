#!/usr/bin/env bash
# One controlled arm of the H3 streaming experiment. The workload is FIXED to
# the accepted golden inputs; only staging policy flags vary.
#   run_arm.sh LABEL OUTDIR EVALS [extra difh3infer flags...]
set -euo pipefail
LABEL="$1"; OUT="$2"; EVALS="$3"; shift 3
ROOT=/home/alex/dc-perf
BUILD=$ROOT/build
ART=/home/alex/diffusion-compiler/artifacts/h3-quality-natural-language-2026-08-30
CKPT=/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA
MEMSAFE=/home/alex/mojodiffusion/scripts/mem_safe_runtime.sh
D="$OUT/$LABEL"; mkdir -p "$D/cache"
# Host-side counters before/after (major faults + read bytes) for the whole scope.
read_before=$(awk '/^read_bytes/{print $2}' /proc/self/io 2>/dev/null || echo 0)
start=$(date +%s.%N)
MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G DESKTOP_RESERVE=16G \
  "$MEMSAFE" /usr/bin/time -v "$BUILD/difh3infer" --backend cuda \
    --denoiser-program "$ART/compiler/h3-832x480x175-t439-exact-cudnn.difir" \
    --denoiser-bundle  "$ART/compiler/h3-832x480x175-t439-exact-cudnn.difbind" \
    --all-text-tokens 439 \
    --text  "$ART/compiler/inputs/text_conditioning.diftensor" \
    --video "$ART/compiler/inputs/video_state_rows.diftensor" \
    --audio "$ART/compiler/inputs/audio_state_rows.diftensor" \
    --video-sigmas "$ART/compiler/smoke_exact_bf16/first_eval_inputs/video_sigmas.diftensor" \
    --audio-sigmas "$ART/compiler/smoke_exact_bf16/first_eval_inputs/audio_sigmas.diftensor" \
    --latent-t 52 --latent-h 30 --latent-w 52 --audio-latents 292 --keyframes none \
    --output-latent "$D/video-latent.diftensor" --output-audio "$D/audio-rows.diftensor" \
    --output-audio-latent "$D/audio-latent.diftensor" \
    --h3-modulation-cache "$CKPT/serenity_runtime_cache_v1/modcache_steps_20_blocks_50.safetensors" \
    --h3-modulation-source-index "$CKPT/transformer/model.safetensors.index.json" \
    --h3-modulation-steps 20 --denoise-only --profile-pipeline \
    --max-evaluations "$EVALS" \
    --cache-dir "$D/cache" --min-free-mib 4096 "$@" > "$D/run.log" 2>&1
status=$?
end=$(date +%s.%N)
echo "ARM=$LABEL exit=$status wall_s=$(echo "$end $start" | awk '{printf "%.2f", $1-$2}') flags=$*" | tee "$D/summary.txt"
python3 "$ROOT/perf/summarize.py" "$D"
