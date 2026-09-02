#!/usr/bin/env bash
# Frozen MiniMax-H3 FL2VA benchmark boundary:
# literal prompt + first/last PNGs -> saved 832x480x124@24fps H.264/AAC MP4.
#
# Static DiffIR programs, bundles, quantized caches, and the shared CUDA cache
# are prepared-model state and intentionally live outside the timed prompt
# boundary, matching a warm ComfyUI server. Every prompt-dependent operation
# is executed below. Run this script under one external /usr/bin/time process.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 PROMPT.txt FIRST.png LAST.png OUTDIR" >&2
  exit 2
fi

PROMPT="$1"
FIRST="$2"
LAST="$3"
OUT="$4"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${DIF_BUILD:-$ROOT/build}"
CKPT="${DIF_CHECKPOINT:-/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA}"
FIXTURE="$ROOT/artifacts/h3-flref-product-2026-09-01"
NATIVE="$FIXTURE/fl2va-t37-native"
SHARED="$FIXTURE/shared"
CACHE="${DIF_CUDA_CACHE:-$SHARED/cuda-cache}"
CK_ATTENTION="${DIF_CK_ATTENTION_DSO:-/home/alex/Ck-INT8/native/lib/sm86/libck_int8_kernels.so}"
CONVROT="$CKPT/serenity_runtime_cache_v1/native_convrot_h256_official_blocks_50.safetensors"
CONDITIONER_CONVROT="$NATIVE/cache/qwen3vl50-convrot-h256-s1256-v780.safetensors"
MODCACHE="$CKPT/serenity_runtime_cache_v1/native_modcache_keyframe_steps8_blocks50.safetensors"
VAE_CONVROT="$NATIVE/video-vae-convrot-f16/video-vae-convrot-f16.safetensors"
EVALUATIONS="${DIF_MAX_EVALUATIONS:-7}"

[[ "$EVALUATIONS" =~ ^[1-7]$ ]] || {
  echo "DIF_MAX_EVALUATIONS must be an integer from 1 through 7" >&2
  exit 2
}

if [[ -e "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing nonempty output directory: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT" "$OUT/cache"

require_file() {
  [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }
}

for path in \
  "$PROMPT" "$FIRST" "$LAST" "$CK_ATTENTION" "$CONVROT" \
  "$CONDITIONER_CONVROT" "$MODCACHE" "$VAE_CONVROT" \
  "$NATIVE/vision/qwen3vl-vision-t1h30w52.difir" \
  "$NATIVE/vision/qwen3vl-vision-t1h30w52.difbind" \
  "$NATIVE/conditioner/qwen3vl-mm-s1256-v780-l50.difir" \
  "$NATIVE/conditioner/qwen3vl-mm-s1256-v780-l50.difbind" \
  "$NATIVE/denoiser/denoiser-s16880-t37-a207-c1256-t3.difir" \
  "$NATIVE/denoiser/denoiser-s16880-t37-a207-c1256-t3.difbind" \
  "$SHARED/video-encoder/tile-f1-h256-w256.difir" \
  "$SHARED/video-encoder/tile-f1-h256-w256.difbind" \
  "$SHARED/video-vae/tile-t7-h16-w16-l36.difir" \
  "$SHARED/video-vae/tile-t7-h16-w16-l36.difbind" \
  "$SHARED/audio-vae-t207/t207.difir" \
  "$SHARED/audio-vae-t207/t207.difbind"; do
  require_file "$path"
done

for tool in difh3vision difcondition difh3encode difh3noise difh3state \
            difh3infer difvaedecode difaudiodecode; do
  require_file "$BUILD/$tool"
done
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
command -v flock >/dev/null

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
power_limit="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n1)"
[[ "$gpu_name" == "NVIDIA GeForce RTX 3090 Ti" ]] || {
  echo "wrong benchmark GPU: $gpu_name" >&2
  exit 2
}
[[ "$power_limit" == "300.00" ]] || {
  echo "wrong benchmark power limit: ${power_limit} W" >&2
  exit 2
}

exec 9>/tmp/diffusion-compiler-h3-benchmark-gpu.lock
flock -n 9 || { echo "benchmark GPU lock is held" >&2; exit 2; }

STAGES="$OUT/stages.tsv"
run_stage() {
  local name="$1"
  shift
  /usr/bin/time -f "$name\t%e" -o "$STAGES" -a "$@"
}

# Literal prompt and both keyframes -> the exact 1,256-token H3 presentation.
run_stage presentation "$BUILD/difh3vision" inputs \
  --checkpoint "$CKPT/text_encoder" --processor "$CKPT/processor" \
  --image "$FIRST" --image "$LAST" --prompt-file "$PROMPT" \
  --grid-t 1 --grid-h 30 --grid-w 52 --strip-trailing-newline \
  --output "$OUT/presentation.safetensors" \
  --ids-out "$OUT/input-ids.diftensor" --tags-out "$OUT/token-tags.diftensor"
cmp "$OUT/input-ids.diftensor" "$NATIVE/vision/input-ids.diftensor"
cmp "$OUT/token-tags.diftensor" "$NATIVE/vision/token-tags.diftensor"

# Keyframe VAE encodes depend only on the submitted images. The vision encoder
# runs first because overlapping all three GPU processes materially slows it.
# Once vision finishes, run both keyframe encodes beside the conditioner; this
# measured schedule keeps their independent work inside the prompt wall while
# shortening the critical path. It is execution scheduling, not model caching.
encode_keyframe() {
  local which="$1"
  local image="$2"
  run_stage "encode-$which" "$BUILD/difh3encode" --backend cuda \
    --program "$SHARED/video-encoder/tile-f1-h256-w256.difir" \
    --weight-bundle "$SHARED/video-encoder/tile-f1-h256-w256.difbind" \
    --image "$image" --pixels-id 1 --moments-id 375 \
    --output-moments "$OUT/$which-moments.diftensor" \
    --output-latent "$OUT/$which-latent.diftensor" \
    --output-rows "$OUT/$which-rows.diftensor" \
    --tile-size 256 --tile-overlap 64 --posterior-seed 42 \
    --cache-dir "$CACHE" --min-free-mib 512
}

run_stage vision "$BUILD/difh3vision" run \
  --program "$NATIVE/vision/qwen3vl-vision-t1h30w52.difir" \
  --bundle "$NATIVE/vision/qwen3vl-vision-t1h30w52.difbind" \
  --inputs "$OUT/presentation.safetensors" --grid-t 1 --grid-h 30 --grid-w 52 \
  --output "$OUT/vision.safetensors" --backend cuda \
  --cache-dir "$CACHE" --min-free-mib 512 --streamed-stage-threads 4 \
  --report "$OUT/vision.json"

encode_keyframe first "$FIRST" &
encode_first_pid=$!
encode_keyframe last "$LAST" &
encode_last_pid=$!

run_stage conditioner "$BUILD/difcondition" run --backend cuda \
  --program "$NATIVE/conditioner/qwen3vl-mm-s1256-v780-l50.difir" \
  --bundle "$NATIVE/conditioner/qwen3vl-mm-s1256-v780-l50.difbind" \
  --vision-inputs "$OUT/presentation.safetensors" \
  --vision-outputs "$OUT/vision.safetensors" --vision-tokens 780 \
  --output "$OUT/conditioning.diftensor" \
  --convrot-int8-checkpoint "$CONDITIONER_CONVROT" \
  --convrot-int8-linear-count 350 --convrot-int8-weight-only-quality \
  --streamed-stage-threads 4 --profile-pipeline \
  --cache-dir "$CACHE" --min-free-mib 512 --report "$OUT/conditioner.json"

wait "$encode_first_pid"
wait "$encode_last_pid"

# Stock ComfyUI uses one Torch CPU seed-42 stream for video then audio and
# independently restarts seed 42 for each visual-condition augmentation.
run_stage noise-video "$BUILD/difh3noise" --rng torch-cpu --layout h3-video \
  --latent-frames 37 --latent-height 30 --latent-width 52 --seed 42 \
  --output "$OUT/target-video-noise.diftensor" \
  --verify-against "$NATIVE/start/target-video-noise.diftensor"
for which in first last; do
  run_stage "noise-condition-$which" "$BUILD/difh3noise" \
    --rng torch-cpu --layout h3-video --latent-frames 1 \
    --latent-height 30 --latent-width 52 --seed 42 \
    --output "$OUT/$which-condition-noise.diftensor" \
    --verify-against "$NATIVE/start/condition-noise.diftensor"
done
run_stage noise-audio "$BUILD/difh3noise" --rng torch-cpu --layout h3-audio \
  --audio-latents 207 --seed 42 --skip-normal-values 1385280 \
  --output "$OUT/audio-noise.diftensor" \
  --verify-against "$NATIVE/start/audio-noise.diftensor"
run_stage initial-state "$BUILD/difh3state" \
  --condition "$OUT/first-rows.diftensor" \
  --condition "$OUT/last-rows.diftensor" \
  --condition-noise "$OUT/first-condition-noise.diftensor" \
  --condition-noise "$OUT/last-condition-noise.diftensor" \
  --target-noise "$OUT/target-video-noise.diftensor" \
  --condition-timestep 0.999 --output "$OUT/video-all-rows.diftensor"

# Compiler-owned strict CK-INT8 attention is selected because its matched
# S=16,880 kernel gate beat Comfy Kitchen and remained bit-identical there.
run_stage denoise "$BUILD/difh3infer" --backend cuda \
  --sampler res_multistep \
  --denoiser-program "$NATIVE/denoiser/denoiser-s16880-t37-a207-c1256-t3.difir" \
  --denoiser-bundle "$NATIVE/denoiser/denoiser-s16880-t37-a207-c1256-t3.difbind" \
  --all-text-tokens 1256 --text "$OUT/conditioning.diftensor" \
  --video "$OUT/video-all-rows.diftensor" --audio "$OUT/audio-noise.diftensor" \
  --simple-steps 7 --max-evaluations "$EVALUATIONS" \
  --latent-t 37 --latent-h 30 --latent-w 52 \
  --audio-latents 207 --keyframes first-last \
  --output-latent "$OUT/video-latent.diftensor" \
  --output-video-rows "$OUT/video-rows.diftensor" \
  --output-audio "$OUT/audio-rows.diftensor" \
  --output-audio-latent "$OUT/audio-latent.diftensor" \
  --h3-convrot-int8-checkpoint "$CONVROT" --h3-convrot-int8-layers 50 \
  --h3-convrot-int8-resident-layers 50 --h3-int8-mlp-chunk-rows 2048 \
  --h3-int8-cutlass-scaled-all --h3-int8-compact-adaln \
  --h3-cache-text-refiner --h3-modulation-cache "$MODCACHE" \
  --h3-modulation-source-index "$CKPT/transformer/model.safetensors.index.json" \
  --h3-modulation-steps 8 --h3-ck-attention-dso "$CK_ATTENTION" \
  --h3-int8-attention-first-layer 0 --h3-int8-attention-layers 50 \
  --lazy-resident-upload --streamed-keep-pages --streamed-stage-threads 4 \
  --denoise-only --profile-pipeline --cache-dir "$CACHE" --min-free-mib 512

# Video and audio decode consume disjoint final latents. Their outputs were
# byte-identical when scheduled concurrently, so overlap them and join before
# mux instead of serializing the independent decoder tails.
run_stage video-decode "$BUILD/difvaedecode" --backend cuda \
  --program "$SHARED/video-vae/tile-t7-h16-w16-l36.difir" \
  --weight-bundle "$SHARED/video-vae/tile-t7-h16-w16-l36.difbind" \
  --input "$OUT/video-latent.diftensor" --latent-id 1 --raw-id 1223 \
  --output-rgb "$OUT/frames.rgb" --clip-length 17 --token-drop 3 \
  --tile-size 256 --tile-overlap 64 --warmups 0 --iterations 1 \
  --min-free-mib 512 --cache-dir "$CACHE" \
  --convrot-int8-checkpoint "$VAE_CONVROT" --convrot-int8-resident &
video_decode_pid=$!

run_stage audio-decode "$BUILD/difaudiodecode" --backend cuda \
  --program "$SHARED/audio-vae-t207/t207.difir" \
  --weight-bundle "$SHARED/audio-vae-t207/t207.difbind" \
  --input "$OUT/audio-rows.diftensor" --output-wav "$OUT/audio.wav" \
  --sample-rate 32000 --cache-dir "$CACHE" --min-free-mib 512 &
audio_decode_pid=$!

wait "$video_decode_pid"
wait "$audio_decode_pid"

run_stage mux ffmpeg -hide_banner -loglevel error -y \
  -f rawvideo -pixel_format rgb24 -video_size 832x480 -framerate 24 \
  -i "$OUT/frames.rgb" -i "$OUT/audio.wav" -frames:v 124 \
  -c:v h264_nvenc -preset p4 -pix_fmt yuv420p \
  -c:a aac -ar 32000 -ac 2 -shortest "$OUT/video.mp4"

ffprobe -v error -count_frames -show_entries \
  format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,sample_rate,channels \
  -of json "$OUT/video.mp4" >"$OUT/ffprobe.json"
sha256sum "$OUT/video.mp4" >"$OUT/video.mp4.sha256"
if [[ "$EVALUATIONS" == "7" ]]; then
  echo "H3_FL2VA_PROMPT_TO_MP4 COMPLETE evaluations=7 artifact=$OUT/video.mp4 comparator_seconds=81.0952614400303 target_seconds=40.54763072001515 acceptance=external_prompt_to_saved_output_wall_only"
else
  echo "H3_FL2VA_PROMPT_TO_MP4 SMOKE_ONLY evaluations=$EVALUATIONS artifact=$OUT/video.mp4"
fi
