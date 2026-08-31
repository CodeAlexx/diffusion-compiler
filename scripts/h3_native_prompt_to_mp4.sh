#!/usr/bin/env bash
# The accepted MiniMax-H3 generation, from LITERAL PROMPT TEXT, entirely
# native: no PyTorch, Python, Serenity, Mojo/MAX, Rust FFI, or model worker
# executes any part of the neural graph.  ffmpeg is the declared mux
# boundary.  Evidence: docs/H3_NATIVE_PROMPT_TO_MP4_GATE_2026-08-31.md
#
#   scripts/h3_native_prompt_to_mp4.sh PROMPT.txt OUTDIR
#
# Wrap GPU work with `flock /tmp/dc-gpu.lock -c ...` if anything else on the
# box uses the device.  Heavy stages run under the mem-safe scope.
set -euo pipefail

PROMPT="${1:?usage: h3_native_prompt_to_mp4.sh PROMPT.txt OUTDIR}"
OUT="${2:?usage: h3_native_prompt_to_mp4.sh PROMPT.txt OUTDIR}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${DIF_BUILD:-$ROOT/build-vendored}"
CKPT="${DIF_CHECKPOINT:-/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA}"
ART="$ROOT/artifacts/h3-quality-natural-language-2026-08-30"
MEMSAFE="${DIF_MEMSAFE:-/home/alex/mojodiffusion/scripts/mem_safe_runtime.sh}"
SAFE=(env MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G DESKTOP_RESERVE=16G "$MEMSAFE")
mkdir -p "$OUT/cache"

# 1. prompt text -> token ids (native BPE over the checkpoint's own tokenizer)
"$BUILD/diftokenize" --processor "$CKPT/processor" --prompt-file "$PROMPT" \
  --strip-trailing-newline --diftensor-out "$OUT/ids.diftensor"

# 2. ids -> [S,5120] BF16 conditioning (native Qwen3-VL text tower)
[ -f "$OUT/cond.difir" ] || "$BUILD/difcondition" program \
  --sequence "$(python3 - "$OUT/ids.diftensor" <<'PY' 2>/dev/null || echo 439
import struct,sys
b=open(sys.argv[1],'rb').read()
print(struct.unpack_from('<Q', b, 24)[0])
PY
)" --output "$OUT/cond.difir"
[ -f "$OUT/cond.difbind" ] || "$BUILD/difcondition" bundle --checkpoint "$CKPT" \
  --program "$OUT/cond.difir" --output "$OUT/cond.difbind"
"${SAFE[@]}" "$BUILD/difcondition" run --program "$OUT/cond.difir" \
  --bundle "$OUT/cond.difbind" --ids "$OUT/ids.diftensor" \
  --output "$OUT/conditioning.diftensor" --cache-dir "$OUT/cache"

# 3. seeded initial states (native RNG; reproduces the accepted states exactly)
"$BUILD/difh3noise" --rows 20280 --cols 96 --seed 4242 --output "$OUT/video_state_rows.diftensor"
"$BUILD/difh3noise" --rows 584  --cols 32 --seed 4243 --output "$OUT/audio_state_rows.diftensor"

# 4. denoise (50 blocks, exact cuDNN attention, released schedule)
"${SAFE[@]}" "$BUILD/difh3infer" --backend cuda \
  --denoiser-program "$ART/compiler/h3-832x480x175-t439-exact-cudnn.difir" \
  --denoiser-bundle  "$ART/compiler/h3-832x480x175-t439-exact-cudnn.difbind" \
  --all-text-tokens 439 --text "$OUT/conditioning.diftensor" \
  --video "$OUT/video_state_rows.diftensor" --audio "$OUT/audio_state_rows.diftensor" \
  --video-sigmas "$ART/compiler/smoke_exact_bf16/first_eval_inputs/video_sigmas.diftensor" \
  --audio-sigmas "$ART/compiler/smoke_exact_bf16/first_eval_inputs/audio_sigmas.diftensor" \
  --latent-t 52 --latent-h 30 --latent-w 52 --audio-latents 292 --keyframes none \
  --output-latent "$OUT/video-latent.diftensor" --output-audio "$OUT/audio-rows.diftensor" \
  --output-audio-latent "$OUT/audio-latent.diftensor" \
  --h3-modulation-cache "$CKPT/serenity_runtime_cache_v1/modcache_steps_20_blocks_50.safetensors" \
  --h3-modulation-source-index "$CKPT/transformer/model.safetensors.index.json" \
  --h3-modulation-steps 20 --denoise-only --cache-dir "$OUT/cache" --min-free-mib 4096

# 5. video decode (tiled ViT3D VAE)
"${SAFE[@]}" "$BUILD/difvaedecode" --backend cuda \
  --program "$ROOT/artifacts/h3-vae/decoder-native-tile-l36.difir" \
  --weight-bundle "$ROOT/artifacts/h3-vae/decoder-native-tile-l36.difbind" \
  --input "$OUT/video-latent.diftensor" --latent-id 1 --raw-id 1223 \
  --output-raw "$OUT/video-raw.diftensor" --output-decoded "$OUT/video-decoded.diftensor" \
  --clip-length 17 --token-drop 3 --tile-size 256 --tile-overlap 64 \
  --warmups 0 --iterations 1 --min-free-mib 4096 --cache-dir "$OUT/cache"

# 6. audio decode (native BigVGAN) and 7. mux
"$BUILD/difaudiodecode" --backend cuda \
  --program "$OUT/audio.difir" --weight-bundle "$OUT/audio.difbind" \
  --input "$OUT/audio-rows.diftensor" --output-wav "$OUT/audio.wav" \
  --cache-dir "$OUT/cache"
"$BUILD/difh3media" --video "$OUT/video-decoded.diftensor" --audio-wav "$OUT/audio.wav" \
  --output-dir "$OUT/media" --input-fps 24 --encoder h264_nvenc

echo "NATIVE_PROMPT_TO_MP4 done: $OUT/media/video.mp4"
