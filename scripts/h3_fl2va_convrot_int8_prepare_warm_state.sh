#!/usr/bin/env bash
# Prepare only prompt-independent H3 model bytes in the Linux page cache.
# Run this before starting the external prompt-to-saved-MP4 timer. This mirrors
# the accepted ComfyUI comparator's cached loader nodes; it never reads a
# prompt, keyframe, latent, or output artifact.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${DIF_CHECKPOINT:-/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA}"
NATIVE="$ROOT/artifacts/h3-flref-product-2026-09-01/fl2va-t37-native"
CONVROT="$CKPT/serenity_runtime_cache_v1/native_convrot_h256_official_blocks_50.safetensors"
CONDITIONER_CONVROT="$NATIVE/cache/qwen3vl50-convrot-h256-s1256-v780.safetensors"
MODCACHE="$CKPT/serenity_runtime_cache_v1/native_modcache_keyframe_steps8_blocks50.safetensors"
VAE_CONVROT="$NATIVE/video-vae-convrot-f16/video-vae-convrot-f16.safetensors"

files=(
  "$CONDITIONER_CONVROT"
  "$MODCACHE"
  "$VAE_CONVROT"
  "$CONVROT"
)

total=0
for path in "${files[@]}"; do
  [[ -f "$path" ]] || { echo "missing static model file: $path" >&2; exit 2; }
  bytes="$(stat -c %s "$path")"
  total=$((total + bytes))
  echo "H3_WARM_MODEL_FILE bytes=$bytes path=$path"
  dd if="$path" of=/dev/null bs=8M status=none
done

echo "H3_WARM_MODEL_STATE PASS files=${#files[@]} bytes=$total prompt_dependent_bytes=0"
