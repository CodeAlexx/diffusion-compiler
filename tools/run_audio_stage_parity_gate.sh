#!/usr/bin/env bash
# CPU stage-parity gate for the native BigVGAN audio decoder (gate 3 of
# docs/BIGVGAN_DECODE_PLAN.md): the DiffIR CPU executor vs the pinned torch
# oracle on the accepted run's REAL [584,32] audio state rows, at every
# staged boundary (pre, stage0..stage6, tail).
#
# usage: bash tools/run_audio_stage_parity_gate.sh WORKDIR [--measure]
#
# Bars FROZEN after the recorded --measure run (2026-08-31, torch
# 2.10.0+cu128 CPU oracle, real [584,32] rows, 9/9 boundaries, zero
# nonfinite everywhere): worst rel-L2 1.97e-5 (stage3), worst cosine
# 0.9999999998 (stage0..tail all >= 0.99999999980), worst max_abs 7.95e-4
# (stage3, activation magnitudes), norm_ratio within 1 +/- 1.9e-7. The
# plan's provisional bars (rel-L2 <= 5e-5, cos >= 0.9999999) hold with
# >2.5x margin and are frozen as-is; never lowered to pass.
set -euo pipefail

WORKDIR="${1:?usage: run_audio_stage_parity_gate.sh WORKDIR [--measure]}"
MODE="${2:-enforce}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
ROWS="/home/alex/diffusion-compiler/artifacts/h3-quality-natural-language-2026-08-30/compiler/inputs/audio_state_rows.diftensor"
CKPT="/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/audio_vae/model.safetensors"
mkdir -p "$WORKDIR"

# 1. Torch oracle boundaries on the real rows (CPU, F32; reused if present).
if [[ ! -f "$WORKDIR/oracle/manifest.json" ]]; then
  python3 "$ROOT/tools/export_audio_bigvgan_reference.py" stage-dump \
    "$ROWS" "$WORKDIR/oracle" --stages 7
fi

# 2. Folded weights (float64 fold + filter expansion).
if [[ ! -f "$WORKDIR/folded.safetensors" ]]; then
  "$BUILD/difimport" fold-audio-weight-norm "$CKPT" "$WORKDIR/folded.safetensors"
fi

# 3. Latent input from the recorded rows.
"$BUILD/difimport" unpack-audio-rows "$ROWS" "$WORKDIR/latent.diftensor"

BOUNDARIES=(pre stage0 stage1 stage2 stage3 stage4 stage5 stage6 tail)
STAGE_ARGS=(0 1 2 3 4 5 6 7 8)

status=0
for index in "${!BOUNDARIES[@]}"; do
  boundary="${BOUNDARIES[$index]}"
  stages="${STAGE_ARGS[$index]}"
  dir="$WORKDIR/$boundary"
  rm -rf "$dir"; mkdir -p "$dir"
  emit=$("$BUILD/difimport" make-audio-program "$dir/program.difir" 2 292 "$stages")
  echo "$emit"
  input_id=$(sed -n 's/.* input_id=\([0-9]*\).*/\1/p' <<<"$emit")
  output_id=$(sed -n 's/.* output_id=\([0-9]*\).*/\1/p' <<<"$emit")
  "$BUILD/difimport" make-audio-bundle "$WORKDIR/folded.safetensors" \
    "$dir/program.difir" "$dir/derived.safetensors" "$dir/bundle.difbind" \
    2 292 "$stages" > "$dir/bundle.log"
  "$BUILD/difrun" --backend cpu --program "$dir/program.difir" \
    --weight-bundle "$dir/bundle.difbind" \
    --input "$input_id=$WORKDIR/latent.diftensor" \
    --output "$output_id=$dir/actual.diftensor" \
    --warmups 0 --iterations 1
  oracle="$WORKDIR/oracle/$boundary.diftensor"
  if [[ "$MODE" == "--measure" || "$MODE" == "measure" ]]; then
    "$BUILD/difcompare" "$oracle" "$dir/actual.diftensor" --min-cos -1 \
      --max-rel-l2 1e9 --min-norm-ratio 0 --max-norm-ratio 1e9 || status=1
  else
    # FROZEN bars (see header provenance).
    "$BUILD/difcompare" "$oracle" "$dir/actual.diftensor" \
      --min-cos 0.9999999 --max-rel-l2 5e-5 \
      --min-norm-ratio 0.9999 --max-norm-ratio 1.0001 || status=1
  fi
done
if [[ $status -eq 0 ]]; then
  echo "AUDIO_STAGE_PARITY PASS boundaries=${#BOUNDARIES[@]} mode=$MODE"
else
  echo "AUDIO_STAGE_PARITY FAIL" >&2
fi
exit $status
