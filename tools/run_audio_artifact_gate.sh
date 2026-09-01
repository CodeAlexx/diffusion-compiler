#!/usr/bin/env bash
# Artifact + perf gate for the native BigVGAN audio decode: decode the
# accepted run's FINAL audio rows with difaudiodecode and compare against the
# accepted audio.wav; measure
# the fresh-process wall under mem_safe_runtime.sh.
#
# usage: flock /tmp/dc-gpu.lock -c 'bash tools/run_audio_artifact_gate.sh WORKDIR [cuda|cpu]'
#
# READ-ONLY inputs from the accepted artifact; every output goes to WORKDIR.
# Bars (plan gate 4): 44-byte WAV header byte-identical; per-sample int16
# |delta| <= 1; differing samples <= 1%; SNR >= 60 dB; zero nonfinite floats
# before quantization (difaudiodecode fails closed on nonfinite). Perf bar
# (gate 5): fresh-process wall <= 3.27 s (beat Serenity's 3.270 s; the
# current Mojo bridge is 5.25 s).
set -euo pipefail

WORKDIR="${1:?usage: run_audio_artifact_gate.sh WORKDIR [cuda|cpu]}"
BACKEND="${2:-cuda}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
ART=/home/alex/diffusion-compiler/artifacts/h3-quality-natural-language-2026-08-30
ROWS="$ART/compiler/full_exact_bf16/audio-rows.diftensor"
ACCEPTED_WAV="$ART/compiler/full_exact_bf16/audio.wav"
CKPT=/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/audio_vae/model.safetensors
mkdir -p "$WORKDIR"

# Folded weights + full program + sealed bundle (reused if present).
if [[ ! -f "$WORKDIR/folded.safetensors" ]]; then
  "$BUILD/difimport" fold-audio-weight-norm "$CKPT" "$WORKDIR/folded.safetensors"
fi
if [[ ! -f "$WORKDIR/program.difir" ]]; then
  "$BUILD/difimport" make-audio-program "$WORKDIR/program.difir" 2 292 8
  "$BUILD/difimport" make-audio-bundle "$WORKDIR/folded.safetensors" \
    "$WORKDIR/program.difir" "$WORKDIR/derived.safetensors" \
    "$WORKDIR/bundle.difbind" 2 292 8
fi

# Gate 4: decode + WAV comparison (also records prepare/run timing).
"$BUILD/difaudiodecode" --backend "$BACKEND" --program "$WORKDIR/program.difir" \
  --weight-bundle "$WORKDIR/bundle.difbind" --input "$ROWS" \
  --output-wav "$WORKDIR/audio-$BACKEND.wav" \
  --output-waveform "$WORKDIR/waveform-$BACKEND.diftensor"
python3 "$ROOT/tools/check_audio_wav_gate.py" "$ACCEPTED_WAV" \
  "$WORKDIR/audio-$BACKEND.wav"

# Gate 5 (cuda only): fresh-process wall under the mem-safe wrapper.
if [[ "$BACKEND" == "cuda" ]]; then
  MEM_MAX=24G MEM_HIGH=infinity SWAP_MAX=2G DESKTOP_RESERVE=16G \
    "$ROOT/scripts/mem_safe_runtime.sh" /usr/bin/time -v \
    "$BUILD/difaudiodecode" --backend cuda --program "$WORKDIR/program.difir" \
    --weight-bundle "$WORKDIR/bundle.difbind" --input "$ROWS" \
    --output-wav "$WORKDIR/audio-perf.wav" 2> "$WORKDIR/perf.log"
  wall=$(grep "Elapsed (wall clock)" "$WORKDIR/perf.log" | awk '{print $NF}')
  peak=$(grep "Maximum resident set size" "$WORKDIR/perf.log" | awk '{print $NF}')
  echo "AUDIO_PERF wall=$wall max_rss_kib=$peak bar=3.27s bridge=5.25s"
fi
echo "AUDIO_ARTIFACT_GATE DONE backend=$BACKEND workdir=$WORKDIR"
