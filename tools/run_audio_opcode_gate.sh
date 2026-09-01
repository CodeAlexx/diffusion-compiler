#!/usr/bin/env bash
# Per-opcode PyTorch fixture gate for the DiffIR audio opcodes (Conv1d,
# SnakeBeta).
#
# usage: bash tools/run_audio_opcode_gate.sh WORKDIR [--measure] [--backend cpu|cuda|both]
#
# Exports torch fixtures (F32), runs every case through difaudioops, and
# admits each output with difcompare. --measure prints measured metrics
# without enforcing bars (used once to set them; bars are then frozen and
# never lowered to pass).
#
# Admission bars (F32, FROZEN after the recorded --measure run 2026-08-31,
# torch 2.10.0+cu128, CPU reference, 14 cases): measured worst
# max_abs 1.43e-6 (conv_k11_dilated5 — the widest dilated F32 dot, pure
# reduction-order vs torch's blocked sums), worst rel_l2 1.80e-7, worst
# cosine 0.99999999999998, norm_ratio within [1-1.8e-8, 1+7.4e-9], zero
# nonfinite everywhere. The plan's provisional 1e-6 max-abs was exceeded by
# that one case for the anticipated reduction-order reason; frozen bars sit
# ~4x above the measured worst (the DiT backward gate's honesty margin)
# and are never lowered to pass.
set -euo pipefail

WORKDIR="${1:?usage: run_audio_opcode_gate.sh WORKDIR [--measure] [--backend cpu|cuda|both]}"
shift || true
MODE="enforce"
BACKENDS=(cpu)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --measure) MODE="measure" ;;
    --backend)
      shift
      case "${1:-}" in
        cpu) BACKENDS=(cpu) ;;
        cuda) BACKENDS=(cuda) ;;
        both) BACKENDS=(cpu cuda) ;;
        *) echo "unknown backend: ${1:-}" >&2; exit 2 ;;
      esac ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift || true
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
FIXTURES="$WORKDIR/fixtures"
mkdir -p "$WORKDIR"

python3 "$ROOT/tools/export_audio_opcode_fixtures.py" "$FIXTURES"

CASES=(conv_k1_pointwise conv_k3_dilated3 conv_k7_plain conv_k11_dilated5 \
       conv_k9_grouped3 conv_k3_stride2_nopad \
       conv_k12_depthwise_replicate_asym conv_k4_transposed_stride2 \
       conv_k9_transposed_stride5 conv_k12_transposed_depthwise_replicate \
       conv_k7_transposed_grouped snake_beta_c7 snake_beta_c1 snake_beta_c64)

BARS=(--min-cos 0.9999999 --max-rel-l2 8e-7 --min-norm-ratio 0.9999999 \
      --max-norm-ratio 1.0000001 --max-abs 6e-6)

status=0
for backend in "${BACKENDS[@]}"; do
  for case_name in "${CASES[@]}"; do
    out="$WORKDIR/$backend/$case_name"
    "$BUILD/difaudioops" "$case_name" --fixture "$FIXTURES/$case_name" \
      --output "$out" --backend "$backend"
    if [[ "$MODE" == "measure" ]]; then
      "$BUILD/difcompare" "$FIXTURES/$case_name/expected.diftensor" \
        "$out/actual.diftensor" --min-cos -1 --max-rel-l2 1e9 \
        --min-norm-ratio 0 --max-norm-ratio 1e9 || status=1
    else
      "$BUILD/difcompare" "$FIXTURES/$case_name/expected.diftensor" \
        "$out/actual.diftensor" "${BARS[@]}" || status=1
    fi
  done
done
if [[ $status -eq 0 ]]; then
  echo "AUDIO_OPCODE_GATE PASS backends=${BACKENDS[*]} cases=${#CASES[@]} mode=$MODE"
else
  echo "AUDIO_OPCODE_GATE FAIL" >&2
fi
exit $status
