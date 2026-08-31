#!/usr/bin/env bash
# Composed DiT-block training gate: N-step F32 AdamW training of the
# make_dit_block_training graph vs the PyTorch reference, on CPU and CUDA.
#
# usage: bash tools/run_dit_block_gate.sh WORKDIR [BLOCKS] [STEPS] [--measure]
#
# Compares, per backend: the loss history, step-1 gradients, final-step
# gradients, final parameters, and final F32 moments, with difcompare.
#
# Admission bars set AFTER measurement at BOTH 2-block and 4-block depth
# (2026-08-31, 100 steps, RTX 3090 Ti, torch 2.10.0+cu128; full log in
# docs/DIT_BACKWARD_GATE_2026-08-31.md).  Measured worst (union of depths,
# both backends):
#   losses  max_abs 3.6e-7  rel_l2 4.4e-7  cos 1.0
#   grad1   max_abs 1.9e-9  rel_l2 7.7e-7  cos 1.0
#   grad    max_abs 2.5e-7  rel_l2 3.3e-3  cos 0.999995
#   param   max_abs 2.4e-4  rel_l2 1.4e-4  cos 0.99999999
#   moment1 max_abs 2.4e-8  rel_l2 5.8e-4  cos 0.9999998
#   moment2 max_abs 3.0e-11 rel_l2 1.5e-4  cos 1.0
# Composition amplifies error (flame lesson): 100-step final params drift
# 1.8e-6 at 2 blocks -> 2.4e-4 at 4 blocks, so the param bars are set from
# the 4-block worst.  Bars are frozen — never lowered to pass.
set -euo pipefail

WORKDIR="${1:?usage: run_dit_block_gate.sh WORKDIR [BLOCKS] [STEPS] [--measure]}"
BLOCKS="${2:-2}"
STEPS="${3:-100}"
MODE="${4:-enforce}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
FIXTURE="$WORKDIR/fixture-b$BLOCKS"
mkdir -p "$WORKDIR"

python3 "$ROOT/tools/export_dit_block_training_reference.py" "$FIXTURE" \
  --blocks "$BLOCKS" --steps "$STEPS"

PARAMS=$((16 * BLOCKS))
failures=0
comparisons=0
compare() { # expected actual category bars...
  local expected="$1" actual="$2" category="$3"; shift 3
  local line
  comparisons=$((comparisons+1))
  if [ "$MODE" = "--measure" ]; then
    set -- --min-cos -1 --max-rel-l2 1e9 --max-abs 1e9 \
        --min-norm-ratio 0 --max-norm-ratio 1e9
  fi
  if line="$("$BUILD/difcompare" "$expected" "$actual" "$@")"; then
    echo "PASS $category $line"
  else
    failures=$((failures+1))
    echo "FAIL $category"
    "$BUILD/difcompare" "$expected" "$actual" --min-cos -1 --max-rel-l2 1e9 \
      --max-abs 1e9 --min-norm-ratio 0 --max-norm-ratio 1e9 || true
  fi
}

for backend in cpu cuda; do
  out="$WORKDIR/actual-b$BLOCKS-$backend"
  rm -rf "$out"
  "$BUILD/difdittrain" --backend "$backend" --fixture "$FIXTURE" \
    --output "$out" | tee "$WORKDIR/difdittrain-b$BLOCKS-$backend.log"
  compare "$FIXTURE/ref-losses.diftensor" "$out/losses.diftensor" \
    "$backend/losses" --min-cos 0.999999 --max-rel-l2 2e-5 --max-abs 2e-6 \
    --min-norm-ratio 0.9999 --max-norm-ratio 1.0001
  for i in $(seq 0 $((PARAMS-1))); do
    compare "$FIXTURE/ref-grad1-$i.diftensor" "$out/grad1-$i.diftensor" \
      "$backend/grad1-$i" --min-cos 0.99999 --max-rel-l2 5e-4 \
      --max-abs 1e-6 --min-norm-ratio 0.999 --max-norm-ratio 1.001
    compare "$FIXTURE/ref-grad-$i.diftensor" "$out/grad-$i.diftensor" \
      "$backend/grad-$i" --min-cos 0.999 --max-rel-l2 2e-2 \
      --max-abs 1e-5 --min-norm-ratio 0.98 --max-norm-ratio 1.02
    compare "$FIXTURE/ref-param-$i.diftensor" "$out/param-$i.diftensor" \
      "$backend/param-$i" --min-cos 0.999999 --max-rel-l2 5e-4 \
      --max-abs 1e-3 --min-norm-ratio 0.9999 --max-norm-ratio 1.0001
    compare "$FIXTURE/ref-moment1-$i.diftensor" "$out/moment1-$i.diftensor" \
      "$backend/moment1-$i" --min-cos 0.999 --max-rel-l2 2e-2 \
      --max-abs 1e-6 --min-norm-ratio 0.98 --max-norm-ratio 1.02
    compare "$FIXTURE/ref-moment2-$i.diftensor" "$out/moment2-$i.diftensor" \
      "$backend/moment2-$i" --min-cos 0.999 --max-rel-l2 2e-2 \
      --max-abs 1e-6 --min-norm-ratio 0.98 --max-norm-ratio 1.02
  done
done
echo "DIT_BLOCK_GATE blocks=$BLOCKS steps=$STEPS comparisons=$comparisons failures=$failures mode=$MODE"
[ "$failures" -eq 0 ]
