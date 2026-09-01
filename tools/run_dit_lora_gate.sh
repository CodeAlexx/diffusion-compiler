#!/usr/bin/env bash
# M2: BF16 LoRA DiT-block training gate — N-step adapter-only AdamW training
# of make_dit_lora_training vs the exactly-mirrored PyTorch reference, on
# CPU and CUDA, plus checkpoint-resume byte-identity and .alpha-guarded
# export. Each run emits the measured evidence used by the frozen bars below.
#
# usage: flock /tmp/dc-gpu.lock -c \
#   'bash tools/run_dit_lora_gate.sh WORKDIR [BLOCKS] [STEPS] [--measure]'
#
# Admission bars frozen AFTER measurement at 2 and 4 blocks, 100 steps
# (2026-08-31, RTX 3090 Ti, torch 2.10.0+cu128).  BF16-at-depth measurement:
# single-step semantics are tight (step-1 dL/dA byte-identical ZERO while
# B==0; step-1 dL/dB worst rel-L2 1.26e-2, cos 0.999924), but 100-step
# trajectories separate through BF16 rounding order — the three-way spread
# among {torch, our CPU, our CUDA} is mutually comparable (our own CPU vs
# CUDA final params rel-L2 0.074 at 2 blocks / 0.210 at 4 blocks, torch-vs-
# ours 0.065/0.237), and drift grows with steps (param rel-L2 8.6e-3 at
# step 10 -> 6.5e-2 at step 100, 2 blocks CPU).  End-state categories are
# therefore gated on ABSOLUTE scale + finiteness, with direction/relative
# bars deliberately open and the mechanism documented — never quote the
# tight step-1 numbers as trajectory parity.
set -euo pipefail

WORKDIR="${1:?usage: run_dit_lora_gate.sh WORKDIR [BLOCKS] [STEPS] [--measure]}"
BLOCKS="${2:-2}"
STEPS="${3:-100}"
MODE="${4:-enforce}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
FIXTURE="$WORKDIR/fixture-b$BLOCKS"
mkdir -p "$WORKDIR"

# Frozen category bars (measured worst -> bar):
#   losses   rel 6.6e-3 / abs 9.4e-3 / cos 0.999978 -> 2e-2 / 2.5e-2 / 0.99995
#   pred     rel 1.52e-1 / abs 3.83e-1 / cos 0.98854 -> 4e-1 / 1.0 / 0.98
#   grad1 B  rel 1.26e-2 / abs 3.25e-5 / cos 0.999924 -> 4e-2 / 1e-4 / 0.9995
#   grad1 A  byte-identical zero REQUIRED
#   grad     abs 1.58e-1 (direction noise-dominated at convergence) -> abs 5e-1
#   moment1  abs 1.11e-2 -> 5e-2 ; moment2 abs 5.6e-5 -> 5e-4 (direction open)
#   param    rel 2.37e-1 / abs 1.73e-1 / cos 0.97271 -> 5e-1 / 5e-1 / 0.95
LOSS_BARS="--min-cos 0.99995 --max-rel-l2 2e-2 --max-abs 2.5e-2 --min-norm-ratio 0.998 --max-norm-ratio 1.002"
PRED_BARS="--min-cos 0.98 --max-rel-l2 4e-1 --max-abs 1.0 --min-norm-ratio 0.9 --max-norm-ratio 1.1"
GRAD1B_BARS="--min-cos 0.9995 --max-rel-l2 4e-2 --max-abs 1e-4 --min-norm-ratio 0.99 --max-norm-ratio 1.01"
GRAD_BARS="--min-cos -1 --max-rel-l2 1e9 --max-abs 5e-1 --min-norm-ratio 0 --max-norm-ratio 1e9"
MOM1_BARS="--min-cos -1 --max-rel-l2 1e9 --max-abs 5e-2 --min-norm-ratio 0 --max-norm-ratio 1e9"
MOM2_BARS="--min-cos -1 --max-rel-l2 1e9 --max-abs 5e-4 --min-norm-ratio 0 --max-norm-ratio 1e9"
PARAM_BARS="--min-cos 0.95 --max-rel-l2 5e-1 --max-abs 5e-1 --min-norm-ratio 0.97 --max-norm-ratio 1.03"

rm -rf "$FIXTURE"
python3 "$ROOT/tools/export_dit_lora_training_reference.py" "$FIXTURE" \
  --blocks "$BLOCKS" --steps "$STEPS"

ADAPTERS=$((12 * BLOCKS))
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
  rm -f "$WORKDIR/b$BLOCKS-$backend.diftrain" \
    "$WORKDIR/b$BLOCKS-$backend-adapters.safetensors"
  "$BUILD/difdittrain" --backend "$backend" --fixture "$FIXTURE" \
    --output "$out" --checkpoint "$WORKDIR/b$BLOCKS-$backend.diftrain" \
    --export-adapters "$WORKDIR/b$BLOCKS-$backend-adapters.safetensors" \
    | tee "$WORKDIR/difdittrain-b$BLOCKS-$backend.log"
  grep -q "^BASE_BITS_UNCHANGED" "$WORKDIR/difdittrain-b$BLOCKS-$backend.log"

  compare "$FIXTURE/ref-losses.diftensor" "$out/losses.diftensor" \
    "$backend/losses" $LOSS_BARS
  compare "$FIXTURE/ref-prediction.diftensor" "$out/prediction.diftensor" \
    "$backend/prediction" $PRED_BARS
  for i in $(seq 0 2 $((ADAPTERS - 2))); do
    comparisons=$((comparisons+1))
    if cmp -s "$FIXTURE/ref-grad1-$i.diftensor" "$out/grad1-$i.diftensor"; then
      echo "PASS $backend/grad1-A-$i byte-identical-zero"
    else
      failures=$((failures+1)); echo "FAIL $backend/grad1-A-$i"
    fi
  done
  for i in $(seq 1 2 $((ADAPTERS - 1))); do
    compare "$FIXTURE/ref-grad1-$i.diftensor" "$out/grad1-$i.diftensor" \
      "$backend/grad1-B-$i" $GRAD1B_BARS
  done
  for i in $(seq 0 $((ADAPTERS - 1))); do
    compare "$FIXTURE/ref-grad-$i.diftensor" "$out/grad-$i.diftensor" \
      "$backend/grad-$i" $GRAD_BARS
    compare "$FIXTURE/ref-param-$i.diftensor" "$out/param-$i.diftensor" \
      "$backend/param-$i" $PARAM_BARS
    compare "$FIXTURE/ref-moment1-$i.diftensor" "$out/moment1-$i.diftensor" \
      "$backend/moment1-$i" $MOM1_BARS
    compare "$FIXTURE/ref-moment2-$i.diftensor" "$out/moment2-$i.diftensor" \
      "$backend/moment2-$i" $MOM2_BARS
  done
  compare "$FIXTURE/ref-param-1.diftensor" \
    "$WORKDIR/b$BLOCKS-$backend-adapters.safetensors::block0.q.lora_B.weight" \
    "$backend/export-spot" $PARAM_BARS
  # Convergence sanity: final loss at least 5x below initial.
  python3 - "$out/losses.diftensor" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
ndim = struct.unpack_from("<III", data, 8)[2]
off = 20 + 8 * ndim
(n,) = struct.unpack_from("<Q", data, off)
import array
vals = array.array("f"); vals.frombytes(data[off+8:off+8+n])
first, last = vals[0], vals[-1]
assert last < 0.2 * first, f"loss did not converge: {first} -> {last}"
print(f"CONVERGED {first:.6f} -> {last:.6f}")
PY
done

# Resume byte-identity (CUDA): STEPS split as 40%+60% must reproduce the
# direct checkpoint bit-for-bit.
R1=$((STEPS * 2 / 5)); R2=$((STEPS - R1))
if [ "$R1" -ge 1 ] && [ "$R2" -ge 1 ]; then
  rm -rf "$WORKDIR/resume-part" "$WORKDIR/resume-out"
  rm -f "$WORKDIR/resume-40.diftrain" "$WORKDIR/resume-100.diftrain"
  "$BUILD/difdittrain" --backend cuda --fixture "$FIXTURE" \
    --output "$WORKDIR/resume-part" --steps "$R1" \
    --checkpoint "$WORKDIR/resume-40.diftrain" > /dev/null
  "$BUILD/difdittrain" --backend cuda --fixture "$FIXTURE" \
    --output "$WORKDIR/resume-out" --steps "$R2" \
    --resume "$WORKDIR/resume-40.diftrain" \
    --checkpoint "$WORKDIR/resume-100.diftrain" > /dev/null
  direct=$(sha256sum "$WORKDIR/b$BLOCKS-cuda.diftrain" | cut -d' ' -f1)
  resumed=$(sha256sum "$WORKDIR/resume-100.diftrain" | cut -d' ' -f1)
  comparisons=$((comparisons+1))
  if [ "$direct" = "$resumed" ]; then
    echo "PASS resume/checkpoint byte-identical $direct"
  else
    failures=$((failures+1)); echo "FAIL resume/checkpoint $direct != $resumed"
  fi
  comparisons=$((comparisons+1))
  if cmp -s "$WORKDIR/actual-b$BLOCKS-cuda/prediction.diftensor" \
      "$WORKDIR/resume-out/prediction.diftensor"; then
    echo "PASS resume/prediction byte-identical"
  else
    failures=$((failures+1)); echo "FAIL resume/prediction"
  fi
fi

echo "DIT_LORA_GATE blocks=$BLOCKS steps=$STEPS comparisons=$comparisons failures=$failures mode=$MODE"
[ "$failures" -eq 0 ]
