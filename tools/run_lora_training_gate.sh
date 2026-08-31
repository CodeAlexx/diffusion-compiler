#!/usr/bin/env bash
# 100-step F32 activation-path LoRA training parity gate vs PyTorch.
# Recorded evidence and frozen admission bars:
#   docs/LORA_TRAINING_GATE_2026-08-31.md
#
# Run the WHOLE script under the machine-wide GPU lock:
#   flock /tmp/dc-gpu.lock -c 'bash tools/run_lora_training_gate.sh WORKDIR'
#
# Requires build/diftrain, build/difcompare, and python3 + torch + CUDA for
# the oracle. WORKDIR is created; an existing oracle/ inside it is reused.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DT="$ROOT/build/diftrain"
DC="$ROOT/build/difcompare"
WORK="${1:?usage: run_lora_training_gate.sh WORKDIR}"
mkdir -p "$WORK"
cd "$WORK"

# Frozen category bars (set from the recorded 2026-08-31 measurement; see the
# gate doc for the measured worst cases they cover). Never lower to pass.
COS="--min-cos 0.999999 --min-norm-ratio 0.9999 --max-norm-ratio 1.0001"
LOSS_BARS="--max-rel-l2 2e-6 --max-abs 2.5e-6"
PRED_BARS="--max-rel-l2 2e-6 --max-abs 5e-6"
PARAM_BARS="--max-rel-l2 1e-5 --max-abs 1e-5"
MOMENT_BARS="--max-rel-l2 1e-4 --max-abs 2e-7"
GRAD_BARS="--max-rel-l2 4e-4 --max-abs 2e-6"

if [ ! -d oracle ]; then
  python3 "$ROOT/tools/export_lora_training_reference.py" \
    --output oracle --steps 100
fi

rm -f lora.difir cpu-100.diftrain cpu-1.diftrain cuda-100.diftrain \
  cuda-40.diftrain cuda-40p60.diftrain cuda-1.diftrain \
  cpu-losses.diftensor cpu-1-losses.diftensor cuda-losses.diftensor \
  cuda-40-losses.diftensor cuda-60-losses.diftensor cuda-1-losses.diftensor \
  cpu-pred.diftensor cuda-pred.diftensor cuda-resume-pred.diftensor \
  cuda-adapters.safetensors cpu-adapters.safetensors inspect.txt
rm -rf cpu-grads cuda-grads cuda-resume-grads cpu-step1-grads \
  cuda-step1-grads cpu-export cuda-export

$DT make-lora lora.difir 16 8 4 16 4 8.0
$DT run-lora --backend cpu --program lora.difir --fixture oracle \
  --steps 100 --checkpoint cpu-100.diftrain --losses cpu-losses.diftensor \
  --prediction cpu-pred.diftensor --gradients-dir cpu-grads
$DT run-lora --backend cpu --program lora.difir --fixture oracle \
  --steps 1 --checkpoint cpu-1.diftrain --losses cpu-1-losses.diftensor \
  --gradients-dir cpu-step1-grads
$DT run-lora --backend cuda --program lora.difir --fixture oracle \
  --steps 100 --checkpoint cuda-100.diftrain --losses cuda-losses.diftensor \
  --prediction cuda-pred.diftensor --gradients-dir cuda-grads
$DT run-lora --backend cuda --program lora.difir --fixture oracle \
  --steps 40 --checkpoint cuda-40.diftrain --losses cuda-40-losses.diftensor
$DT run-lora --backend cuda --program lora.difir --fixture oracle \
  --resume cuda-40.diftrain --steps 60 --checkpoint cuda-40p60.diftrain \
  --losses cuda-60-losses.diftensor --prediction cuda-resume-pred.diftensor \
  --gradients-dir cuda-resume-grads
$DT run-lora --backend cuda --program lora.difir --fixture oracle \
  --steps 1 --checkpoint cuda-1.diftrain --losses cuda-1-losses.diftensor \
  --gradients-dir cuda-step1-grads

echo "== resume byte identity =="
DIRECT_SHA=$(sha256sum cuda-100.diftrain | cut -d' ' -f1)
RESUME_SHA=$(sha256sum cuda-40p60.diftrain | cut -d' ' -f1)
[ "$DIRECT_SHA" = "$RESUME_SHA" ] && echo "RESUME_CHECKPOINT_BYTE_IDENTICAL $DIRECT_SHA"
cmp cuda-pred.diftensor cuda-resume-pred.diftensor && echo "RESUME_PRED_BYTE_IDENTICAL"
for f in cuda-grads/*.diftensor; do
  cmp "$f" "cuda-resume-grads/$(basename "$f")" \
    && echo "RESUME_GRAD_BYTE_IDENTICAL $(basename "$f")"
done

$DT inspect cuda-100.diftrain > inspect.txt
head -1 inspect.txt
$DT export cpu-100.diftrain cpu-export > /dev/null
$DT export cuda-100.diftrain cuda-export > /dev/null
$DT export-lora --program lora.difir --checkpoint cuda-100.diftrain \
  --output cuda-adapters.safetensors
$DT export-lora --program lora.difir --checkpoint cpu-100.diftrain \
  --output cpu-adapters.safetensors

# Moment id mapping: checkpoint state ids above the six adapter parameters
# (12..17), ascending, pair (m1,m2) per parameter in order.
mapfile -t MOMENTS < <(grep '^STATE' inspect.txt \
  | sed 's/.*id=\([0-9]*\) .*/\1/' | sort -n | awk '$1>17')
echo "moment ids: ${MOMENTS[*]}"

for BE in cpu cuda; do
  echo "== $BE vs oracle =="
  $DC oracle/losses.diftensor $BE-losses.diftensor $COS $LOSS_BARS
  $DC oracle/prediction.diftensor $BE-pred.diftensor $COS $PRED_BARS
  for k in 0 1 2 3 4 5; do
    pid=$((12 + k))
    m1=${MOMENTS[$((2 * k))]}
    m2=${MOMENTS[$((2 * k + 1))]}
    $DC oracle/state-$pid.diftensor $BE-export/tensor-$pid.diftensor \
      $COS $PARAM_BARS
    $DC oracle/moment1-$pid.diftensor $BE-export/tensor-$m1.diftensor \
      $COS $MOMENT_BARS
    $DC oracle/moment2-$pid.diftensor $BE-export/tensor-$m2.diftensor \
      $COS $MOMENT_BARS
    $DC oracle/gradient-final-$pid.diftensor $BE-grads/gradient-$pid.diftensor \
      $COS $GRAD_BARS
  done
  # Step-1 ordering law: while B == 0, dL/dA is EXACTLY zero on both sides
  # (byte-identical zero tensors); dL/dB is nonzero and bar-compared.
  for pid in 12 14 16; do
    cmp oracle/gradient-step-1-$pid.diftensor \
      $BE-step1-grads/gradient-$pid.diftensor \
      && echo "STEP1_A_GRAD_BYTE_IDENTICAL_ZERO $BE $pid"
  done
  for pid in 13 15 17; do
    $DC oracle/gradient-step-1-$pid.diftensor \
      $BE-step1-grads/gradient-$pid.diftensor $COS $GRAD_BARS
  done
  for spec in "12 latent_proj.lora_A.weight" "13 latent_proj.lora_B.weight" \
              "14 time_proj.lora_A.weight" "15 time_proj.lora_B.weight" \
              "16 out_proj.lora_A.weight" "17 out_proj.lora_B.weight"; do
    set -- $spec
    $DC oracle/state-$1.diftensor "$BE-adapters.safetensors::$2" \
      $COS $PARAM_BARS
  done
done
echo "LORA_GATE PASS"
