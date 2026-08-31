#!/usr/bin/env bash
# Per-opcode PyTorch fixture gate for the DiT backward opcodes.
#
# usage: bash tools/run_dit_backward_gate.sh WORKDIR [--measure]
#
# Exports torch fixtures (F32 and BF16 per case), runs every case's backward
# operation through difditops on CPU and CUDA, and admits each gradient with
# difcompare.  --measure prints difcompare's measured metrics without
# enforcing the recorded bars (used once to set them; bars are then frozen
# and never lowered to pass).
#
# Admission bars (set AFTER measurement, 2026-08-31, RTX 3090 Ti +
# PyTorch 2.10.0+cu128 — see docs/DIT_BACKWARD_GATE_2026-08-31.md).
# Measured worst over all 104 comparisons (both backends):
#   F32:  max_abs 4.77e-7 (rms_norm dx), rel_l2 2.69e-7, cos 1.0,
#         norm_ratio [1.0, 1.0]
#   BF16: max_abs 3.91e-3 (attention dv, one BF16 ulp), rel_l2 3.97e-3,
#         cos 0.999992, norm_ratio [0.99936, 1.00027]
# Admitted (~4x-8x above worst, frozen, never lowered to pass):
#   F32:  cos >= 0.999999, rel-L2 <= 5e-5, max-abs <= 2e-6
#   BF16: cos >= 0.999,    rel-L2 <= 2e-2, max-abs <= 3e-2
# BF16 expectations are F32-math-on-BF16-values rounded once at the store
# (the kernel contract); the C++ kernels round intermediate results the
# reference does not, so BF16 differences sit at a few BF16 ulps.
set -euo pipefail

WORKDIR="${1:?usage: run_dit_backward_gate.sh WORKDIR [--measure]}"
MODE="${2:-enforce}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
FIXTURES="$WORKDIR/fixtures"
mkdir -p "$WORKDIR"

python3 "$ROOT/tools/export_dit_backward_fixtures.py" "$FIXTURES"

CASES=(rms_norm rms_norm_modulate_weighted rms_norm_modulate_plain \
       swiglu_gatefirst swiglu_valuefirst residual_gate layer_norm \
       qk_norm_rope_fulltable qk_norm_rope_halftable \
       attention_full attention_causal)

declare -A WRT=(
  [rms_norm]="x weight"
  [rms_norm_modulate_weighted]="x weight scale shift"
  [rms_norm_modulate_plain]="x scale shift"
  [swiglu_gatefirst]="x"
  [swiglu_valuefirst]="x"
  [residual_gate]="branch gate"
  [layer_norm]="x weight bias"
  [qk_norm_rope_fulltable]="x weight"
  [qk_norm_rope_halftable]="x weight"
  [attention_full]="q k v"
  [attention_causal]="q k v"
)

failures=0
comparisons=0
for case_name in "${CASES[@]}"; do
  for dtype in f32 bf16; do
    for backend in cpu cuda; do
      out="$WORKDIR/actual/$case_name/$dtype/$backend"
      mkdir -p "$out"
      "$BUILD/difditops" "$case_name" \
        --fixture "$FIXTURES/$case_name/$dtype" \
        --output "$out" --backend "$backend" > /dev/null
      for grad in ${WRT[$case_name]}; do
        expected="$FIXTURES/$case_name/$dtype/expected-grad-$grad.diftensor"
        actual="$out/actual-grad-$grad.diftensor"
        if [ "$dtype" = f32 ]; then
          bars=(--min-cos 0.999999 --max-rel-l2 5e-5 --max-abs 2e-6 \
                --min-norm-ratio 0.9999 --max-norm-ratio 1.0001)
        else
          bars=(--min-cos 0.999 --max-rel-l2 2e-2 --max-abs 3e-2 \
                --min-norm-ratio 0.98 --max-norm-ratio 1.02)
        fi
        if [ "$MODE" = "--measure" ]; then
          bars=(--min-cos -1 --max-rel-l2 1e9 --max-abs 1e9 \
                --min-norm-ratio 0 --max-norm-ratio 1e9)
        fi
        line="$("$BUILD/difcompare" "$expected" "$actual" "${bars[@]}" || echo COMPARE_FAILED)"
        comparisons=$((comparisons+1))
        if [[ "$line" == *COMPARE_FAILED* ]]; then
          failures=$((failures+1))
          echo "FAIL $case_name/$dtype/$backend grad=$grad"
          "$BUILD/difcompare" "$expected" "$actual" --min-cos -1 \
            --max-rel-l2 1e9 --max-abs 1e9 --min-norm-ratio 0 \
            --max-norm-ratio 1e9 || true
        else
          echo "PASS $case_name/$dtype/$backend grad=$grad $line"
        fi
      done
    done
  done
done
echo "DIT_BACKWARD_GATE comparisons=$comparisons failures=$failures mode=$MODE"
[ "$failures" -eq 0 ]
