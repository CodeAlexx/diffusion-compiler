#!/usr/bin/env bash
# W1-R streamed gate: run the synthetic streamed H3 stack, record wall time,
# difrun stdout+stderr, and the SHA-256 of the output tensor.
# usage: gate_run.sh LABEL [extra difrun args...]
set -u
label="$1"; shift
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$here/logs"
log="$here/logs/$label.log"
start=$(date +%s.%N)
flock /tmp/dc-gpu.lock -c "${DIFRUN:-$here/../build/difrun} --backend cuda \
  --program $here/gate-stack.difir \
  $(tr '\n' ' ' < "$here/tensors/difrun_inputs.txt") \
  $(tr '\n' ' ' < "$here/tensors/difrun_outputs.txt") \
  --map-inputs --warmups 1 --iterations 3 --profile-pipeline \
  --cache-dir $here/ptx-cache $*" > "$log" 2>&1
status=$?
end=$(date +%s.%N)
wall=$(echo "$end $start" | awk '{printf "%.3f", $1-$2}')
cp "$here"/tensors/out_*.diftensor "$here/logs/$label.out.diftensor" 2>/dev/null
sha=$(sha256sum "$here"/tensors/out_*.diftensor | awk '{print $1}')
echo "GATE label=$label exit=$status wall_s=$wall output_sha=$sha"
grep -h "PIPELINE_PROFILE\|CUDA_LAUNCH_TELEMETRY\|RESULT \|SESSION_RESULT" "$log"
