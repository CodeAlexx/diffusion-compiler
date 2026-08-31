#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 START_BLOCK END_BLOCK CREATOR_DIR CHECKPOINT ARTIFACT_DIR BUILD_DIR" >&2
  exit 2
fi

start_block=$1
end_block=$2
creator_dir=$3
checkpoint=$4
artifact_dir=$5
build_dir=$6
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if (( start_block < 1 || end_block < start_block || end_block > 27 )); then
  echo "trajectory range must satisfy 1 <= START <= END <= 27" >&2
  exit 2
fi

mkdir -p "$artifact_dir"
for ((block=start_block; block<=end_block; ++block)); do
  previous=$((block - 1))
  creator_previous="$artifact_dir/creator-block${previous}.safetensors"
  native_previous="$artifact_dir/native-block${previous}.safetensors"
  creator_current="$artifact_dir/creator-block${block}.safetensors"
  native_current="$artifact_dir/native-block${block}.safetensors"
  echo "KREA2_TRAJECTORY creator block=$block input=$creator_previous"
  flock /tmp/dc-gpu.lock python3 "$repo_dir/scripts/krea2_creator_block_fixture.py" \
    --creator "$creator_dir" \
    --checkpoint "$checkpoint" \
    --input-fixture "$creator_previous" \
    --output "$creator_current" \
    --report "$artifact_dir/creator-block${block}-report.json" \
    --block "$block" --final-only
  echo "KREA2_TRAJECTORY native block=$block input=$native_previous"
  flock /tmp/dc-gpu.lock "$build_dir/difkrea2block" \
    --checkpoint "$checkpoint" \
    --fixture "$creator_current" \
    --sequence-override "$native_previous::final_output" \
    --output "$native_current" \
    --report "$artifact_dir/native-block${block}-report.json" \
    --diffir "$artifact_dir/block${block}.diffir" \
    --block "$block" --final-only
done
