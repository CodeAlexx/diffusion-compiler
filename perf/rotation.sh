#!/usr/bin/env bash
# Controlled A/B/A rotation. Sequential by construction (memory safety).
set -uo pipefail
P="$1"; EV="${2:-6}"; R=/home/alex/dc-perf/perf
run() { flock /tmp/dc-gpu.lock -c "$R/run_arm.sh $1 $P $EV ${2:-}" >> "$P/rotation.log" 2>&1; echo "done $1"; }
run B-keeppages   "--streamed-keep-pages"
run A1-baseline   ""
run C-threads4    "--streamed-stage-threads 4"
run A2-baseline   ""
run D-ring4depth3 "--streamed-staging-buffers 4 --streamed-prefetch-depth 3"
run A3-baseline   ""
echo "ROTATION COMPLETE"
