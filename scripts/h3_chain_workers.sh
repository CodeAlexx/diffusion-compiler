#!/usr/bin/env bash
# Manage the persistent H3 chain workers started by
# h3_fl2va_convrot_int8_prompt_to_mp4.sh in serve mode (H3_SERVE_DIR).
#   h3_chain_workers.sh status DIR   -> which sockets are live
#   h3_chain_workers.sh stop DIR     -> --shutdown both workers
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 status|stop SERVE_DIR" >&2; exit 2; }
ACTION="$1"; DIR="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${DIF_BUILD:-$ROOT/build}"
case "$ACTION" in
  status)
    for w in denoiser vae; do
      if [[ -S "$DIR/$w.sock" ]]; then echo "$w: live ($DIR/$w.sock)"; else echo "$w: down"; fi
    done ;;
  stop)
    [[ -S "$DIR/denoiser.sock" ]] && "$BUILD/difh3infer" --connect "$DIR/denoiser.sock" --shutdown || true
    [[ -S "$DIR/vae.sock" ]] && "$BUILD/difvaedecode" --connect "$DIR/vae.sock" --shutdown || true
    sleep 0.5
    for w in denoiser vae; do [[ -S "$DIR/$w.sock" ]] && echo "$w socket still present" || echo "$w: stopped"; done ;;
  *) echo "unknown action $ACTION" >&2; exit 2 ;;
esac
