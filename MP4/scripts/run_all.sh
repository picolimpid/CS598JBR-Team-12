#!/usr/bin/env bash
# Run the pipeline on both assigned projects, then the independent final evaluation.
#   bash MP4/scripts/run_all.sh
set -euo pipefail
MP4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
CONFIG="${CONFIG:-$MP4_DIR/configs/config.json}"
for p in 1 2; do
  python3 "$MP4_DIR/src/pipeline.py" --project "$p" --config "$CONFIG" "$@" || echo "[run_all] project $p exited with $?"
done
python3 "$MP4_DIR/src/evaluate.py" --config "$CONFIG"
