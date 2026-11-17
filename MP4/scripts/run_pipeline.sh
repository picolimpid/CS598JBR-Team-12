#!/usr/bin/env bash
# Reproduce the agentic translation of one assigned project.
#   bash MP4/scripts/run_pipeline.sh 1            # project_1 (satc)
#   bash MP4/scripts/run_pipeline.sh 2 --resume   # continue project_2 (jccc) after a quota stop
#   CONFIG=MP4/configs/config.gemini.json bash MP4/scripts/run_pipeline.sh 1
set -euo pipefail
MP4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
PROJECT="${1:?usage: run_pipeline.sh <1|2|name> [extra pipeline args]}"; shift
CONFIG="${CONFIG:-$MP4_DIR/configs/config.json}"
python3 "$MP4_DIR/src/pipeline.py" --project "$PROJECT" --config "$CONFIG" "$@"
