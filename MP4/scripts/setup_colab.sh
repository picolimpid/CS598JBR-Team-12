#!/usr/bin/env bash
# One-time environment setup for the MP4 pipeline on Google Colab / any Ubuntu machine.
#   bash MP4/scripts/setup_colab.sh
# Installs: rustup + nightly-2025-05-13 (clippy, rustfmt, rustc-dev, llvm-tools-preview),
#           the course's newmetrics unsafe-metrics tool, and the Python dependency (requests).
set -euo pipefail

TOOLCHAIN="nightly-2025-05-13"
MP4_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v rustup >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  echo "[setup] installing rustup"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"

echo "[setup] installing toolchain $TOOLCHAIN"
rustup toolchain install "$TOOLCHAIN" --profile minimal \
  --component clippy,rustfmt,rustc-dev,llvm-tools-preview
# newmetrics' wrapper invokes plain `rustc` to compare sysroots, so make the pinned nightly the default.
rustup default "$TOOLCHAIN"

cargo +"$TOOLCHAIN" --version
cargo +"$TOOLCHAIN" clippy --version
rustfmt +"$TOOLCHAIN" --version

echo "[setup] building the newmetrics unsafe-metrics tool"
pushd "$MP4_DIR/newmetrics" >/dev/null
if ! cargo install --debug --locked --path . --force; then
  echo "[setup] --locked install failed (no Cargo.lock shipped?) - retrying without --locked"
  cargo install --debug --path . --force
fi
popd >/dev/null
command -v cargo-newmetrics

python3 -m pip install -q "requests>=2.28"
echo "[setup] done. Remember: export OPENROUTER_API_KEY=... (or the key named in your config)"
