#!/usr/bin/env bash
# Serve a free open-weight model locally (Colab GPU runtime) through Ollama's OpenAI-compatible API.
#   bash MP4/scripts/setup_ollama.sh            # default qwen3:8b (fits a free T4)
#   bash MP4/scripts/setup_ollama.sh qwen3:14b  # L4 / A100 runtimes
# Then run the pipeline with --config MP4/configs/config.ollama.json [--model <same tag>].
set -euo pipefail
MODEL="${1:-qwen3:8b}"
CTX="${OLLAMA_CONTEXT_LENGTH:-32768}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "[ollama] installing"
  # Colab images lack these; the installer needs them to detect the GPU
  (command -v lspci >/dev/null 2>&1 && command -v zstd >/dev/null 2>&1) || \
    (apt-get update -qq && apt-get install -y -qq pciutils zstd >/dev/null)
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -fsS http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "[ollama] starting server (context length $CTX)"
  OLLAMA_CONTEXT_LENGTH="$CTX" OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
    nohup ollama serve > /tmp/ollama.log 2>&1 &
  for _ in $(seq 1 60); do
    curl -fsS http://localhost:11434/api/version >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -fsS http://localhost:11434/api/version && echo

echo "[ollama] pulling $MODEL"
ollama pull "$MODEL"

echo "[ollama] smoke test (OpenAI-compatible endpoint)"
curl -fsS http://localhost:11434/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK\"}], \"max_tokens\": 20}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:200])'
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true
echo "[ollama] ready: base_url http://localhost:11434/v1, model $MODEL"
