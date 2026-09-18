#!/usr/bin/env bash
set -euo pipefail

model="${MODEL_ID:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}"
canvas="${CANVAS_LENGTH:-32}"
upstream_port="${UPSTREAM_PORT:-8000}"
structured_port="${STRUCTURED_PORT:-8011}"
moe_backend="${MOE_BACKEND:-cutlass}"

cleanup() {
  if [[ -n "${structured_pid:-}" ]]; then kill "$structured_pid" 2>/dev/null || true; fi
  if [[ -n "${upstream_pid:-}" ]]; then kill "$upstream_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

python -m vllm.entrypoints.openai.api_server \
  --model "$model" \
  --served-model-name dgemma \
  --host 0.0.0.0 \
  --port "$upstream_port" \
  --diffusion-config "{\"canvas_length\":${canvas}}" \
  --max-logprobs 32 \
  --enable-prefix-caching \
  --async-scheduling \
  --attention-backend TRITON_ATTN \
  --moe-backend "$moe_backend" \
  --language-model-only \
  --max-num-seqs 32 \
  "$@" &
upstream_pid=$!

until curl --fail --silent "http://127.0.0.1:${upstream_port}/health" >/dev/null; do
  if ! kill -0 "$upstream_pid" 2>/dev/null; then
    wait "$upstream_pid"
  fi
  sleep 2
done

python /opt/vllm/examples/features/diffusion_reads/structured_server.py \
  --upstream "http://127.0.0.1:${upstream_port}" \
  --model dgemma \
  --tokenizer "$model" \
  --canvas "$canvas" \
  --host 0.0.0.0 \
  --port "$structured_port" &
structured_pid=$!

wait -n "$upstream_pid" "$structured_pid"
