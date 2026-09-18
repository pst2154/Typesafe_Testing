#!/usr/bin/env bash
set -euo pipefail
cleanup() {
  kill "${runtime_pid:-}" "${gateway_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
/usr/local/bin/run-structured-diffusiongemma "$@" &
runtime_pid=$!
until curl -fsS "http://127.0.0.1:${STRUCTURED_PORT:-8011}/health" >/dev/null; do
  if ! kill -0 "$runtime_pid" 2>/dev/null; then wait "$runtime_pid"; exit 1; fi
  sleep 2
done
export DIFFUSION_URLS="${DIFFUSION_URLS:-http://127.0.0.1:${STRUCTURED_PORT:-8011}/v1/chat/completions}"
export RAW_URLS="${RAW_URLS:-http://127.0.0.1:${UPSTREAM_PORT:-8000}/v1/chat/completions}"
python /opt/decision-lab/gateway.py &
gateway_pid=$!
wait -n "$runtime_pid" "$gateway_pid"
