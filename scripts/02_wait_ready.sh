#!/usr/bin/env bash
# 02 - Wait until vLLM answers /v1/models, measure cold-start time, and extract
# the KV-cache size (in tokens) and announced maximum concurrency from the log.
#
# Writes results/startup.json and results/startup_lines.txt.
#
# "Cold start" = container start (docker StartedAt) -> first successful
# /v1/models. On the very first run it INCLUDES the model download; re-run after
# the weights are cached for a pure load-time figure.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

TIMEOUT="${TIMEOUT:-1800}"       # seconds to wait for readiness
POLL_EVERY="${POLL_EVERY:-2}"

# Container start time -> epoch (GNU date on the Linux instance).
started_raw="$(docker inspect -f '{{.State.StartedAt}}' "${VLLM_CONTAINER_NAME}" 2>/dev/null || true)"
if [[ -n "${started_raw}" ]]; then
  start_epoch="$(date -d "${started_raw}" +%s 2>/dev/null || echo "")"
else
  start_epoch=""
fi
[[ -n "${start_epoch}" ]] || { log "WARN: could not read container StartedAt; timing from now."; start_epoch="$(date +%s)"; }

log "Waiting up to ${TIMEOUT}s for ${VLLM_BASE_URL}/v1/models ..."
deadline=$(( $(date +%s) + TIMEOUT ))
ready_epoch=""
while :; do
  if curl -fsS -o /dev/null "${VLLM_BASE_URL}/v1/models" 2>/dev/null; then
    ready_epoch="$(date +%s)"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "${VLLM_CONTAINER_NAME}"; then
    die "Container '${VLLM_CONTAINER_NAME}' is no longer running. Check: docker logs ${VLLM_CONTAINER_NAME}"
  fi
  [[ "$(date +%s)" -lt "${deadline}" ]] || die "Timed out after ${TIMEOUT}s waiting for readiness."
  sleep "${POLL_EVERY}"
done
cold_start=$(( ready_epoch - start_epoch ))
log "Ready after ${cold_start}s (cold start)."

# --- Parse startup facts from the container log --------------------------
logs="$(docker logs "${VLLM_CONTAINER_NAME}" 2>&1 || true)"

kv_line="$(grep -iE 'GPU KV cache size|KV cache size|# GPU blocks:' <<<"${logs}" | tail -n1 || true)"
conc_line="$(grep -iE 'Maximum concurrency for' <<<"${logs}" | tail -n1 || true)"
{ printf '%s\n' "${kv_line}"; printf '%s\n' "${conc_line}"; } > "${RESULTS_DIR}/startup_lines.txt"

# "GPU KV cache size: 2,236,416 tokens" -> 2236416
kv_tokens="$(sed -n 's/.*[Cc]ache size:[^0-9]*\([0-9][0-9,]*\).*/\1/p' <<<"${kv_line}" | tr -d ',' | head -n1)"
# "Maximum concurrency for 8192 tokens per request: 272.95x" -> 272.95 and 8192
max_conc="$(sed -n 's/.*: *\([0-9][0-9.]*\)x.*/\1/p' <<<"${conc_line}" | head -n1)"
conc_ctx="$(sed -n 's/.*for *\([0-9][0-9,]*\) *tokens.*/\1/p' <<<"${conc_line}" | tr -d ',' | head -n1)"

json_num() { [[ -n "$1" ]] && printf '%s' "$1" || printf 'null'; }

cat > "${RESULTS_DIR}/startup.json" <<JSON
{
  "model": "${MODEL}",
  "served_model_name": "${SERVED_MODEL_NAME}",
  "max_model_len": ${MAX_MODEL_LEN},
  "gpu_memory_utilization": ${GPU_MEM_UTIL},
  "kv_cache_dtype": "${KV_CACHE_DTYPE}",
  "cold_start_seconds": ${cold_start},
  "kv_cache_tokens": $(json_num "${kv_tokens}"),
  "max_concurrency": $(json_num "${max_conc}"),
  "max_concurrency_ctx_tokens": $(json_num "${conc_ctx}"),
  "container_started_at": "${started_raw}"
}
JSON

log "Wrote ${RESULTS_DIR}/startup.json"
cat "${RESULTS_DIR}/startup.json" >&2
[[ -n "${kv_tokens}" ]] || log "WARN: could not parse KV cache size from log; see startup_lines.txt and adjust the regex for your vLLM version."
