#!/usr/bin/env bash
# 04 - Raw copy of vLLM's /metrics endpoint. Called by 03 before and after each
# concurrency level so the deltas (counters, gauges) bracket each run.
#
# Usage: 04_snapshot_metrics.sh <label>     e.g. 04_snapshot_metrics.sh c8_before
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

label="${1:?usage: 04_snapshot_metrics.sh <label>}"
out="${RESULTS_DIR}/metrics_${label}.txt"

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  printf '# snapshot %s  ts=%s\n' "${label}" "${ts}"
  curl -fsS "${VLLM_BASE_URL}/metrics" 2>/dev/null || printf '# ERROR: could not scrape %s/metrics\n' "${VLLM_BASE_URL}"
} > "${out}"
log "Metrics snapshot -> ${out}"
