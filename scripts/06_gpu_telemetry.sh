#!/usr/bin/env bash
# 06 - GPU telemetry for one vLLM concurrency level.
#
# While running it:
#   - writes a PERFORMANCE+CLOCK snapshot on start and on stop (throttle check),
#   - streams a 1 Hz CSV of DCGM profiling metrics + the coarse nvidia-smi
#     utilisation, so you can see how different the two "utilisation" numbers are.
#
# 03 launches this in the background with a label and sends SIGTERM when the
# benchmark for that level finishes; the SIGTERM writes the "after" snapshot.
#
# Usage: 06_gpu_telemetry.sh <label> [max_seconds]
#
# DCGM_FI_PROF_* are ratios in [0,1]. They need DCGM profiling support and come
# from dcgm-exporter (start the monitoring stack first). DCGM_FI_DEV_GPU_UTIL /
# nvidia-smi utilisation is the coarse "% of time a kernel ran" number.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

label="${1:?usage: 06_gpu_telemetry.sh <label> [max_seconds]}"
max_seconds="${2:-0}"           # 0 = until killed

csv="${RESULTS_DIR}/telemetry_${label}.csv"
perf_before="${RESULTS_DIR}/perf_${label}_before.txt"
perf_after="${RESULTS_DIR}/perf_${label}_after.txt"

snapshot_perf() { nvidia-smi -q -d PERFORMANCE,CLOCK > "$1" 2>&1 || echo "nvidia-smi perf snapshot failed" > "$1"; }

finish() { snapshot_perf "${perf_after}"; log "Telemetry stopped (${label}); perf 'after' -> ${perf_after}"; exit 0; }
trap finish TERM INT

snapshot_perf "${perf_before}"
log "Telemetry started (${label}); perf 'before' -> ${perf_before}; CSV -> ${csv}"

echo "ts_iso,epoch,sm_active,sm_occupancy,tensor_active,dram_active,fp16_active,power_w,temp_c,dcgm_gpu_util,smi_gpu_util,smi_sm_clock_mhz" > "${csv}"

interval="$(awk "BEGIN{print 1/${TELEMETRY_HZ}}")"
end_epoch=0
[[ "${max_seconds}" -gt 0 ]] && end_epoch=$(( $(date +%s) + max_seconds ))

# Last whitespace field of the first line for metric $1 on gpu 0 (shell globbing,
# no regex pitfalls with the '{' in metric lines).
dcgm_val() {
  local line
  while IFS= read -r line; do
    case "${line}" in
      "$1{"*) case "${line}" in *'gpu="0"'*) printf '%s' "${line##* }"; return 0;; esac ;;
    esac
  done <<<"${scrape}"
  return 0   # metric absent -> empty string, never fail under set -e
}

while :; do
  now_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; now_epoch="$(date +%s)"

  scrape="$(curl -fsS --max-time 2 "${DCGM_ENDPOINT}" 2>/dev/null || true)"
  sm_active="$(dcgm_val DCGM_FI_PROF_SM_ACTIVE)"
  sm_occ="$(dcgm_val DCGM_FI_PROF_SM_OCCUPANCY)"
  tensor="$(dcgm_val DCGM_FI_PROF_PIPE_TENSOR_ACTIVE)"
  dram="$(dcgm_val DCGM_FI_PROF_DRAM_ACTIVE)"
  fp16="$(dcgm_val DCGM_FI_PROF_PIPE_FP16_ACTIVE)"
  power="$(dcgm_val DCGM_FI_DEV_POWER_USAGE)"
  temp="$(dcgm_val DCGM_FI_DEV_GPU_TEMP)"
  dcgm_util="$(dcgm_val DCGM_FI_DEV_GPU_UTIL)"

  smi="$(nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv,noheader,nounits -i 0 2>/dev/null || echo ',')"
  smi_util="$(echo "${smi}" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
  smi_clk="$(echo "${smi}"  | awk -F',' '{gsub(/ /,"",$2); print $2}')"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${now_iso}" "${now_epoch}" \
    "${sm_active}" "${sm_occ}" "${tensor}" "${dram}" "${fp16}" \
    "${power}" "${temp}" "${dcgm_util}" "${smi_util}" "${smi_clk}" >> "${csv}"

  [[ "${end_epoch}" -gt 0 && "${now_epoch}" -ge "${end_epoch}" ]] && finish
  sleep "${interval}"
done
