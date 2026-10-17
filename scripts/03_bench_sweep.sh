#!/usr/bin/env bash
# 03 - Run the benchmark sweep for one target over a set of concurrency levels.
#
#   TARGET=vllm    (default)  benchmark the local vLLM server
#   TARGET=bedrock            benchmark the Bedrock "mantle" endpoint
#
# Identical workload for both targets: random dataset, fixed seed,
# INPUT_LEN in / OUTPUT_LEN out, prompts-per-level = max(50, 4*concurrency).
# One result JSON per level: results/<target>_c<concurrency>.json
#
# For the vLLM target only, each level is bracketed by:
#   - /metrics snapshots (04) before and after,
#   - 1 Hz GPU telemetry + perf snapshots (06) for the duration of the run.
# Bedrock is remote, so there is no local GPU to watch.
#
# The client is just an HTTP load generator, so we run `vllm bench serve` from
# the vLLM image WITHOUT --gpus (works the same on a GPU box or a tiny VM).
#
# Differences that matter for the comparison:
#   - vLLM: --ignore-eos forces exactly OUTPUT_LEN tokens (clean decode measure).
#   - Bedrock: cannot force output length; we record the ACTUAL tokens generated
#     (--save-detailed) and only TTFT and per-token latency are comparable.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

[[ "${TARGET}" == "vllm" || "${TARGET}" == "bedrock" ]] || die "TARGET must be 'vllm' or 'bedrock' (got '${TARGET}')."
is_vllm=false; [[ "${TARGET}" == "vllm" ]] && is_vllm=true

# --- per-target endpoint config ------------------------------------------
if ${is_vllm}; then
  BASE_URL="${VLLM_BASE_URL}"; ENDPOINT="/v1/chat/completions"
  MODEL_NAME="${SERVED_MODEL_NAME}"; TOKENIZER="${MODEL}"
  EXTRA_BODY="${VLLM_EXTRA_BODY}"; API_KEY="EMPTY"
else
  BASE_URL="${BEDROCK_BASE_URL}"; ENDPOINT="${BEDROCK_ENDPOINT}"
  MODEL_NAME="${BEDROCK_MODEL}"; TOKENIZER="${BEDROCK_TOKENIZER}"
  EXTRA_BODY="${BEDROCK_EXTRA_BODY}"; API_KEY="${OPENAI_API_KEY:-}"
  [[ -n "${API_KEY}" ]] || die "Bedrock target needs OPENAI_API_KEY (a short-lived Bedrock token)."
fi

have docker || die "docker not found."
${is_vllm} && "${SCRIPTS_DIR}/05_collect_env.sh" || true

# --- client metadata + network latency to the target --------------------
zone="${CLOUD_ZONE}"
if [[ -z "${zone}" ]]; then
  zone="$(curl -fsS --max-time 2 'http://169.254.42.42/conf?format=json' 2>/dev/null \
          | sed -n 's/.*"zone"[: ]*"\([^"]*\)".*/\1/p' | head -n1 || true)"
fi
probe_url="${BASE_URL}/v1/models"
auth_args=()
[[ "${TARGET}" == "bedrock" && -n "${API_KEY}" ]] && auth_args=(-H "Authorization: Bearer ${API_KEY}")

net="${RESULTS_DIR}/${TARGET}_netlatency.json"
{
  printf '{\n'
  printf '  "target": "%s",\n' "${TARGET}"
  printf '  "base_url": "%s",\n' "${BASE_URL}"
  printf '  "probe_url": "%s",\n' "${probe_url}"
  printf '  "client_hostname": "%s",\n' "$(hostname)"
  printf '  "cloud_provider": "%s",\n' "${CLOUD_PROVIDER:-unknown}"
  printf '  "cloud_zone": "%s",\n' "${zone}"
  printf '  "samples": [\n'
  min_c=""; min_t=""
  for i in 1 2 3; do
    read -r tc tt code < <(curl -o /dev/null -s --max-time 10 "${auth_args[@]}" \
      -w '%{time_connect} %{time_starttransfer} %{http_code}' "${probe_url}" 2>/dev/null || echo "0 0 000")
    sep=","; [[ "${i}" -eq 3 ]] && sep=""
    printf '    {"time_connect_s": %s, "time_ttfb_s": %s, "http_code": %s}%s\n' "${tc}" "${tt}" "${code}" "${sep}"
    [[ -z "${min_c}" ]] && min_c="${tc}"; awk "BEGIN{exit !(${tc}<${min_c})}" && min_c="${tc}"
    [[ -z "${min_t}" ]] && min_t="${tt}"; awk "BEGIN{exit !(${tt}<${min_t})}" && min_t="${tt}"
  done
  printf '  ],\n'
  printf '  "min_time_connect_s": %s,\n' "${min_c}"
  printf '  "min_time_ttfb_s": %s\n' "${min_t}"
  printf '}\n'
} > "${net}"
log "Network latency to ${TARGET} -> ${net} (connect>=${min_c}s, ttfb>=${min_t}s)"

# --- sweep ----------------------------------------------------------------
log "Sweeping TARGET=${TARGET} over concurrencies: ${CONCURRENCIES}"
for c in ${CONCURRENCIES}; do
  prompts=$(( 4 * c )); (( prompts < 50 )) && prompts=50
  out="${TARGET}_c${c}.json"
  log "=== level concurrency=${c}, prompts=${prompts} -> results/${out} ==="

  tele_pid=""
  if ${is_vllm}; then
    "${SCRIPTS_DIR}/04_snapshot_metrics.sh" "c${c}_before"
    "${SCRIPTS_DIR}/06_gpu_telemetry.sh" "c${c}" &
    tele_pid=$!
  fi

  bench=(
    run --rm --network host
    -e "OPENAI_API_KEY=${API_KEY}"
    -v "${RESULTS_DIR}:/results"
    -v "${HF_CACHE}:/root/.cache/huggingface"
    --entrypoint vllm "${VLLM_IMAGE}"
    bench serve
    --backend openai-chat
    --base-url "${BASE_URL}"
    --endpoint "${ENDPOINT}"
    --model "${MODEL_NAME}"
    --tokenizer "${TOKENIZER}"
    --dataset-name random
    --random-input-len "${INPUT_LEN}"
    --random-output-len "${OUTPUT_LEN}"
    --random-range-ratio 0.0
    --num-prompts "${prompts}"
    --max-concurrency "${c}"
    --seed "${SEED}"
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles "${PERCENTILES}"
    --extra-body "${EXTRA_BODY}"
    --save-result --result-dir /results --result-filename "${out}"
    --metadata "target=${TARGET}" "concurrency=${c}" "input_len=${INPUT_LEN}" "output_len=${OUTPUT_LEN}"
  )
  if ${is_vllm}; then
    bench+=(--ignore-eos)                 # force exactly OUTPUT_LEN tokens
  else
    bench+=(--save-detailed)              # keep actual output_lens + errors (429s)
  fi

  # Don't abort the whole sweep if one level errors out.
  docker "${bench[@]}" || log "WARN: benchmark at concurrency=${c} returned non-zero."

  if [[ -n "${tele_pid}" ]]; then
    kill -TERM "${tele_pid}" 2>/dev/null || true
    wait "${tele_pid}" 2>/dev/null || true
    "${SCRIPTS_DIR}/04_snapshot_metrics.sh" "c${c}_after"
  fi
done

log "Sweep complete. Results in ${RESULTS_DIR}/"
