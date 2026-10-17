#!/usr/bin/env bash
# 05 - Capture the environment the measurements ran in, into results/env.json:
# GPU model, driver, driver CUDA version, vLLM version, pinned image digest,
# UTC date, cloud provider and zone.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

gpu_name="";   gpu_mem="";   driver=""; driver_cuda=""
if have nvidia-smi; then
  IFS=',' read -r gpu_name gpu_mem driver < <(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1)
  gpu_name="$(echo "${gpu_name}" | sed 's/^ *//;s/ *$//')"
  gpu_mem="$(echo "${gpu_mem}" | sed 's/^ *//;s/ *$//')"
  driver="$(echo "${driver}" | sed 's/^ *//;s/ *$//')"
  driver_cuda="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
fi

vllm_version=""
if have docker; then
  vllm_version="$(docker run --rm --entrypoint vllm "${VLLM_IMAGE}" --version 2>/dev/null | head -n1 | tr -d '\r')"
fi

image_digest=""
if have docker; then
  image_digest="$(docker inspect --format '{{index .RepoDigests 0}}' "${VLLM_IMAGE}" 2>/dev/null || true)"
fi

# Cloud zone: env override wins; otherwise try the Scaleway metadata endpoint.
zone="${CLOUD_ZONE}"
if [[ -z "${zone}" ]]; then
  zone="$(curl -fsS --max-time 2 'http://169.254.42.42/conf?format=json' 2>/dev/null \
          | sed -n 's/.*"zone"[: ]*"\([^"]*\)".*/\1/p' | head -n1 || true)"
fi
provider="${CLOUD_PROVIDER:-unknown}"

jstr() { [[ -n "$1" ]] && printf '"%s"' "$1" || printf 'null'; }

cat > "${RESULTS_DIR}/env.json" <<JSON
{
  "date_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "$(hostname)",
  "cloud_provider": $(jstr "${provider}"),
  "cloud_zone": $(jstr "${zone}"),
  "gpu_name": $(jstr "${gpu_name}"),
  "gpu_memory_total": $(jstr "${gpu_mem}"),
  "driver_version": $(jstr "${driver}"),
  "driver_cuda_version": $(jstr "${driver_cuda}"),
  "vllm_image": $(jstr "${VLLM_IMAGE}"),
  "vllm_image_digest": $(jstr "${image_digest}"),
  "vllm_version": $(jstr "${vllm_version}"),
  "model": $(jstr "${MODEL}"),
  "max_model_len": ${MAX_MODEL_LEN},
  "gpu_memory_utilization": ${GPU_MEM_UTIL},
  "kv_cache_dtype": $(jstr "${KV_CACHE_DTYPE}")
}
JSON

log "Wrote ${RESULTS_DIR}/env.json"
cat "${RESULTS_DIR}/env.json" >&2
