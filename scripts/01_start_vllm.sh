#!/usr/bin/env bash
# 01 - Start the vLLM OpenAI-compatible server in Docker.
#
# The container port is published ONLY to 127.0.0.1 so the API is never exposed
# on the public interface (the default Scaleway firewall may allow inbound). Use
# the SSH tunnel from the README to reach it from your laptop.
#
# Parameterised entirely by env vars (see env.example / _common.sh):
#   VLLM_IMAGE MODEL SERVED_MODEL_NAME MAX_MODEL_LEN GPU_MEM_UTIL VLLM_PORT
#   VLLM_EXTRA_ARGS HF_TOKEN HF_CACHE
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# Remove any previous container with the same name.
if docker ps -a --format '{{.Names}}' | grep -qx "${VLLM_CONTAINER_NAME}"; then
  log "Removing existing container '${VLLM_CONTAINER_NAME}'"
  docker rm -f "${VLLM_CONTAINER_NAME}" >/dev/null
fi

mkdir -p "${HF_CACHE}"

# Assemble docker args as an array (robust quoting).
docker_args=(
  run -d
  --name "${VLLM_CONTAINER_NAME}"
  --gpus all
  --ipc=host                                   # vLLM needs a large /dev/shm
  -p "${VLLM_HOST_BIND}:${VLLM_PORT}:8000"      # loopback-only publish
  -v "${HF_CACHE}:/root/.cache/huggingface"
  --restart no
)
[[ -n "${HF_TOKEN}" ]] && docker_args+=(-e "HF_TOKEN=${HF_TOKEN}")

# Flags passed to the image's `vllm serve` entrypoint.
serve_args=(
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0 --port 8000                    # inside the container only
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"          # auto=BF16; fp8 ~doubles KV capacity on L40S
  # Hard-disable Qwen3 "thinking" for every request on the server side.
  --default-chat-template-kwargs '{"enable_thinking": false}'
)
# shellcheck disable=SC2206  # intentional word-splitting of extra args
[[ -n "${VLLM_EXTRA_ARGS}" ]] && serve_args+=(${VLLM_EXTRA_ARGS})

log "Starting vLLM: ${MODEL} (max_model_len=${MAX_MODEL_LEN}, gpu_mem_util=${GPU_MEM_UTIL}, kv_cache_dtype=${KV_CACHE_DTYPE})"
docker "${docker_args[@]}" "${VLLM_IMAGE}" "${serve_args[@]}"

log "Container started. Follow the boot with:  docker logs -f ${VLLM_CONTAINER_NAME}"
log "Then run scripts/02_wait_ready.sh to wait for readiness and capture startup facts."
