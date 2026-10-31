#!/usr/bin/env bash
# 00 - Pre-flight checks on the GPU instance.
#   - GPU visible to the host (nvidia-smi)
#   - Docker installed and the NVIDIA runtime usable (docker info + a GPU probe)
#   - enough free disk for the image + FP8 weights
#   - pull the pinned vLLM image
#   - verify the driver's max CUDA version >= the CUDA version the image's torch
#     was built against (stop with a clear message otherwise)
#
# Safe to run repeatedly. Touches nothing except the Docker image cache.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

MIN_FREE_GB="${MIN_FREE_GB:-80}"   # image (~10-15 GB) + Qwen3-32B-FP8 (~33 GB) + headroom

# --- GPU visible ----------------------------------------------------------
have nvidia-smi || die "nvidia-smi not found. This must run on the GPU instance."
log "GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# Driver's MAX supported CUDA version, from the nvidia-smi header line
# "... CUDA Version: 12.4 ...".
DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1)"
[[ -n "${DRIVER_CUDA}" ]] || die "Could not parse the driver's CUDA version from nvidia-smi."
log "Driver max CUDA version: ${DRIVER_CUDA}"

# --- Docker + NVIDIA runtime ---------------------------------------------
have docker || die "docker not found."
docker info >/dev/null 2>&1 || die "Cannot talk to the Docker daemon (permissions? is it running?)."
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
  log "Docker 'nvidia' runtime: present"
else
  log "WARN: 'nvidia' runtime not listed by 'docker info'. Relying on '--gpus all' (NVIDIA Container Toolkit)."
fi

# --- Disk space -----------------------------------------------------------
FREE_GB="$(df -Pk /var/lib/docker 2>/dev/null | awk 'NR==2{printf "%d", $4/1024/1024}')"
[[ -n "${FREE_GB}" ]] || FREE_GB="$(df -Pk / | awk 'NR==2{printf "%d", $4/1024/1024}')"
log "Free disk on Docker root: ${FREE_GB} GB (need >= ${MIN_FREE_GB} GB)"
[[ "${FREE_GB}" -ge "${MIN_FREE_GB}" ]] || die "Not enough free disk: ${FREE_GB} GB < ${MIN_FREE_GB} GB."

# --- Pull the pinned image -----------------------------------------------
log "Pulling ${VLLM_IMAGE} (this can take a few minutes)..."
docker pull "${VLLM_IMAGE}"

# --- GPU reachable from inside a container --------------------------------
# The image entrypoint is `vllm serve`, so a bare
#   docker run ... "${VLLM_IMAGE}" nvidia-smi -L
# would run `vllm serve nvidia-smi -L` (nvidia-smi treated as the model), not
# nvidia-smi. Override the entrypoint so `-L` reaches nvidia-smi itself.
log "Probing the GPU from inside a container..."
if probe_out="$(docker run --rm --gpus all --entrypoint nvidia-smi "${VLLM_IMAGE}" -L 2>&1)"; then
  printf '%s\n' "${probe_out}"
elif printf '%s\n' "${probe_out}" | grep -Eqi 'could not select device driver|nvidia-container|capabilities: \[\[gpu\]\]|unknown runtime|response from daemon.*(gpu|nvidia|runtime|driver)'; then
  die "'--gpus all' failed: the NVIDIA Container Toolkit is not wired into Docker.
${probe_out}"
else
  # Not a Docker GPU/runtime error (e.g. nvidia-smi ran but found no device, or
  # some other failure): surface what actually happened instead of blaming the toolkit.
  die "GPU probe failed:
${probe_out}"
fi

# --- CUDA compatibility: image torch CUDA <= driver max CUDA --------------
IMAGE_CUDA="$(docker run --rm --gpus all --entrypoint python3 "${VLLM_IMAGE}" \
  -c 'import torch; print(torch.version.cuda)' 2>/dev/null | tr -d '[:space:]')"
[[ -n "${IMAGE_CUDA}" ]] || die "Could not read torch.version.cuda from the image."
log "Image was built against CUDA ${IMAGE_CUDA}"

# Highest of the two via version sort; if that is the image version and the two
# differ, the image needs a newer driver than this host provides.
HIGHEST="$(printf '%s\n%s\n' "${IMAGE_CUDA}" "${DRIVER_CUDA}" | sort -V | tail -n1)"
if [[ "${HIGHEST}" == "${IMAGE_CUDA}" && "${IMAGE_CUDA}" != "${DRIVER_CUDA}" ]]; then
  die "CUDA mismatch: image needs CUDA ${IMAGE_CUDA} but the driver supports at most ${DRIVER_CUDA}.
       Use a vLLM image built for CUDA <= ${DRIVER_CUDA}, or update the NVIDIA driver."
fi
log "CUDA check OK: image ${IMAGE_CUDA} <= driver ${DRIVER_CUDA}"

log "All pre-flight checks passed."
