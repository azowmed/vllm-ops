#!/usr/bin/env bash
# Shared defaults + helpers, sourced by every scripts/*.sh.
# Not meant to be executed directly.
#
# Every value below can be overridden from the environment or from a .env file
# at the repo root (loaded here, never committed).
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
: "${RESULTS_DIR:=${REPO_ROOT}/results}"

# Load .env (git-ignored) if present so overrides/secrets stay out of git.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

# ---- vLLM server / image -------------------------------------------------
: "${VLLM_IMAGE:=vllm/vllm-openai:v0.27.0}"
: "${MODEL:=Qwen/Qwen3-32B-FP8}"
: "${SERVED_MODEL_NAME:=${MODEL}}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEM_UTIL:=0.90}"
# KV-cache dtype. 'auto' = BF16 KV (vLLM default, even with FP8 weights).
# On an L40S (Ada, SM 8.9) 'fp8' is supported and ~halves KV bytes/token, so it
# roughly doubles the small ~6-7 GB KV budget left after the FP8 weights. See
# PREDICTIONS.md for both scenarios.
: "${KV_CACHE_DTYPE:=auto}"
: "${VLLM_PORT:=8000}"
: "${VLLM_HOST_BIND:=127.0.0.1}"      # host interface docker publishes to
: "${VLLM_CONTAINER_NAME:=vllm}"
: "${VLLM_EXTRA_ARGS:=}"              # extra flags appended to `vllm serve`
: "${HF_TOKEN:=}"
: "${HF_CACHE:=${HOME}/.cache/huggingface}"

# ---- benchmark workload --------------------------------------------------
: "${TARGET:=vllm}"                   # vllm | bedrock
# Brackets the L40S KV knee (~20 full requests at BF16 KV): 1/8/16 fit, 32/64 preempt.
: "${CONCURRENCIES:=1 8 16 32 64}"
: "${INPUT_LEN:=1024}"
: "${OUTPUT_LEN:=256}"
: "${SEED:=42}"
: "${PERCENTILES:=50,95,99}"

# vLLM target endpoint (served over loopback only)
: "${VLLM_BASE_URL:=http://127.0.0.1:${VLLM_PORT}}"

# Bedrock target (OpenAI-compatible "mantle" gateway)
: "${BEDROCK_BASE_URL:=https://bedrock-mantle.eu-north-1.api.aws}"
: "${BEDROCK_MODEL:=qwen.qwen3-32b}"
: "${BEDROCK_ENDPOINT:=/v1/chat/completions}"
: "${BEDROCK_TOKENIZER:=Qwen/Qwen3-32B}"
: "${AWS_REGION:=eu-north-1}"

# ---- "thinking" control --------------------------------------------------
# Qwen3 enables a reasoning ("thinking") phase by default. We disable it on both
# targets so the measured work is comparable. The *mechanism* differs per
# backend (see README "Disabling Qwen3 thinking"); the JSON is injected verbatim
# into every request via `vllm bench serve --extra-body`.
# JSON contains '}' so it cannot go through the ${VAR:=default} form safely.
if [[ -z "${VLLM_EXTRA_BODY:-}" ]]; then
  VLLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
fi
if [[ -z "${BEDROCK_EXTRA_BODY:-}" ]]; then
  BEDROCK_EXTRA_BODY='{"reasoning_effort": "none"}'
fi

# ---- telemetry -----------------------------------------------------------
: "${DCGM_ENDPOINT:=http://127.0.0.1:9400/metrics}"
: "${TELEMETRY_HZ:=1}"

# ---- cloud metadata ------------------------------------------------------
: "${CLOUD_PROVIDER:=}"
: "${CLOUD_ZONE:=}"

log()  { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

mkdir -p "${RESULTS_DIR}"
