#!/usr/bin/env bash
# LAPTOP-side driver for a fresh GPU instance. The instance never needs GitHub
# access: the repo is pushed to it with rsync over SSH, and results come back
# the same way, to be committed from the laptop.
#
#   RUN=2026-11-10_fp8-kv ./experiments/src/instance.sh setup <ip>   # sync, .env, check, monitoring, vLLM up + ready
#   ./experiments/src/instance.sh shell <ip>     # SSH in with the vLLM/Grafana/Prometheus tunnel
#   ./experiments/src/instance.sh fetch <ip>     # Prometheus snapshot + pull experiments/ back
#
# .env: the repo-root .env on the laptop is used if present (git-ignored), else
# experiments/src/env.example. GPU / RUN / KV_CACHE_DTYPE / CONCURRENCIES given in
# the environment override it. RUN must be set one way or the other.
#
# SSH: user root, key $SSH_KEY (default ~/.ssh/gpu_hosts). A passphrase is asked
# ONCE: the key is loaded into ssh-agent for $SSH_KEY_TTL, and every connection
# then runs in BatchMode (never prompts). Scaleway reuses IPs across instances,
# so `setup` forgets the old host key for that IP first.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SSH_KEY:=${HOME}/.ssh/gpu_hosts}"
: "${SSH_USER:=root}"
: "${REMOTE_DIR:=vllm-ops}"            # relative to the remote home
: "${SSH_KEY_TTL:=8h}"                # how long ssh-agent keeps the unlocked key

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

cmd="${1:-}"; ip="${2:-}"
[[ -n "${cmd}" && -n "${ip}" ]] || die "usage: $0 setup|shell|fetch <instance-ip>"

ssh_opts=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=10 -o BatchMode=yes)
[[ -f "${SSH_KEY}" ]] && ssh_opts+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes)
target="${SSH_USER}@${ip}"
rsh() { ssh "${ssh_opts[@]}" "${target}" "$@"; }

cleanup=()
trap 'for c in "${cleanup[@]+"${cleanup[@]}"}"; do eval "${c}"; done' EXIT

# Unlock $SSH_KEY once into ssh-agent (starting a temporary agent if none runs).
unlock_key() {
  [[ -f "${SSH_KEY}" ]] || return 0
  local pub="${SSH_KEY}.pub" fp
  [[ -f "${pub}" ]] || pub="${SSH_KEY}"
  fp="$(ssh-keygen -lf "${pub}" 2>/dev/null | awk '{print $2}')"
  if [[ -z "${SSH_AUTH_SOCK:-}" ]] || { ssh-add -l >/dev/null 2>&1; [[ $? -eq 2 ]]; }; then
    eval "$(ssh-agent -s)" >/dev/null
    cleanup+=('ssh-agent -k >/dev/null 2>&1')
  fi
  if [[ -n "${fp}" ]] && ssh-add -l 2>/dev/null | grep -qF "${fp}"; then
    return 0                                        # already unlocked
  fi
  log "Unlocking ${SSH_KEY} (asked once, kept in ssh-agent for ${SSH_KEY_TTL})"
  ssh-add -t "${SSH_KEY_TTL}" "${SSH_KEY}" || die "Could not unlock ${SSH_KEY}."
}
unlock_key

# Build the .env to install: laptop .env (or env.example) + overrides from the environment.
make_env() {
  local base="${REPO}/.env" out="$1" k
  [[ -f "${base}" ]] || base="${REPO}/experiments/src/env.example"
  cp "${base}" "${out}"
  for k in GPU RUN KV_CACHE_DTYPE CONCURRENCIES; do
    [[ -n "${!k:-}" ]] || continue
    grep -v "^${k}=" "${out}" > "${out}.tmp" && mv "${out}.tmp" "${out}"
    printf '%s="%s"\n' "${k}" "${!k}" >> "${out}"
  done
  if [[ "${base}" == *env.example && -z "${RUN:-}" ]]; then
    die "No laptop .env, so RUN must be given (e.g. RUN=$(date +%F)_bf16-kv $0 setup ${ip})."
  fi
  log "Using .env from ${base#"${REPO}/"} (overrides: $(for k in GPU RUN KV_CACHE_DTYPE CONCURRENCIES; do [[ -n "${!k:-}" ]] && printf '%s ' "${k}"; done))"
}

case "${cmd}" in
setup)
  # Build .env first: a missing RUN fails here, before touching the instance.
  envfile="$(mktemp)"; cleanup+=("rm -f '${envfile}'")
  make_env "${envfile}"
  ssh-keygen -R "${ip}" >/dev/null 2>&1 || true
  log "Waiting for SSH on ${ip} (up to 5 min)..."
  for i in $(seq 1 60); do
    if out="$(rsh true 2>&1)"; then break; fi
    case "${out}" in   # the host answered but refused us: retrying won't help
      *"Permission denied"*) die "SSH to ${target} refused the key ${SSH_KEY}:
${out}";;
    esac
    (( i == 60 )) && die "SSH to ${target} not reachable after 5 min:
${out}"
    (( i % 6 == 0 )) && log "  still waiting ($(( i * 5 ))s): ${out##*$'\n'}"
    sleep 5
  done

  log "Syncing the repo to ${target}:${REMOTE_DIR}/"
  rsh 'command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null)'
  rsync -az -e "ssh ${ssh_opts[*]}" \
    --exclude .env --exclude .DS_Store --exclude .claude/ --exclude __pycache__/ \
    --exclude .venv/ --exclude experiments/src/monitoring/.offline-run/ \
    "${REPO}/" "${target}:${REMOTE_DIR}/"

  scp -q "${ssh_opts[@]}" "${envfile}" "${target}:${REMOTE_DIR}/.env"
  rsh "grep -E '^(GPU|RUN)=' ${REMOTE_DIR}/.env" >&2

  # 02 waits for the model download on a fresh instance (can take tens of minutes).
  # If SSH drops meanwhile, vLLM keeps running: re-run 02 from `shell`.
  rsh "cd ${REMOTE_DIR} && set -e
    bash experiments/src/scripts/00_check_instance.sh
    (cd experiments/src/monitoring && docker compose --profile gpu up -d)
    bash experiments/src/scripts/01_start_vllm.sh
    bash experiments/src/scripts/02_wait_ready.sh"

  log "Instance ready. Next:"
  log "  $0 shell ${ip}                       # then: bash experiments/src/scripts/03_bench_sweep.sh"
  log "  $0 fetch ${ip}                       # at the end, before deleting the instance"
  ;;

shell)
  exec ssh "${ssh_opts[@]}" -t \
    -L 8000:localhost:8000 -L 3000:localhost:3000 -L 9090:localhost:9090 \
    "${target}" "cd ${REMOTE_DIR} && exec bash -l"
  ;;

fetch)
  # Snapshot Prometheus into the run (runbook step 6), then pull experiments/ back.
  rsh "cd ${REMOTE_DIR} && set -e && set -a && source .env && set +a
    if docker volume inspect monitoring_prometheus-data >/dev/null 2>&1; then
      (cd experiments/src/monitoring && docker compose stop prometheus)
      docker run --rm -v monitoring_prometheus-data:/src:ro -v \"\$PWD/experiments/\$GPU/\$RUN\":/dst \
        alpine sh -c 'rm -rf /dst/prometheus-snapshot && cp -a /src/data /dst/prometheus-snapshot'
      (cd experiments/src/monitoring && docker compose start prometheus)
      echo \"Prometheus snapshot -> experiments/\$GPU/\$RUN/prometheus-snapshot\"
    else
      echo 'No Prometheus volume: skipping the snapshot.'
    fi" >&2
  log "Pulling experiments/ back (nothing local is deleted)"
  rsync -az -e "ssh ${ssh_opts[*]}" --exclude /src/ \
    "${target}:${REMOTE_DIR}/experiments/" "${REPO}/experiments/"
  run="$(rsh "cd ${REMOTE_DIR} && set -a && . ./.env && echo \"\$GPU/\$RUN\"")"
  log "Done. Summarize + commit on the laptop:"
  log "  python experiments/src/analysis/summarize.py --run-dir experiments/${run}"
  log "Then DELETE the instance in the Scaleway console."
  ;;

*) die "unknown command '${cmd}' (setup|shell|fetch)";;
esac
