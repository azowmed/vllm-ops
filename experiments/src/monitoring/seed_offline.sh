#!/usr/bin/env bash
# Seed the writable working copy used by docker-compose.offline.yml from a run's
# COMMITTED TSDB snapshot (experiments/<gpu>/<run>/prometheus-snapshot).
# Prometheus replays + compacts its TSDB, so it must write to this throwaway copy
# (.offline-run/, git-ignored), never into the committed archive.
#
#   ./seed_offline.sh ../../h100/2026-10-20_bf16-kv
#
# Safe to re-run: it resets .offline-run/ to the pristine snapshot of that run.
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 <run-dir>   (e.g. ../../h100/2026-10-20_bf16-kv)" >&2; exit 1; }
SRC="$(cd "$1" && pwd)/prometheus-snapshot"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST=".offline-run/prometheus"

[[ -d "${SRC}" ]] || { echo "No snapshot at ${SRC}" >&2; exit 1; }
rm -rf "${DST}"
mkdir -p "$(dirname "${DST}")"
cp -R "${SRC}" "${DST}"
echo "Seeded ${DST} from ${SRC}"
