#!/usr/bin/env bash
# Seed the writable working copy used by docker-compose.offline.yml from the
# COMMITTED TSDB snapshot. Prometheus replays + compacts its TSDB, so it must
# write to this throwaway copy (.offline-run/, git-ignored), never into the
# committed archive under snapshots/.
#
# Safe to re-run: it resets .offline-run/ back to the pristine committed data.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="snapshots/prometheus-data/data"
DST=".offline-run/prometheus"

[[ -d "${SRC}" ]] || { echo "No committed snapshot at ${SRC}" >&2; exit 1; }
rm -rf "${DST}"
mkdir -p "$(dirname "${DST}")"
cp -R "${SRC}" "${DST}"
echo "Seeded ${DST} from ${SRC}"
