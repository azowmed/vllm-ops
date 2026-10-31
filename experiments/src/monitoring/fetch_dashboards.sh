#!/usr/bin/env bash
# Download the official vLLM and NVIDIA DCGM Grafana dashboards into the
# provisioning folder and rewrite their datasource references to the UID of the
# provisioned Prometheus datasource ("prometheus").
#
# Downloaded dashboards (vllm.json, dcgm.json) are git-ignored; the hand-made
# benchmark_overview.json is committed.
#
#   VLLM_IMAGE and/or VLLM_DASHBOARD_REF select the vLLM version to match.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASH_DIR="${HERE}/grafana/dashboards"
mkdir -p "${DASH_DIR}"

: "${VLLM_IMAGE:=vllm/vllm-openai:v0.27.0}"
# vLLM dashboard is versioned with the code; pin it to the image tag by default.
: "${VLLM_DASHBOARD_REF:=${VLLM_IMAGE##*:}}"
[[ -n "${VLLM_DASHBOARD_REF}" && "${VLLM_DASHBOARD_REF}" != "${VLLM_IMAGE}" ]] || VLLM_DASHBOARD_REF="v0.27.0"

VLLM_URL="https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_DASHBOARD_REF}/examples/online_serving/prometheus_grafana/grafana.json"
DCGM_URL="https://grafana.com/api/dashboards/12239/revisions/latest/download"   # "NVIDIA DCGM Exporter Dashboard"

# Rewrite every datasource reference in a dashboard JSON to uid=prometheus and
# drop the __inputs import block, so file-provisioning binds it automatically.
fixup() {
  local f="$1"
  if command -v python3 >/dev/null 2>&1; then
    python3 -I - "$f" <<'PY'
import json, sys
p = sys.argv[1]
with open(p) as fh:
    d = json.load(fh)
d.pop("__inputs", None)
d.pop("__requires", None)
UID = "prometheus"
def fix(o):
    if isinstance(o, dict):
        if "datasource" in o:
            ds = o["datasource"]
            if isinstance(ds, str) or ds is None:
                o["datasource"] = {"type": "prometheus", "uid": UID}
            elif isinstance(ds, dict):
                ds.setdefault("type", "prometheus")
                ds["uid"] = UID
        for v in o.values():
            fix(v)
    elif isinstance(o, list):
        for v in o:
            fix(v)
fix(d)
with open(p, "w") as fh:
    json.dump(d, fh, indent=2)
PY
  else
    # Fallback: string replacement of the common input variable.
    sed -i.bak 's/${DS_PROMETHEUS}/prometheus/g' "$f" && rm -f "$f.bak"
  fi
}

echo "Fetching vLLM dashboard @ ${VLLM_DASHBOARD_REF} ..."
if curl -fsSL "${VLLM_URL}" -o "${DASH_DIR}/vllm.json"; then
  fixup "${DASH_DIR}/vllm.json"
  echo "  -> ${DASH_DIR}/vllm.json"
else
  echo "  WARN: could not download vLLM dashboard from ${VLLM_URL}" >&2
fi

echo "Fetching NVIDIA DCGM dashboard (id 12239) ..."
if curl -fsSL "${DCGM_URL}" -o "${DASH_DIR}/dcgm.json"; then
  fixup "${DASH_DIR}/dcgm.json"
  echo "  -> ${DASH_DIR}/dcgm.json"
else
  echo "  WARN: could not download DCGM dashboard from ${DCGM_URL}" >&2
fi

echo "Done. (benchmark_overview.json is committed and already in place.)"
