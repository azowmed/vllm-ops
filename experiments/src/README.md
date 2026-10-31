# experiments/src: runbook

This code serves Qwen3-32B-FP8 with vLLM on a rented GPU, monitors the server
and the GPU under load, and benchmarks it against the same model on Amazon
Bedrock. The load generator is vLLM's own `vllm bench serve`, so both targets
are driven by the same client, dataset and tokenizer.

> All CLI flags, metric names and the KV-cache math were checked against
> **vLLM v0.27.0** and the Qwen3-32B `config.json`. If you pin another image,
> re-run `00_check_instance.sh` and re-check metric names (see
> [below](#checking-vllm-metric-names)).

All commands are run **from the repo root**. Each run writes to
`experiments/<GPU>/<RUN>/`, where `GPU` and `RUN` are set in `.env`.

## Layout

```
src/
├── env.example                copy to <repo root>/.env (git-ignored)
├── scripts/
│   ├── _common.sh             shared defaults, .env loader, run folder (GPU/RUN)
│   ├── 00_check_instance.sh   GPU / Docker / NVIDIA runtime / disk / CUDA-compat / image pull
│   ├── 01_start_vllm.sh       docker run vLLM, published on loopback only
│   ├── 02_wait_ready.sh       wait for /v1/models; cold start, KV cache and max concurrency from the log
│   ├── 03_bench_sweep.sh      TARGET=vllm|bedrock × concurrency sweep; telemetry + snapshots
│   ├── 04_snapshot_metrics.sh raw /metrics before and after each level
│   ├── 05_collect_env.sh      GPU, driver, CUDA, vLLM version, image digest, provider/zone
│   ├── 06_gpu_telemetry.sh    1 Hz DCGM CSV + perf/clock snapshots per level
│   ├── make_bedrock_token.py  LAPTOP: print a short-lived Bedrock token
│   └── bedrock_smoke.py       smoke test + "which param disables thinking" probe
├── monitoring/
│   ├── docker-compose.yml          live stack: Prometheus + Grafana + dcgm-exporter (loopback only)
│   ├── docker-compose.offline.yml  laptop viewer for a run's committed snapshot
│   ├── seed_offline.sh             copy <run>/prometheus-snapshot into the viewer's working copy
│   ├── prometheus.yml, dcgm-metrics.csv, fetch_dashboards.sh
│   ├── offline/                    datasource for the offline viewer
│   └── grafana/                    provisioning + dashboards (Benchmark overview, DCGM)
└── analysis/
    ├── summarize.py            latency / throughput / errors / cost tables + charts
    └── fixtures/               fake results to test summarize.py without a GPU
```

## Architecture

1. **vLLM part** (serve, benchmark, monitoring) runs entirely on the GPU
   instance and needs no AWS access.
2. **Bedrock part** is optional and needs only `OPENAI_API_KEY` (a short-lived
   Bedrock token minted on your laptop) and network access. **AWS credentials
   never touch the GPU instance.**

```
        laptop                    GPU instance (single GPU)
   ┌───────────────┐   SSH tunnel   ┌──────────────────────────────────────┐
   │ make_bedrock_ │  8000/3000/9090│  vLLM (127.0.0.1:8000) ── /metrics     │
   │ token.py      │◀──────────────▶│  dcgm-exporter (127.0.0.1:9400)        │
   │ browser→Grafana│               │  Prometheus(9090) + Grafana(3000)      │
   └───────┬───────┘                │  vllm bench serve ── experiments/<gpu>/<run>/raw │
           │ OPENAI_API_KEY         └─────────────────┬────────────────────┘
           ▼                                          │ (optional, end of session)
   Amazon Bedrock  ◀──────────────── vllm bench serve ┘
   bedrock-mantle.eu-north-1.api.aws / qwen.qwen3-32b
```

## Prerequisites

- A Scaleway GPU instance from the **"Ubuntu Noble GPU OS 13 (Nvidia)"** image
  (ships with NVIDIA drivers and Docker). With a single GPU, vLLM runs without
  tensor parallelism. Tested on **H100-1-80G**; **L40S-1-48G** is next. Only the
  KV-cache budget and the hourly price differ between cards.
- On your laptop, for the Bedrock part only: Python 3.10+,
  `pip install aws-bedrock-token-generator openai`, and AWS credentials that
  can mint a Bedrock token.

---

## Session runbook

### 0. Create the instance and open a tunnel

```bash
# laptop: one tunnel for vLLM, Grafana and Prometheus
ssh -L 8000:localhost:8000 -L 3000:localhost:3000 -L 9090:localhost:9090 root@<instance-ip>
```

On the instance:

```bash
git clone <this-repo> vllm-ops && cd vllm-ops
cp experiments/src/env.example .env     # set GPU=h100|l40s and a NEW RUN=<date>_<label>
```

### 1. Pre-flight

```bash
bash experiments/src/scripts/00_check_instance.sh
```

This checks that the GPU is visible, Docker and the NVIDIA runtime work, and
there is enough disk. It also pulls the pinned image and **stops if the
driver's max CUDA version is lower than the image's**.

### 2. Start vLLM and wait for readiness

```bash
bash experiments/src/scripts/01_start_vllm.sh
bash experiments/src/scripts/02_wait_ready.sh    # -> <run>/raw/startup.json
```

The API is published on **127.0.0.1:8000 only**, because the default Scaleway
firewall may allow inbound traffic. Reach it through the tunnel.

### 3. Start live monitoring

```bash
cd experiments/src/monitoring
./fetch_dashboards.sh                    # optional: official vLLM dashboard (dcgm.json is committed)
docker compose --profile gpu up -d       # Prometheus + Grafana + dcgm-exporter
cd -
```

Open Grafana at <http://localhost:3000> through the tunnel. Anonymous access is
**Admin with no login**. That is acceptable only because Grafana is reachable
solely through the tunnel.

No-stack alternative:

```bash
watch -n1 "curl -s localhost:8000/metrics | grep -E 'vllm:(num_requests_(running|waiting)|kv_cache_usage_perc|num_preemptions)'"
```

### 4. Run the vLLM sweep

```bash
bash experiments/src/scripts/03_bench_sweep.sh          # TARGET=vllm by default
```

For each level in `CONCURRENCIES`, the sweep runs `vllm bench serve` (random
dataset, 1024 in / 256 out, fixed seed, `prompts = max(50, 4×concurrency)`),
takes `/metrics` snapshots before and after, records 1 Hz GPU telemetry, and
writes one JSON per level to `<run>/raw/`.

To re-run only some levels into the same run: `CONCURRENCIES="1024 2048" bash experiments/src/scripts/03_bench_sweep.sh`.

### 5. (Optional) Bedrock sweep

```bash
# laptop (needs local AWS creds):
export OPENAI_API_KEY=$(python experiments/src/scripts/make_bedrock_token.py)
python experiments/src/scripts/bedrock_smoke.py                  # confirm it works
python experiments/src/scripts/bedrock_smoke.py --probe-thinking # find the param that disables thinking
```

```bash
# on the instance (end of session) or on a small CPU VM:
TARGET=bedrock OPENAI_API_KEY=... bash experiments/src/scripts/03_bench_sweep.sh
```

### 6. Snapshot Prometheus into the run

This makes the Grafana dashboards replayable offline after the instance is gone:

```bash
source .env
docker compose -f experiments/src/monitoring/docker-compose.yml stop prometheus   # flush the TSDB
docker run --rm -v monitoring_prometheus-data:/src:ro -v "$PWD/experiments/$GPU/$RUN":/dst \
  alpine sh -c 'rm -rf /dst/prometheus-snapshot && cp -a /src/data /dst/prometheus-snapshot'
```

(The live stack writes its TSDB to `/prometheus/data` inside the volume.)

### 7. Summarize, commit, DELETE the instance

```bash
pip install matplotlib
python experiments/src/analysis/summarize.py --run-dir "experiments/$GPU/$RUN"
```

This writes `summary.md`, `summary.json` and `charts/` next to `raw/`. Then
add a run `README.md` (config, data caveats, Grafana links with the run's time
window), list the run in `experiments/<gpu>/README.md`, commit, and push.

**Delete the instance** (H100 PCIe ~€2.73/h, L40S ~€1.47/h). A forgotten GPU
instance is the most expensive mistake here.

---

## Disabling Qwen3 "thinking"

Qwen3 has a reasoning ("thinking") phase that is on by default. It changes how
many tokens come out, which skews latency. It is disabled on both targets, but
the mechanism differs:

- **vLLM:** `--default-chat-template-kwargs '{"enable_thinking": false}'` on the
  server, plus `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`
  on the client.
- **Bedrock mantle:** it does not accept `chat_template_kwargs`. The default
  `BEDROCK_EXTRA_BODY` is `{"reasoning_effort": "none"}`. **Verify** with
  `bedrock_smoke.py --probe-thinking` and set whatever works. **TODO:** confirm
  on a live endpoint.

vLLM output is forced to exactly 256 tokens (`--ignore-eos`). Bedrock output
length can't be forced, so its `mean out tok` is the real count. Compare TTFT
and per-token latency, not total duration.

## KV-cache dtype

`KV_CACHE_DTYPE` controls how many tokens the KV budget left after the weights
can hold:

- `auto`: **BF16 KV** at 256 KiB per token. This is the baseline.
- `fp8`: **FP8 KV** at 128 KiB per token. That roughly doubles the token
  capacity and max concurrency, at a small accuracy cost.

Measure each dtype as its own run (separate `RUN` names), restarting the server
between them:

```bash
RUN=2026-11-02_fp8-kv KV_CACHE_DTYPE=fp8 bash experiments/src/scripts/01_start_vllm.sh
RUN=2026-11-02_fp8-kv bash experiments/src/scripts/02_wait_ready.sh
RUN=2026-11-02_fp8-kv bash experiments/src/scripts/03_bench_sweep.sh
```

## Error handling

Errors and throttling (HTTP 429) are counted separately and never mixed into the
latency stats. The Bedrock sweep uses `--save-detailed`, so the per-request
`errors` array is saved. `summarize.py` reports the error rate and the 429
count per level.

## Checking vLLM metric names

vLLM metric names change across versions:

```bash
curl -s localhost:8000/metrics | grep '^vllm:' | cut -d'{' -f1 | sort -u
```

The **Benchmark overview** dashboard uses the v0.27.0 names:
`vllm:num_requests_waiting`, `vllm:num_requests_running`,
`vllm:kv_cache_usage_perc`, `vllm:num_preemptions_total`,
`vllm:generation_tokens_total`, `vllm:time_to_first_token_seconds`,
`vllm:inter_token_latency_seconds`.

## Running the Bedrock bench without a GPU

`vllm bench serve` is only an HTTP load generator, so it runs from the vLLM image
**without** `--gpus`. **TODO:** confirm that `import vllm` works on a host with
no GPU. Fallback: `bedrock_smoke.py` for a lighter latency check.

## Cost model

- **Self-hosted** cost per 1M output tokens = `GPU_price_per_hour / (output_throughput × 3600) × 1e6`
  (`--l40s-price 1.47`, `--h100-price 2.73`, or `--gpu-price`; EUR/h).
- **Bedrock** prices are `--bedrock-in-price 0.15` and `--bedrock-out-price 0.60`
  (USD per 1M tokens).
- The break-even throughput treats EUR and USD as equal. Adjust for FX.

## Local testing (no GPU)

```bash
python experiments/src/analysis/summarize.py --results-dir experiments/src/analysis/fixtures --out-dir /tmp/out
bash -n experiments/src/scripts/*.sh experiments/src/monitoring/*.sh
docker compose -f experiments/src/monitoring/docker-compose.yml --profile gpu config
```
