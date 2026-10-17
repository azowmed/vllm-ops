# LLM serving benchmark: self-hosted vLLM vs Amazon Bedrock (Qwen3-32B)

Serve **Qwen3-32B-FP8** with [vLLM](https://docs.vllm.ai/) on a rented GPU,
watch the server and the GPU while it is under load, and compare latency,
throughput and cost against the **same model on Amazon Bedrock** (the
OpenAI-compatible "mantle" endpoint, `qwen.qwen3-32b`).

The workload generator is vLLM's own `vllm bench serve`, so both targets are
driven by the exact same client, dataset and tokenizer.

> All CLI flags, metric names and the KV-cache math were checked against the
> **vLLM v0.27.0** docs and the Qwen3-32B `config.json`. If you pin a different
> vLLM image, re-check metric names (see [Checking vLLM metric names](#checking-vllm-metric-names))
> and re-run `scripts/00_check_instance.sh`.

## Architecture: two decoupled parts

1. **vLLM part** (serve + bench + monitoring) runs entirely on the GPU instance
   and needs **no AWS anything** — no token, no AWS env var.
2. **Bedrock measurement** is a separate, optional invocation that needs only
   `OPENAI_API_KEY` (a short-lived Bedrock token) and network access. It can run
   from the GPU instance at the end of the session, or from a small CPU-only VM.
   **AWS credentials never touch the GPU instance** — the token is minted on your
   laptop and only the token travels.

```
        laptop                    GPU instance (Scaleway L40S-1-48G, single GPU)
   ┌───────────────┐   SSH tunnel   ┌──────────────────────────────────────┐
   │ make_bedrock_ │  8000/3000/9090│  vLLM (127.0.0.1:8000) ── /metrics     │
   │ token.py      │◀──────────────▶│  dcgm-exporter (127.0.0.1:9400)        │
   │ browser→Grafana│               │  Prometheus(9090) + Grafana(3000)      │
   └───────┬───────┘                │  vllm bench serve  ── results/*.json   │
           │ OPENAI_API_KEY                └─────────────────┬────────────────────┘
           ▼                                                 │ (optional, end of session)
   Amazon Bedrock  ◀──────────────── vllm bench serve ───────┘
   bedrock-mantle.eu-north-1.api.aws / qwen.qwen3-32b
```

## Repository layout

```
.
├── README.md                  this file
├── PREDICTIONS.md             pre-session predictions (KV cache, cold start, floor latency)
├── env.example                copy to .env (git-ignored)
├── .gitignore
├── scripts/
│   ├── _common.sh             shared defaults + .env loader
│   ├── 00_check_instance.sh   GPU / Docker / NVIDIA runtime / disk / CUDA-compat / image pull
│   ├── 01_start_vllm.sh       docker run vLLM, loopback-only publish
│   ├── 02_wait_ready.sh       wait /v1/models; cold-start time; KV-cache + max-concurrency from log
│   ├── 03_bench_sweep.sh      TARGET=vllm|bedrock x concurrency sweep; telemetry + snapshots
│   ├── 04_snapshot_metrics.sh raw /metrics before & after each level
│   ├── 05_collect_env.sh      GPU, driver, CUDA, vLLM version, image digest, provider/zone
│   ├── 06_gpu_telemetry.sh    1 Hz DCGM CSV + perf/clock snapshots per level
│   ├── make_bedrock_token.py  LOCAL: print a short-lived Bedrock token
│   └── bedrock_smoke.py       smoke test + "which param disables thinking" probe
├── monitoring/
│   ├── docker-compose.yml      Prometheus + Grafana + dcgm-exporter (all loopback-only)
│   ├── prometheus.yml
│   ├── dcgm-metrics.csv         custom counters incl. DCGM_FI_PROF_* profiling metrics
│   ├── fetch_dashboards.sh      download vLLM + NVIDIA DCGM dashboards, fix datasource
│   └── grafana/                 provisioning + hand-made "Benchmark overview" dashboard
├── analysis/
│   ├── summarize.py            latency / throughput / error / cost + charts
│   └── fixtures/               fake results so summarize.py can be tested offline
└── results/                    benchmark outputs (git-ignored)
```

## Prerequisites

- A Scaleway GPU instance from the **"Ubuntu Noble GPU OS 13 (Nvidia)"** image
  (ships NVIDIA drivers + Docker). Primary target: a single **L40S-1-48G**
  (48 GB GDDR6, ~864 GB/s, Ada SM 8.9). One GPU, so vLLM runs with no tensor
  parallelism (`--gpus all` sees the single card). The scripts also run
  unchanged on a bigger card (e.g. **H100-1-80G PCIe**, 80 GB, ~2 TB/s) — only
  the KV-cache budget and cost per hour change.
- On your laptop (only for the Bedrock part): Python 3.10+, `pip install
  aws-bedrock-token-generator openai`, and AWS credentials that can mint a
  Bedrock token.

---

## Session runbook

### 0. Create the instance and open a tunnel

Create the instance in the Scaleway console (or CLI) with the GPU OS image. Then,
from your laptop, open **one** tunnel that forwards vLLM, Grafana and Prometheus:

```bash
ssh -L 8000:localhost:8000 -L 3000:localhost:3000 -L 9090:localhost:9090 root@<instance-ip>
```

Everything below runs **on the instance** unless marked **(laptop)**.

```bash
git clone <this-repo> llm-serving-bench && cd llm-serving-bench
cp env.example .env         # review and edit
```

### 1. Pre-flight

```bash
bash scripts/00_check_instance.sh
```

Checks the GPU is visible, Docker + the NVIDIA runtime work, there is enough
disk, pulls the pinned vLLM image, and **stops with a clear message if the
driver's max CUDA version is lower than the image's CUDA version**.

### 2. Start vLLM and wait for readiness

```bash
bash scripts/01_start_vllm.sh
bash scripts/02_wait_ready.sh     # writes results/startup.json (cold start, KV cache tokens)
```

`01` publishes the API on **127.0.0.1:8000 only** (the default Scaleway firewall
may allow inbound — do not bind to `0.0.0.0`). Reach it through the tunnel.

### 3. Start live monitoring

```bash
cd monitoring
VLLM_IMAGE=$(grep ^VLLM_IMAGE ../.env | cut -d= -f2) ./fetch_dashboards.sh
docker compose --profile gpu up -d     # Prometheus + Grafana + dcgm-exporter
cd ..
```

Open Grafana at <http://localhost:3000> (through the tunnel). Anonymous access is
**Admin with no login screen** — acceptable *only* because it is reachable solely
through the SSH tunnel; never expose it publicly. Dashboards: **Benchmark
overview** (hand-made), plus the official vLLM and NVIDIA DCGM dashboards.

**Ultra-light alternative, no stack:**

```bash
watch -n1 "curl -s localhost:8000/metrics | grep -E 'vllm:(num_requests_(running|waiting)|kv_cache_usage_perc|num_preemptions)'"
```

### 4. Run the vLLM sweep

```bash
bash scripts/03_bench_sweep.sh          # TARGET=vllm by default
```

For each concurrency level (default `1 8 16 32 64`) this runs `vllm bench serve`
(random dataset, 1024 in / 256 out, fixed seed, `prompts = max(50, 4×concurrency)`),
brackets it with `/metrics` snapshots and 1 Hz GPU telemetry, and writes one JSON
per level to `results/`. It also records client location and network latency
(`results/vllm_netlatency.json`) and the environment (`results/env.json`).

### 5. (Optional) Bedrock sweep

**(laptop)** mint a token and pass it into your tunneled shell, or export it on
the instance just for this step:

```bash
# (laptop) requires local AWS creds:
export OPENAI_API_KEY=$(python scripts/make_bedrock_token.py)
python scripts/bedrock_smoke.py                 # confirm it works
python scripts/bedrock_smoke.py --probe-thinking # find the param that disables thinking
```

Then run the Bedrock sweep (from the instance at end of session, or a small VM):

```bash
TARGET=bedrock OPENAI_API_KEY=... bash scripts/03_bench_sweep.sh
```

See [Running the Bedrock bench without a GPU](#running-the-bedrock-bench-without-a-gpu).

### 6. Analyze

```bash
pip install matplotlib        # only needed for the charts
python analysis/summarize.py  # reads results/, writes results/summary.md + charts
```

Works with vLLM results alone; the Bedrock comparison appears only if
`results/bedrock_c*.json` exist.

### 7. Copy results back and DELETE the instance

```bash
# (laptop)
scp -r root@<instance-ip>:~/llm-serving-bench/results ./results
```

Then **delete the instance** in the Scaleway console (or CLI). A short session is
typically ~1–2 h: at **L40S €1.47/h** that is ~**€1.5–€3**, at **H100 PCIe €2.73/h**
~**€2.7–€5.5**, plus a little storage. **Do not forget to delete the instance** —
a forgotten GPU instance is the most expensive mistake here.

---

## Disabling Qwen3 "thinking"

Qwen3 runs a reasoning ("thinking") phase by default, which changes how many
tokens come out and skews a latency comparison. We disable it on **both** targets,
but the mechanism differs because the backends expose different knobs:

- **vLLM** (we control the server): launched with
  `--default-chat-template-kwargs '{"enable_thinking": false}'` (hard off for
  every request), and the bench also passes
  `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`.
- **Bedrock mantle**: it does not accept `chat_template_kwargs`; it appears to use
  OpenAI-style `reasoning_effort`. The default `BEDROCK_EXTRA_BODY` is
  `{"reasoning_effort": "none"}`. **Verify on the live endpoint** with
  `python scripts/bedrock_smoke.py --probe-thinking`, which tries each variant and
  reports whether reasoning content came back, then set `BEDROCK_EXTRA_BODY` in
  `.env` to whatever works. **TODO:** this needs a live endpoint to confirm.

If no Bedrock parameter disables thinking, say so plainly in the write-up: vLLM
output has thinking off, Bedrock does not; compare **TTFT and per-token latency**
(not total duration) and rely on the **actual** output-token counts the sweep
records. vLLM output length is forced to exactly 256 tokens (`--ignore-eos`);
**Bedrock cannot be forced**, so `mean out tok` in the summary is the real number.

## KV-cache dtype on the L40S

After the FP8 weights (~34–35 GB) and runtime overhead, an L40S leaves only
~6–7 GB for the KV cache at `GPU_MEM_UTIL=0.90` — the scarcest resource on this
card and the thing that caps concurrency. `KV_CACHE_DTYPE` (in `.env`) controls
how many tokens that budget holds:

- `auto` (default) — **BF16 KV**, 256 KiB per 1,280-token request. The baseline run.
- `fp8` — **FP8 KV**, 128 KiB per request: roughly **double** the token capacity
  (and max concurrency) for a small accuracy cost. Supported on the L40S (Ada).

To measure both, run the sweep twice into separate result dirs:

```bash
RESULTS_DIR=results/kv_bf16 KV_CACHE_DTYPE=auto bash scripts/01_start_vllm.sh && \
  RESULTS_DIR=results/kv_bf16 bash scripts/02_wait_ready.sh && \
  RESULTS_DIR=results/kv_bf16 bash scripts/03_bench_sweep.sh
RESULTS_DIR=results/kv_fp8  KV_CACHE_DTYPE=fp8  bash scripts/01_start_vllm.sh && \
  RESULTS_DIR=results/kv_fp8  bash scripts/02_wait_ready.sh && \
  RESULTS_DIR=results/kv_fp8  bash scripts/03_bench_sweep.sh
```

The chosen dtype is recorded in `results/startup.json` and `results/env.json`, and
the measured KV-cache token count there is what you compare against PREDICTIONS.md.

## Error handling

Errors and throttling (HTTP 429) are counted separately and never mixed into the
latency stats. The Bedrock sweep runs with `--save-detailed`, so the per-request
`errors` array is saved; `summarize.py` reports the error rate and a separate
count of 429s per level.

## Checking vLLM metric names

vLLM metric names change across versions. List the ones your server actually
exposes:

```bash
curl -s localhost:8000/metrics | grep '^vllm:' | cut -d'{' -f1 | sort -u
```

The **Benchmark overview** dashboard uses the v0.27.0 names:
`vllm:num_requests_waiting`, `vllm:num_requests_running`,
`vllm:kv_cache_usage_perc`, `vllm:num_preemptions_total`,
`vllm:generation_tokens_total`, `vllm:time_to_first_token_seconds`,
`vllm:inter_token_latency_seconds`. (Counters are exposed with a `_total`
suffix.) Adjust the panel queries if your version differs.

## Running the Bedrock bench without a GPU

`vllm bench serve` is only an HTTP load generator (tokenizer + requests), so it is
run from the vLLM image **without** `--gpus` — identical on the GPU box or on a
small CPU-only VM. The simplest path is to run the Bedrock sweep **from the GPU
instance at the end of the session**, where the image is already present.

**TODO (verify on the box):** confirm `import vllm` succeeds in the image on a
host with no GPU. If a future build hard-requires CUDA at import, the lightest
fallback is a CPU-only Python env with just the OpenAI client + the Qwen
tokenizer, or use `scripts/bedrock_smoke.py` for a lighter latency check.

## Cost model

- **Self-hosted** cost per 1M output tokens = `GPU_price_per_hour / (output_throughput × 3600) × 1e6`.
  Prices are parameters (`--l40s-price 1.47`, `--h100-price 2.73`, EUR/h).
- **Bedrock** price is a parameter (`--bedrock-in-price 0.15`,
  `--bedrock-out-price 0.60`, USD per 1M tokens by default).
- `summarize.py` also reports the **sustained output throughput needed to
  break even** with Bedrock. Note the self-host (EUR) and Bedrock (USD) prices
  are in different currencies — the break-even line assumes they are equal, so
  adjust for FX before drawing conclusions.

## Local testing (no GPU)

```bash
python analysis/summarize.py --results-dir analysis/fixtures --out-dir /tmp/out
bash -n scripts/*.sh
docker compose -f monitoring/docker-compose.yml --profile gpu config   # needs the compose plugin
# shellcheck scripts/*.sh                                               # if installed
```
