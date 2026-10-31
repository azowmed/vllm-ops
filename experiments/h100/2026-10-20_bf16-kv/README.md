# H100 run 2026-10-20: BF16 KV cache

The default server config, swept from 1 to 2048 concurrent requests.

## Config

| | |
|---|---|
| Image | `vllm/vllm-openai:v0.27.0` |
| Model | `Qwen/Qwen3-32B-FP8`, thinking disabled |
| `--kv-cache-dtype` | `auto` (BF16) |
| `--max-model-len` | 8192 |
| `--gpu-memory-utilization` | 0.90 |
| Workload | `vllm bench serve`, random dataset, 1024 in / 256 out (`--ignore-eos`), seed 42 |
| Concurrency levels | 1 8 16 32 64 128 256 512 1024 2048, with `prompts = max(50, 4 × concurrency)` |
| Client | Same host as the server, over loopback (`raw/vllm_netlatency.json`) |
| Time window | 2026-10-20, 22:47 → 23:25 UTC |

## Data caveats

- **Concurrency 1024 and 2048 are invalid.** The benchmark client container had
  the default open-files limit (~1024). Both levels stopped at exactly 996
  completed requests and failed the rest with `[Errno 24] Too many open files`.
  This was a client limit, not a server limit. `--ulimit nofile` has since been
  added to `01` and `03`. Re-run these two levels before using them.
- **DCGM metrics in Prometheus start around 23:13 UTC.** dcgm-exporter was
  crash-looping on a malformed `dcgm-metrics.csv` until then. The Grafana DCGM
  panels only cover the later levels. The per-level 1 Hz `raw/telemetry_c*.csv`
  covers the whole sweep.
- **Cold start (2418 s) probably includes the model download** into an empty
  Hugging Face cache, so it is not a warm restart time.
- **No Bedrock run** was made in this session.

## How to view

- **Tables + charts:** [`summary.md`](summary.md)
- **Grafana, offline** (see [`../../README.md`](../../README.md#three-ways-to-view-a-run) for setup):
  - Benchmark overview: <http://localhost:3000/d/bench-overview?from=1792536300000&to=1792539600000&tz=UTC>
  - NVIDIA DCGM: <http://localhost:3000/d/Oxed_c6Wz/nvidia-dcgm-exporter-dashboard?from=1792537800000&to=1792539600000&tz=UTC&var-instance=127.0.0.1%3A9400&var-gpu=0>.
    Some panels stay empty: they use DCGM fields that were not exported.
- **Raw files:** [`raw/`](raw/). The file list is in [`../../README.md`](../../README.md#what-raw-contains).
