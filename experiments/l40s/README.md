# L40S 48 GB

| | |
|---|---|
| Provider / offer | Scaleway `L40S-1-48G` (zone `pl-waw-2`) |
| GPU | NVIDIA L40S, 48 GB GDDR6 (46,068 MiB reported), ~864 GB/s |
| Driver / CUDA | 580.178.04 / 13.0 |
| Price used in the cost tables | 1.47 EUR/h |
| Model | `Qwen/Qwen3-32B-FP8`, single GPU, no tensor parallelism |

## Runs

Both runs: vLLM 0.27.0, `max_model_len` 8192, `gpu_memory_utilization` 0.90,
1024 in / 256 out, concurrency 1 8 16 32 64 128. They ran on the same instance,
one after the other.

| Run | KV cache | KV cache size | Time (UTC) | Notes |
|---|---|---|---|---|
| [`2026-11-15_bf16-kv`](2026-11-15_bf16-kv/) | BF16 (`--kv-cache-dtype auto`) | 24,912 tokens | 20:28 → 21:01 | All requests completed, no errors |
| [`2026-11-15_fp8-kv`](2026-11-15_fp8-kv/) | FP8 (`--kv-cache-dtype fp8`) | 49,840 tokens | 21:20 → 21:47 | All requests completed, no errors. Started with the model already cached |

Data caveats, both runs:
- **DCGM telemetry has 30 s resolution.** dcgm-exporter refreshed its profiling
  values every 30 s (its default), so each value repeats ~30 times in
  `raw/telemetry_c*.csv`.
- **Prometheus snapshots:** the BF16 snapshot was taken after its sweep. The FP8
  snapshot covers both runs' time windows, so it's the one to load in the
  offline viewer (see [`../README.md`](../README.md#three-ways-to-view-a-run)).

To add a run, set `GPU=l40s` and a new `RUN=<date>_<label>`, then follow
[`../src/README.md`](../src/README.md).
