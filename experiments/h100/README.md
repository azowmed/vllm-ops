# H100 PCIe 80 GB

| | |
|---|---|
| Provider / offer | Scaleway `H100-1-80G` (zone `pl-waw-2`) |
| GPU | NVIDIA H100 PCIe, 80 GB HBM2e (81,559 MiB reported), ~2 TB/s |
| Driver / CUDA | 580.178.04 / 13.0 |
| Price used in the cost tables | 2.73 EUR/h |
| Model | `Qwen/Qwen3-32B-FP8`, single GPU, no tensor parallelism |

## Runs

| Run | Server config | Sweep | Notes |
|---|---|---|---|
| [`2026-10-20_bf16-kv`](2026-10-20_bf16-kv/) | vLLM 0.27.0, BF16 KV (`--kv-cache-dtype auto`), `max_model_len` 8192, `gpu_memory_utilization` 0.90 | Concurrency 1 → 2048, 1024 in / 256 out | Concurrency 1024 and 2048 are invalid (client hit its file-descriptor limit) |

To add a run, set `GPU=h100` and a new `RUN=<date>_<label>` in `.env`, then follow
[`../src/README.md`](../src/README.md).
