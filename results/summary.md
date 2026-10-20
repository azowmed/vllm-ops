# Benchmark summary

- GPU: **NVIDIA H100 PCIe** (81559 MiB), driver 580.178.04 (CUDA 13.0)
- vLLM: `0.27.0` image `vllm/vllm-openai:v0.27.0`
- client: scw-charming-euler @ scaleway/pl-waw-2
- KV cache: **150240 tokens**, cold start **2418 s**, announced max concurrency 18.34x

## vllm
_network latency to target: connect >= 0.000 s, TTFB >= 0.001 s (client scw-charming-euler @ pl-waw-2)_

| conc | completed/total | err% | 429 | TTFT p50/p95 (ms) | per-token p50/p95 (ms) | out tok/s | mean out tok |
|---|---|---|---|---|---|---|---|
| 1 | 50/50 | 0.0 | 0 | 137.9/139.0 | 20.46/20.47 | 47.8 | 256 |
| 8 | 50/50 | 0.0 | 0 | 67.9/78.7 | 22.40/22.42 | 319.7 | 256 |
| 16 | 64/64 | 0.0 | 0 | 78.3/1635.7 | 24.17/28.82 | 615.6 | 256 |
| 32 | 128/128 | 0.0 | 0 | 214.1/2574.6 | 30.31/38.40 | 887.9 | 256 |
| 64 | 256/256 | 0.0 | 0 | 2840.6/5777.7 | 47.87/59.95 | 1054.1 | 256 |
| 128 | 512/512 | 0.0 | 0 | 3840.3/13302.3 | 98.63/114.97 | 1085.3 | 256 |
| 256 | 1024/1024 | 0.0 | 0 | 33595.5/41203.9 | 107.29/151.38 | 1055.8 | 256 |
| 512 | 2048/2048 | 0.0 | 0 | 94437.3/96428.8 | 107.30/153.16 | 1063.9 | 256 |
| 1024 | 996/4096 | 75.7 | 0 | 107962.9/221392.3 | 106.94/148.12 | 1072.4 | 256 |
| 2048 | 996/8192 | 87.8 | 0 | 108600.5/223476.5 | 108.10/147.75 | 1062.7 | 256 |

## Cost

Self-hosted price assumed: **2.73 EUR/h** (H100 PCIe). Cost/Mtok(out) = price_per_hour / (output_throughput x 3600) x 1e6.

| conc | out tok/s | self-host EUR / Mtok(out) |
|---|---|---|
| 1 | 47.8 | 15.860 |
| 8 | 319.7 | 2.372 |
| 16 | 615.6 | 1.232 |
| 32 | 887.9 | 0.854 |
| 64 | 1054.1 | 0.719 |
| 128 | 1085.3 | 0.699 |
| 256 | 1055.8 | 0.718 |
| 512 | 1063.9 | 0.713 |
| 1024 | 1072.4 | 0.707 |
| 2048 | 1062.7 | 0.714 |

Best measured output throughput: **1085.3 tok/s** -> **0.699 EUR / Mtok(out)**.

Bedrock output price assumed: **0.6 USD / Mtok** (input 0.15 USD / Mtok).
Break-even sustained output throughput (self-host EUR/Mtok == Bedrock price, **treating the two currencies as equal — adjust for FX**): **1263.9 tok/s**.
At the measured best throughput, self-hosting is **more expensive than** Bedrock per output token (same-currency assumption).

## Prediction vs measured

Measured facts to drop into PREDICTIONS.md:

- `kv_cache_tokens` = **150240**
- `cold_start_s` = **2418**
- `floor_tpot_ms_c1` = **20.46**

## Charts

- `results/ttft_p95.png`
- `results/tpot_p95.png`
- `results/throughput.png`
- `results/error_rate.png`
- `results/cost.png`
- `results/telemetry_c1.png`
- `results/telemetry_c1024.png`
- `results/telemetry_c128.png`
- `results/telemetry_c16.png`
- `results/telemetry_c2048.png`
- `results/telemetry_c256.png`
- `results/telemetry_c32.png`
- `results/telemetry_c512.png`
- `results/telemetry_c64.png`
- `results/telemetry_c8.png`

