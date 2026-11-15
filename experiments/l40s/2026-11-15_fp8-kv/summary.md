# Benchmark summary

- GPU: **NVIDIA L40S** (46068 MiB), driver 580.178.04 (CUDA 13.0)
- vLLM: `0.27.0` image `vllm/vllm-openai:v0.27.0`
- client: scw-objective-ganguly @ scaleway/pl-waw-2
- KV cache: **49840 tokens**, cold start **152 s**, announced max concurrency 6.08x

## vllm
_network latency to target: connect >= 0.000 s, TTFB >= 0.001 s (client scw-objective-ganguly @ pl-waw-2)_

| conc | completed/total | err% | 429 | TTFT p50/p95 (ms) | per-token p50/p95 (ms) | out tok/s | mean out tok |
|---|---|---|---|---|---|---|---|
| 1 | 50/50 | 0.0 | 0 | 299.6/301.3 | 50.70/50.72 | 19.3 | 256 |
| 8 | 50/50 | 0.0 | 0 | 1140.7/1947.1 | 56.68/58.62 | 119.9 | 256 |
| 16 | 64/64 | 0.0 | 0 | 1652.4/4064.8 | 62.86/67.97 | 228.0 | 256 |
| 32 | 128/128 | 0.0 | 0 | 1971.1/7393.5 | 83.11/87.96 | 348.4 | 256 |
| 64 | 256/256 | 0.0 | 0 | 9061.7/31614.7 | 102.98/148.17 | 353.6 | 256 |
| 128 | 512/512 | 0.0 | 0 | 59923.8/64032.1 | 102.40/149.08 | 365.6 | 256 |

## Cost

Self-hosted price assumed: **1.47 EUR/h** (L40S). Cost/Mtok(out) = price_per_hour / (output_throughput x 3600) x 1e6.

| conc | out tok/s | self-host EUR / Mtok(out) |
|---|---|---|
| 1 | 19.3 | 21.144 |
| 8 | 119.9 | 3.404 |
| 16 | 228.0 | 1.791 |
| 32 | 348.4 | 1.172 |
| 64 | 353.6 | 1.155 |
| 128 | 365.6 | 1.117 |

Best measured output throughput: **365.6 tok/s** -> **1.117 EUR / Mtok(out)**.

Bedrock output price assumed: **0.6 USD / Mtok** (input 0.15 USD / Mtok).
Break-even sustained output throughput (self-host EUR/Mtok == Bedrock price, **treating the two currencies as equal — adjust for FX**): **680.6 tok/s**.
At the measured best throughput, self-hosting is **more expensive than** Bedrock per output token (same-currency assumption).

## Prediction vs measured

Measured facts to drop into findings/predictions.md:

- `kv_cache_tokens` = **49840**
- `cold_start_s` = **152**
- `floor_tpot_ms_c1` = **50.70**

## Charts

![ttft_p95.png](charts/ttft_p95.png)

![tpot_p95.png](charts/tpot_p95.png)

![throughput.png](charts/throughput.png)

![error_rate.png](charts/error_rate.png)

![cost.png](charts/cost.png)

![telemetry_c1.png](charts/telemetry_c1.png)

![telemetry_c8.png](charts/telemetry_c8.png)

![telemetry_c16.png](charts/telemetry_c16.png)

![telemetry_c32.png](charts/telemetry_c32.png)

![telemetry_c64.png](charts/telemetry_c64.png)

![telemetry_c128.png](charts/telemetry_c128.png)


