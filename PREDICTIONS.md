# Predictions

Written **before** each run. Fill the "Measured" column from the vLLM startup log and the benchmark results, then explain every gap.

## Run 1 — Qwen3-32B-FP8 on 1× L40S (Scaleway L40S-1-48G)

### Setup

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA L40S, 48 GB GDDR6 (~45 GiB as reported by `nvidia-smi`) | Scaleway, NVIDIA datasheet |
| Model | `Qwen/Qwen3-32B-FP8` | Hugging Face |
| Layers | 64 | `config.json` |
| KV heads (GQA) | 8 (64 query heads) | `config.json` |
| Head dimension | 128 | `config.json` |
| `gpu_memory_utilization` | 0.90 | vLLM default |
| `max_model_len` | 8192 | our setting |

### KV cache capacity

**Memory budget for the KV cache**

```
usable GPU memory   = 48 GB × 0.90                  = 43.2 GB
weights (FP8)       ≈ 34.5 GB   (verify: total_size in model.safetensors.index.json;
                                  embeddings and lm_head stay in BF16)
runtime overhead    ≈ 2 GB      (activation peak, CUDA graphs, CUDA context)
KV budget           ≈ 43.2 − 34.5 − 2               ≈ 6.7 GB
```

**KV size per token**

```
values per token = 2 (K and V) × 64 layers × 8 KV heads × 128 dims = 131,072
```

| KV cache dtype | Bytes per value | Bytes per token |
|---|---|---|
| BF16 (vLLM default, `--kv-cache-dtype auto`, even with FP8 weights) | 2 | 262,144 (256 KiB) |
| FP8 (`--kv-cache-dtype fp8`) | 1 | 131,072 (128 KiB) |

**Capacity**

| | Predicted | Measured (vLLM log) |
|---|---|---|
| KV cache size, BF16 KV | 6.7 GB ÷ 256 KiB ≈ **25,500 tokens** | |
| Full 1,280-token requests that fit, BF16 KV | 25,500 ÷ 1,280 ≈ **20** | |
| KV cache size, FP8 KV | 6.7 GB ÷ 128 KiB ≈ **51,000 tokens** | |
| Full 1,280-token requests that fit, FP8 KV | 51,000 ÷ 1,280 ≈ **40** | |

Note: vLLM reports memory in GiB (1 GiB = 1.074 GB). Convert before comparing.

**Which scenario the default run measures.** The scripts default to `KV_CACHE_DTYPE=auto`
(**BF16 KV**), so the baseline numbers below are the BF16 column (~25,500 tokens,
~20 full requests). Re-run with `KV_CACHE_DTYPE=fp8` to hit the FP8 column
(~51,000 tokens, ~40 full requests). The concurrency sweep `1 8 16 32 64` is chosen
to **bracket the BF16 KV knee**: 1/8/16 fit under the ~20-request cap, 32/64 exceed
it and should show queueing and preemption. On an L40S this is the binding limit —
the card has only ~6-7 GB left for KV after the FP8 weights, so KV capacity, not
compute, is what caps concurrency.

### Other predictions (to fill before the run)

| Prediction | Reasoning | Predicted | Measured |
|---|---|---|---|
| Decode latency floor, batch 1 | | | |
| Output throughput at each concurrency level (1, 8, 16, 32, 64) | | | |
| Concurrency where throughput plateaus | | | |
| TTFT behaviour as concurrency grows | | | |
| Preemptions (which levels, how many) | | | |
| GPU telemetry: DRAM vs tensor-core activity, decode vs prefill | | | |
| Cold start time (launch → ready) | | | |
| Cost per million output tokens vs Bedrock | | | |
