# Predictions

Written **before** each run. Fill the "Measured" column from the vLLM startup log and the benchmark results, then explain every gap.

## Run 1 — Qwen3-32B-FP8 on 1× H100 PCIe (Scaleway H100-1-80G)

The L40S was out of stock, so the first run uses the H100. The L40S predictions below are kept for a later run.

### Inputs

| Item | Value |
|---|---|
| GPU | H100 PCIe, 80 GB HBM2e, ~2.0 TB/s, ~1,513 TFLOPS FP8 dense |
| Weights in VRAM | ~34.5 GB (FP8; embeddings and lm_head in BF16) |
| Weights read per decode step | ~32.8 GB (one embedding row only) → 32.8 GB ÷ 2.0 TB/s ≈ 16.4 ms |
| FLOPs per decode token | ~66 GFLOP (2 × non-embedding params + attention at ~1,000 tokens) → ~44 µs |
| KV per token | 256 KiB in BF16 (vLLM default), 128 KiB in FP8 |
| Benchmark load | 1,024 input + 256 output tokens → ~1,150 tokens of context on average during decode |

### KV cache capacity

| | Predicted | Measured |
|---|---|---|
| KV budget | 80 × 0.90 − 34.5 − ~2.5 ≈ 35 GB | |
| KV cache size, BF16 KV | ≈ 133,000 tokens | |
| Full 1,280-token requests that fit, BF16 KV | ≈ 104 | |
| KV cache size, FP8 KV | ≈ 267,000 tokens | |
| Full 1,280-token requests that fit, FP8 KV | ≈ 208 | |

### Decode, BF16 KV (theoretical ceilings; expect measurements 15–30% lower)

KV read per request per step ≈ 1,150 × 256 KiB ≈ 0.30 GB → 0.15 ms.

| Concurrency | Compute | Memory (weights + KV) | Step | Output throughput | Per-user speed | Measured throughput |
|---|---|---|---|---|---|---|
| 1 | 0.04 ms | 16.4 + 0.15 = 16.6 ms | 16.6 ms | ~60 tok/s | ~60 tok/s | |
| 8 | 0.35 ms | 16.4 + 1.2 = 17.6 ms | 17.6 ms | ~450 tok/s | ~57 tok/s | |
| 16 | 0.7 ms | 16.4 + 2.4 = 18.8 ms | 18.8 ms | ~850 tok/s | ~53 tok/s | |
| 32 | 1.4 ms | 16.4 + 4.8 = 21.2 ms | 21.2 ms | ~1,500 tok/s | ~47 tok/s | |
| 64 | 2.8 ms | 16.4 + 9.7 = 26.1 ms | 26.1 ms | ~2,450 tok/s | ~38 tok/s | |
| ~104 (cache full) | 4.6 ms | 16.4 + 15.7 = 32.1 ms | 32.1 ms | ~3,250 tok/s (plateau) | ~31 tok/s | |

Memory-bound at every level: each extra request adds ~0.04 ms of compute but ~0.15 ms of KV reads. Levels 1–64 all fit in the cache, so add 96 and 128 to reach the plateau.

### Other predictions

| Prediction | Predicted | Measured |
|---|---|---|
| Decode latency floor, batch 1 | ~16.6 ms theoretical, ~18–20 ms realistic | |
| Prefill of 1,024 tokens | compute-bound: ~67.6 TFLOP ÷ 1,513 TFLOPS ≈ 45 ms (weights read: 16 ms) | |
| TTFT, levels 1–32 | ~50–150 ms (prefill + sharing steps with decodes) | |
| TTFT, level 64 | median still low; p99 up to ~3 s from the initial burst (64 prefills at once ≈ 64 × 45 ms) | |
| TTFT, level 128 (above capacity) | seconds: new requests wait for running ones to finish (~256 steps × ~32 ms ≈ 8 s each) | |
| Preemptions | none at ≤ 64 (BF16 KV); likely at 128 | |
| GPU telemetry | DRAM activity high at every level during decode; tensor-core activity ~0.3% at 1 → ~11% at 64, high during prefill bursts | |
| Cold start, first launch | ~5–12 min: image pull, ~34 GB download, weight loading, torch.compile, CUDA graph capture | |
| Cold start, warm restart | ~1–2 min (image, model and compile cache already present) | |
| Cost per 1M output tokens | €2.73/h ÷ throughput: ~€0.31 at 64, ~€0.23 at the plateau (theoretical) | |
| Break-even vs Bedrock (~€0.55 per 1M output tokens) | ~1,400 tok/s sustained, i.e. around concurrency 32 | |
| FP8 KV variant (`--kv-cache-dtype fp8`) | at 64: step ≈ 21 ms, ~3,000 tok/s; capacity ~208 requests | |

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

### Other predictions (to fill before the run)

| Prediction | Reasoning | Predicted | Measured |
|---|---|---|---|
| Decode latency floor, batch 1 | Memory-bound: reading the FP8 weights from VRAM dominates each step | 38 ms | |
| Output throughput at each concurrency level (1, 8, 16, 32, 64) | Grows with concurrency because the weights are read once per step for all requests, until the KV cache is full | 26 tok/s at 1 → ~1,100 tok/s | |
| Concurrency where throughput plateaus | When the KV cache is full | ~1,100 tok/s plateau | |
| TTFT behaviour as concurrency grows | TTFT increases with concurrency because each step takes longer, until it reaches the plateau | Rising, then flat | |
| Preemptions (which levels, how many) | To discover: how vLLM handles a 52nd request while the 51 that fit in the cache are running. I believe one setting controls at which level and how many | Unknown | |
| GPU telemetry: DRAM vs tensor-core activity, decode vs prefill | DRAM should grow as queries arrive. Tokenizer and embedding lookup run on CPU memory, not the GPU; the CPU sits mostly idle. Tensor-core activity: to explore. Decode memory-bound, prefill compute-bound | — | |
| Cold start time (launch → ready) | Loading the weights into GPU memory takes longer than the ready state, where only the VRAM → GPU transfer remains (~74 ms for BF16) | > 74 ms | |
| Cost per million output tokens vs Bedrock | H100 (available on Scaleway today): 1,513 TFLOPS FP8, 3.35 TB/s. One token with 1,000 tokens of KV: 66 GFLOP ÷ 1,513 TFLOPS = 43 µs. 1M tokens without batching: ~11 h. With 240 concurrent requests (KV cache 31.2 GB): ~2.7 h, i.e. ~$8 for ~3 h. Bedrock probably cheaper | ~$8 per 1M output tokens | |
