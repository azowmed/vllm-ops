# LLM serving benchmark: self-hosted vLLM vs Amazon Bedrock (Qwen3-32B)

The question: what does it take to serve **Qwen3-32B-FP8** yourself with
[vLLM](https://docs.vllm.ai/) on a single rented GPU? Specifically:
- how far one card can be pushed (KV cache, concurrency, latency, throughput),
- what the GPU is actually doing while it serves,
- how the cost per token compares with the **same model on Amazon Bedrock**.

## How this repo is split

The repo has two folders, and the split is about **who wrote what**:

| Folder | Written by | Contains |
|---|---|---|
| [`experiments/`](experiments/) | **AI** (Claude Code), run by the owner | The code that runs the benchmarks, the raw outputs, generated summaries and charts, Grafana snapshots, and how to view all of it |
| [`findings/`](findings/) | **The repo owner**. AI only formats it | Predictions written *before* each run, the interpretation of what was measured, and conclusions. It shows how well the owner understands the results |

The measurements in `experiments/` can be checked against each other. The
reasoning in `findings/` is the human part. Commits that touch `findings/` carry
no `Co-Authored-By: Claude` line, so `git log -- findings/` shows the human
history.

## Where to start

- **Read the conclusions:** [`findings/`](findings/)
- **See what was measured:** [`experiments/`](experiments/). There is one folder
  per GPU type (`h100/`, later `l40s/`) and one subfolder per run. Each run has a
  `summary.md` with tables and charts.
- **Reproduce a run on a GPU instance:** [`experiments/src/README.md`](experiments/src/README.md)

## Runs so far

| GPU | Run | Status |
|---|---|---|
| H100 PCIe 80 GB (Scaleway) | [`2026-10-20_bf16-kv`](experiments/h100/2026-10-20_bf16-kv/) | Done. Concurrency 1024 and 2048 are invalid (the client ran out of file descriptors) |
| L40S 48 GB (Scaleway) | — | Planned |
