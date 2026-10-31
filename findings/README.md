# findings/ (human-written)

This folder is **written by the repo owner**. AI may fix formatting (tables,
headings, typos) but does not write the content: no reasoning, no
interpretation, no conclusions. It is meant to show what the owner understood
after running the experiments, so the gaps and wrong guesses stay in on purpose.

The measured evidence lives in [`../experiments/`](../experiments/). Link to it
instead of copying numbers.

| File | What it is |
|---|---|
| [`predictions.md`](predictions.md) | Predictions written **before** each run (KV cache size, knee, latency floor, cost), then compared with what was measured and each gap explained |
| `h100.md`, `l40s.md` | Interpretation of each GPU's runs (to write) |
| `conclusions.md` | Across GPUs and against Bedrock: what to buy or rent, and why (to write) |

Convention: commits touching this folder carry no `Co-Authored-By: Claude`
trailer, so `git log -- findings/` shows the human history.
