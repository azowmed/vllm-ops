#!/usr/bin/env python3
"""Summarise benchmark results: latency, throughput, error rate, cost.

Works with vLLM results alone; Bedrock comparison is added only if bedrock_*
result files are present (with its measured network latency shown).

Inputs (in --results-dir, all optional except at least one vllm_c*.json):
  vllm_c<N>.json, bedrock_c<N>.json     per-level `vllm bench serve` output
  startup.json                          KV cache tokens + cold start (02)
  env.json                              GPU / driver / vLLM version (05)
  <target>_netlatency.json              client location + network latency (03)
  telemetry_c<N>.csv                    1 Hz GPU telemetry (06)

Outputs (in --out-dir, default <results-dir>):
  summary.md, summary.json, and PNG charts (if matplotlib is installed).

Usage:
  python analysis/summarize.py                      # uses ./results
  python analysis/summarize.py --results-dir analysis/fixtures
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # noqa: BLE001
    HAVE_MPL = False

# Default per-hour GPU prices (EUR) and Bedrock prices (USD per 1M tokens).
DEFAULTS = {
    "l40s_price": 1.47,   # EUR/h
    "h100_price": 2.73,   # EUR/h
    "bedrock_in": 0.15,   # USD / 1M input tokens
    "bedrock_out": 0.60,  # USD / 1M output tokens
}


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def pick(d, *keys):
    """First present, non-null key."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def collect(results_dir):
    """Return {target: {level: result_dict}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*_c*.json"))):
        name = os.path.basename(path)
        m = re.match(r"(?P<target>[a-zA-Z0-9]+)_c(?P<c>\d+)\.json$", name)
        if not m:
            continue
        data = load_json(path)
        if data is None:
            continue
        out.setdefault(m["target"], {})[int(m["c"])] = data
    return out


def error_stats(res):
    """Return (error_rate, throttle_429_count)."""
    num = pick(res, "num_prompts") or 0
    completed = pick(res, "completed") or 0
    failed = max(num - completed, 0)
    rate = (failed / num) if num else 0.0
    throttles = 0
    errors = res.get("errors")
    if isinstance(errors, list):
        throttles = sum(1 for e in errors if e and "429" in str(e))
    return rate, throttles


def row_for(res):
    num = pick(res, "num_prompts") or 0
    completed = pick(res, "completed") or 0
    out_tok = pick(res, "total_output_tokens") or 0
    rate, throttles = error_stats(res)
    return {
        "num_prompts": num,
        "completed": completed,
        "duration_s": pick(res, "duration"),
        "output_throughput_tok_s": pick(res, "output_throughput"),
        "total_token_throughput_tok_s": pick(res, "total_token_throughput"),
        "mean_output_tokens": (out_tok / completed) if completed else None,
        "ttft_p50_ms": pick(res, "p50_ttft_ms", "median_ttft_ms"),
        "ttft_p95_ms": pick(res, "p95_ttft_ms"),
        "ttft_mean_ms": pick(res, "mean_ttft_ms"),
        "tpot_p50_ms": pick(res, "p50_tpot_ms", "median_tpot_ms"),
        "tpot_p95_ms": pick(res, "p95_tpot_ms"),
        "tpot_mean_ms": pick(res, "mean_tpot_ms"),
        "error_rate": rate,
        "throttle_429": throttles,
    }


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def self_host_cost_per_mtok(price_per_hour, throughput_tok_s):
    if not throughput_tok_s:
        return None
    return price_per_hour / (throughput_tok_s * 3600.0) * 1e6


def gpu_price(env, args):
    if args.gpu_price is not None:
        return args.gpu_price, "override"
    name = ((env or {}).get("gpu_name") or "").upper()
    if "L40S" in name:
        return args.l40s_price, "L40S"
    if "H100" in name:
        return args.h100_price, "H100 PCIe"
    return None, name or "unknown"


# --------------------------------------------------------------------------- #
# Predictions table (parsed from PREDICTIONS.md)
# --------------------------------------------------------------------------- #
def measured_values(data, startup):
    """Measured facts to compare against predictions."""
    vals = {}
    if startup:
        vals["kv_cache_tokens"] = startup.get("kv_cache_tokens")
        vals["cold_start_s"] = startup.get("cold_start_seconds")
    # Floor per-token latency: median TPOT at the lowest concurrency (batch 1).
    vllm = data.get("vllm", {})
    if vllm:
        c1 = min(vllm)
        vals["floor_tpot_ms_c%d" % c1] = pick(vllm[c1], "p50_tpot_ms", "median_tpot_ms")
    return vals


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def line_plot(out_dir, fname, title, ylabel, series):
    """series: list of (label, xs, ys)."""
    if not HAVE_MPL:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, xs, ys in series:
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
    ax.set_xlabel("concurrency")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def telemetry_plots(results_dir, out_dir):
    made = []
    if not HAVE_MPL:
        return made
    for path in sorted(glob.glob(os.path.join(results_dir, "telemetry_c*.csv"))):
        level = re.search(r"telemetry_(c\d+)\.csv$", os.path.basename(path))
        level = level.group(1) if level else os.path.basename(path)
        t0 = None
        xs, dram, tensor = [], [], []
        with open(path) as fh:
            for r in csv.DictReader(fh):
                try:
                    ep = float(r["epoch"])
                except (KeyError, ValueError):
                    continue
                t0 = ep if t0 is None else t0
                xs.append(ep - t0)
                dram.append(_f(r.get("dram_active")))
                tensor.append(_f(r.get("tensor_active")))
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(xs, dram, label="DRAM active")
        ax.plot(xs, tensor, label="Tensor-core active")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("activity ratio (0-1)")
        ax.set_ylim(0, 1)
        ax.set_title(f"GPU activity over time - {level}\n(prefill: tensor high; decode: DRAM high)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, f"telemetry_{level}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)
    return made


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=os.path.join(repo_root, "results"))
    ap.add_argument("--out-dir", default=None, help="where to write summary + charts (default: results-dir)")
    ap.add_argument("--predictions", default=os.path.join(repo_root, "PREDICTIONS.md"))
    ap.add_argument("--gpu-price", type=float, default=None, help="EUR/h override")
    ap.add_argument("--l40s-price", type=float, default=DEFAULTS["l40s_price"])
    ap.add_argument("--h100-price", type=float, default=DEFAULTS["h100_price"])
    ap.add_argument("--bedrock-in-price", type=float, default=DEFAULTS["bedrock_in"])
    ap.add_argument("--bedrock-out-price", type=float, default=DEFAULTS["bedrock_out"])
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or results_dir
    os.makedirs(out_dir, exist_ok=True)

    data = collect(results_dir)
    if "vllm" not in data or not data["vllm"]:
        sys.stderr.write(f"No vllm_c*.json found in {results_dir}\n")
        return 1

    env = load_json(os.path.join(results_dir, "env.json"))
    startup = load_json(os.path.join(results_dir, "startup.json"))
    net = {t: load_json(os.path.join(results_dir, f"{t}_netlatency.json")) for t in data}

    price, price_label = gpu_price(env, args)

    md = []
    summary = {"env": env, "startup": startup, "targets": {}}
    md.append("# Benchmark summary\n")
    if env:
        md.append(f"- GPU: **{env.get('gpu_name')}** ({env.get('gpu_memory_total')}), "
                  f"driver {env.get('driver_version')} (CUDA {env.get('driver_cuda_version')})")
        md.append(f"- vLLM: `{env.get('vllm_version')}` image `{env.get('vllm_image')}`")
        md.append(f"- client: {env.get('hostname')} @ {env.get('cloud_provider')}/{env.get('cloud_zone')}")
    if startup:
        md.append(f"- KV cache: **{startup.get('kv_cache_tokens')} tokens**, "
                  f"cold start **{startup.get('cold_start_seconds')} s**, "
                  f"announced max concurrency {startup.get('max_concurrency')}x")
    md.append("")

    # Per-target tables.
    hdr = ("| conc | completed/total | err% | 429 | TTFT p50/p95 (ms) | per-token p50/p95 (ms) | "
           "out tok/s | mean out tok |")
    sep = "|---|---|---|---|---|---|---|---|"
    for target in sorted(data):
        md.append(f"## {target}")
        nl = net.get(target)
        if nl:
            md.append(f"_network latency to target: connect >= {fmt(nl.get('min_time_connect_s'),3)} s, "
                      f"TTFB >= {fmt(nl.get('min_time_ttfb_s'),3)} s "
                      f"(client {nl.get('client_hostname')} @ {nl.get('cloud_zone')})_\n")
        if target == "bedrock":
            md.append("> Bedrock output length is not forced; `mean out tok` is the ACTUAL tokens "
                      "generated. Only TTFT and per-token latency are comparable to vLLM.\n")
        md.append(hdr)
        md.append(sep)
        summary["targets"][target] = {}
        for c in sorted(data[target]):
            r = row_for(data[target][c])
            summary["targets"][target][c] = r
            md.append("| {c} | {cp}/{np} | {er} | {t4} | {t50}/{t95} | {p50}/{p95} | {ot} | {mo} |".format(
                c=c, cp=r["completed"], np=r["num_prompts"],
                er=fmt(r["error_rate"] * 100, 1), t4=r["throttle_429"],
                t50=fmt(r["ttft_p50_ms"], 1), t95=fmt(r["ttft_p95_ms"], 1),
                p50=fmt(r["tpot_p50_ms"], 2), p95=fmt(r["tpot_p95_ms"], 2),
                ot=fmt(r["output_throughput_tok_s"], 1), mo=fmt(r["mean_output_tokens"], 0),
            ))
        md.append("")

    # Cost.
    md.append("## Cost\n")
    if price is None:
        md.append("_Self-hosted cost skipped: unknown GPU. Pass --gpu-price <EUR/h>._\n")
    else:
        md.append(f"Self-hosted price assumed: **{price} EUR/h** ({price_label}). "
                  "Cost/Mtok(out) = price_per_hour / (output_throughput x 3600) x 1e6.\n")
        md.append("| conc | out tok/s | self-host EUR / Mtok(out) |")
        md.append("|---|---|---|")
        best_tp = 0.0
        for c in sorted(data["vllm"]):
            tp = pick(data["vllm"][c], "output_throughput") or 0.0
            best_tp = max(best_tp, tp)
            md.append(f"| {c} | {fmt(tp,1)} | {fmt(self_host_cost_per_mtok(price, tp),3)} |")
        md.append("")
        md.append(f"Best measured output throughput: **{fmt(best_tp,1)} tok/s** "
                  f"-> **{fmt(self_host_cost_per_mtok(price, best_tp),3)} EUR / Mtok(out)**.")
        # Break-even throughput vs Bedrock output price.
        be = price * 1e6 / (args.bedrock_out_price * 3600.0) if args.bedrock_out_price else None
        md.append(f"\nBedrock output price assumed: **{args.bedrock_out_price} USD / Mtok** "
                  f"(input {args.bedrock_in_price} USD / Mtok).")
        md.append(f"Break-even sustained output throughput (self-host EUR/Mtok == Bedrock price, "
                  f"**treating the two currencies as equal — adjust for FX**): "
                  f"**{fmt(be,1)} tok/s**.")
        if be:
            verdict = "cheaper than" if best_tp > be else "more expensive than"
            md.append(f"At the measured best throughput, self-hosting is **{verdict}** Bedrock per output token "
                      "(same-currency assumption).")
        summary["cost"] = {"gpu_price_eur_h": price, "gpu": price_label,
                           "best_output_tok_s": best_tp,
                           "break_even_tok_s": be}
    md.append("")

    # Prediction vs measured.
    meas = measured_values(data, startup)
    if os.path.exists(args.predictions):
        md.append("## Prediction vs measured\n")
        md.append("Measured facts to drop into PREDICTIONS.md:\n")
        for k, v in meas.items():
            md.append(f"- `{k}` = **{fmt(v)}**")
        md.append("")
    summary["measured"] = meas

    # Charts.
    charts = []
    if not args.no_plots and HAVE_MPL:
        levels = {t: sorted(data[t]) for t in data}
        charts.append(line_plot(out_dir, "ttft_p95.png", "TTFT p95 vs concurrency", "TTFT p95 (ms)",
                                 [(t, levels[t], [row_for(data[t][c])["ttft_p95_ms"] for c in levels[t]]) for t in data]))
        charts.append(line_plot(out_dir, "tpot_p95.png", "Per-token latency p95 vs concurrency", "TPOT p95 (ms)",
                                 [(t, levels[t], [row_for(data[t][c])["tpot_p95_ms"] for c in levels[t]]) for t in data]))
        charts.append(line_plot(out_dir, "throughput.png", "Output throughput vs concurrency", "tokens/s",
                                 [(t, levels[t], [row_for(data[t][c])["output_throughput_tok_s"] for c in levels[t]]) for t in data]))
        charts.append(line_plot(out_dir, "error_rate.png", "Error rate vs concurrency", "error rate (%)",
                                 [(t, levels[t], [row_for(data[t][c])["error_rate"] * 100 for c in levels[t]]) for t in data]))
        if price is not None:
            charts.append(line_plot(out_dir, "cost.png", "Self-host cost vs concurrency", "EUR / Mtok(out)",
                                     [("vllm", levels["vllm"],
                                       [self_host_cost_per_mtok(price, pick(data["vllm"][c], "output_throughput")) for c in levels["vllm"]])]))
        charts += telemetry_plots(results_dir, out_dir)
    elif not HAVE_MPL:
        md.append("_matplotlib not installed; charts skipped (pip install matplotlib)._\n")

    charts = [c for c in charts if c]
    if charts:
        md.append("## Charts\n")
        for c in charts:
            md.append(f"- `{os.path.relpath(c, repo_root)}`")
        md.append("")

    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n".join(md))
    print(f"\nWrote {os.path.join(out_dir, 'summary.md')} and summary.json")
    if charts:
        print(f"Wrote {len(charts)} chart(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
