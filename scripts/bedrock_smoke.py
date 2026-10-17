#!/usr/bin/env python3
"""Smoke-test the Bedrock OpenAI-compatible "mantle" endpoint for Qwen3-32B.

Two jobs:
  1. Confirm the endpoint + token work and print a streamed TTFT, end-to-end
     latency and the ACTUAL number of output tokens (Bedrock cannot be forced
     to a fixed output length).
  2. `--probe-thinking`: find out which request parameter actually disables
     Qwen3's "thinking" phase on this endpoint, since the mechanism differs from
     vLLM. It sends the same prompt three ways and reports, for each, whether
     reasoning content came back.

Auth uses OPENAI_API_KEY (a short-lived Bedrock token, see make_bedrock_token.py).

    export OPENAI_API_KEY=$(python scripts/make_bedrock_token.py)
    python scripts/bedrock_smoke.py
    python scripts/bedrock_smoke.py --probe-thinking

Install once:  pip install openai
"""
import argparse
import os
import sys
import time


def _client(base_url: str):
    try:
        from openai import OpenAI
    except ImportError:
        sys.stderr.write("Missing dependency. Run: pip install openai\n")
        raise SystemExit(1)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.stderr.write("OPENAI_API_KEY is not set (expected a Bedrock token).\n")
        raise SystemExit(1)
    return OpenAI(base_url=base_url, api_key=api_key)


def _run_once(client, model, prompt, max_tokens, extra_body, stream=True):
    """Return (ttft_s, total_s, text, reasoning_text, usage_completion_tokens)."""
    messages = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    ttft = None
    text_parts, reasoning_parts = [], []
    completion_tokens = None

    if stream:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True},
            extra_body=(extra_body or None),
        )
        for chunk in resp:
            if chunk.usage is not None:
                completion_tokens = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # Some backends expose the thinking trace as `reasoning_content`.
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning_parts.append(rc)
                if ttft is None:
                    ttft = time.perf_counter() - t0
            if delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text_parts.append(delta.content)
    else:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, stream=False,
            extra_body=(extra_body or None),
        )
        msg = resp.choices[0].message
        text_parts.append(msg.content or "")
        rc = getattr(msg, "reasoning_content", None)
        if rc:
            reasoning_parts.append(rc)
        if resp.usage is not None:
            completion_tokens = resp.usage.completion_tokens
        ttft = time.perf_counter() - t0

    total = time.perf_counter() - t0
    return ttft, total, "".join(text_parts), "".join(reasoning_parts), completion_tokens


def _thought(text: str, reasoning: str) -> bool:
    """Heuristic: did the model 'think'?"""
    return bool(reasoning.strip()) or "<think>" in text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("BEDROCK_BASE_URL", "https://bedrock-mantle.eu-north-1.api.aws"))
    ap.add_argument("--model", default=os.environ.get("BEDROCK_MODEL", "qwen.qwen3-32b"))
    ap.add_argument("--prompt", default="In one sentence, what is a tensor core?")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-stream", action="store_true", help="disable streaming (no TTFT)")
    ap.add_argument("--probe-thinking", action="store_true",
                    help="try several ways to disable Qwen3 thinking and report what works")
    args = ap.parse_args()

    client = _client(args.base_url)
    stream = not args.no_stream

    if not args.probe_thinking:
        ttft, total, text, reasoning, ctoks = _run_once(
            client, args.model, args.prompt, args.max_tokens, extra_body=None, stream=stream)
        print(f"endpoint        : {args.base_url}")
        print(f"model           : {args.model}")
        print(f"ttft_s          : {ttft:.3f}" if ttft is not None else "ttft_s          : n/a")
        print(f"total_s         : {total:.3f}")
        print(f"output_tokens   : {ctoks if ctoks is not None else 'unknown (no usage returned)'}")
        print(f"thought?        : {_thought(text, reasoning)}")
        print("---- reply ----")
        print(text.strip()[:2000])
        return 0

    # Probe which knob disables thinking on this endpoint.
    variants = [
        ("baseline (nothing)", None),
        ("reasoning_effort=none", {"reasoning_effort": "none"}),
        ("chat_template_kwargs.enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}),
    ]
    print(f"Probing thinking control on {args.base_url} / {args.model}\n")
    print(f"{'variant':<48} {'ok':<4} {'thought':<8} {'out_tok':<8} ttft_s")
    print("-" * 82)
    for name, eb in variants:
        try:
            ttft, total, text, reasoning, ctoks = _run_once(
                client, args.model, args.prompt, args.max_tokens, extra_body=eb, stream=stream)
            ttft_s = f"{ttft:.3f}" if ttft is not None else "n/a"
            print(f"{name:<48} {'yes':<4} {str(_thought(text, reasoning)):<8} {str(ctoks):<8} {ttft_s}")
        except Exception as exc:  # noqa: BLE001 - smoke test wants the raw error
            msg = str(exc).replace("\n", " ")[:60]
            print(f"{name:<48} {'ERR':<4} {'-':<8} {'-':<8} {msg}")
    print("\nUse the variant that is accepted (ok=yes) AND shows thought=False as")
    print("BEDROCK_EXTRA_BODY in .env. If none works, document the asymmetry: vLLM")
    print("output has thinking disabled, Bedrock does not; compare per-token latency")
    print("and record the actual output tokens (the sweep already does).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
