#!/usr/bin/env python3
"""Print a short-lived Amazon Bedrock bearer token to stdout. LOCAL USE ONLY.

The token is derived from your *local* AWS credentials (default credential
chain) via `aws-bedrock-token-generator` and is valid for a limited time (max
12 h). Export it as OPENAI_API_KEY for the OpenAI-compatible Bedrock endpoint:

    export OPENAI_API_KEY=$(python scripts/make_bedrock_token.py)

Never run this on the GPU instance and never commit the token: the whole point
is that AWS credentials stay on your laptop and only a temporary token travels.

Install once:  pip install aws-bedrock-token-generator
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-north-1"),
                    help="AWS region (default: $AWS_REGION or eu-north-1)")
    ap.add_argument("--expires-in", type=int, default=None,
                    help="token lifetime in seconds (max 43200 = 12 h)")
    args = ap.parse_args()

    try:
        from aws_bedrock_token_generator import provide_token
    except ImportError:
        sys.stderr.write("Missing dependency. Run: pip install aws-bedrock-token-generator\n")
        return 1

    # The region can be passed as a kwarg; older releases only read $AWS_REGION.
    os.environ.setdefault("AWS_REGION", args.region)
    kwargs = {}
    if args.expires_in is not None:
        kwargs["expires_in"] = args.expires_in
    try:
        token = provide_token(region=args.region, **kwargs)
    except TypeError:
        # Signature differs across versions; fall back to the zero-arg form.
        token = provide_token(**kwargs) if kwargs else provide_token()

    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
