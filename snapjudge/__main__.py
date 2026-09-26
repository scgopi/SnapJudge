import argparse
import json
import sys
import time

from .model import DEFAULT_WEIGHTS, PACKAGE_ROOT, SnapJudge


def main():
    ap = argparse.ArgumentParser(prog="python -m snapjudge", description="Run SnapJudge on a JSON request.")
    ap.add_argument("request", help="path to a request JSON file, or - for stdin")
    ap.add_argument("--backbone", default=None, help="backbone directory (default: models/snapjudge-text, then models/snapjudge-vision)")
    ap.add_argument("--size", choices=["4b", "2b"], default="4b", help="4b (default, most accurate) or 2b (smaller, faster)")
    ap.add_argument("--weights", default=None, help="directory with heads and config.json (overrides --size)")
    ap.add_argument("--timing", action="store_true", help="warm up once, then report latency_ms")
    args = ap.parse_args()

    request = json.load(sys.stdin if args.request == "-" else open(args.request))
    weights = args.weights or (DEFAULT_WEIGHTS if args.size == "4b" else PACKAGE_ROOT / "weights-2b")
    judge = SnapJudge(backbone=args.backbone, weights=weights)
    if args.timing:
        judge.classify(request)
    start = time.perf_counter()
    response = judge.classify(request)
    if args.timing:
        response["latency_ms"] = round((time.perf_counter() - start) * 1000)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
