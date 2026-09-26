from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure API latency using representative JSONL requests.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL; each line is an API request body")
    parser.add_argument("--url", default="http://localhost:8000/v1/predict")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else None,
    }


def post(url: str, payload: bytes, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, data=payload, headers={"content-type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"API returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:500]}") from exc
    return {"round_trip_ms": (time.perf_counter() - started) * 1000, **body}


def main() -> None:
    args = parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("--requests and --concurrency must be positive")
    payloads = [line.strip().encode("utf-8") for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not payloads:
        raise SystemExit("Input JSONL contains no request bodies")
    decoded = [json.loads(payload) for payload in payloads]
    if any(not isinstance(body, dict) or "transaction" not in body for body in decoded):
        raise SystemExit('Each JSONL line must be an object containing a "transaction" field')
    if args.requests > len(payloads):
        raise SystemExit("Provide at least --requests distinct representative rows to avoid cache reuse")

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(post, args.url, payloads[i], args.timeout) for i in range(args.requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    elapsed = time.perf_counter() - started
    server_uncached = [float(item["inference_ms"]) for item in results if not item.get("cache_hit", False)]
    report = {
        "requests": len(results),
        "concurrency": args.concurrency,
        "elapsed_seconds": round(elapsed, 3),
        "throughput_requests_per_second": round(len(results) / elapsed, 2),
        "round_trip": summarize([float(item["round_trip_ms"]) for item in results]),
        "server_inference_uncached": summarize(server_uncached),
        "cache_hit_count": sum(bool(item.get("cache_hit")) for item in results),
        "model_versions": sorted({str(item.get("model_version")) for item in results}),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
