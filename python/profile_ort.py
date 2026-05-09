#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ONNX Runtime frame latency.")
    parser.add_argument("model")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--num-bins", type=int, default=257)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.threads
    sess_options.inter_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(args.model, sess_options, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(1337)
    h = np.zeros((1, 1, args.hidden_size), dtype=np.float32)
    latencies: list[float] = []
    for index in range(args.warmup + args.frames):
        mag_log = rng.random((1, 1, args.num_bins), dtype=np.float32)
        start = time.perf_counter()
        mask, h = session.run(["mask", "hn"], {"mag_log": mag_log, "h0": h})
        _ = mask
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if index >= args.warmup:
            latencies.append(elapsed_ms)

    p50 = statistics.median(latencies)
    p95 = percentile(latencies, 95)
    rtf = (statistics.mean(latencies) / 1000.0) / 0.020
    row = {
        "model": args.model,
        "threads": args.threads,
        "frames": args.frames,
        "mean_ms": f"{statistics.mean(latencies):.4f}",
        "p50_ms": f"{p50:.4f}",
        "p95_ms": f"{p95:.4f}",
        "rtf": f"{rtf:.5f}",
    }

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    print(row)


if __name__ == "__main__":
    main()
