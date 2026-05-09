#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pesq import pesq
from pystoi import stoi

from edgespeech_rt.audio import si_sdr
from edgespeech_rt.dataset import read_audio_16k


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PESQ, STOI, and SI-SDR for paired WAV files.")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--enhanced-dir", required=True)
    parser.add_argument("--output", default="benchmarks/metrics.csv")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    enhanced_dir = Path(args.enhanced_dir)
    clean_paths = sorted(clean_dir.glob("*.wav"))
    if args.max_files is not None:
        clean_paths = clean_paths[: args.max_files]

    rows: list[dict[str, str]] = []
    for clean_path in clean_paths:
        enhanced_path = enhanced_dir / clean_path.name
        if not enhanced_path.exists():
            continue
        clean = read_audio_16k(clean_path)
        enhanced = read_audio_16k(enhanced_path)
        length = min(clean.size, enhanced.size)
        clean = clean[:length]
        enhanced = enhanced[:length]
        rows.append(
            {
                "file": clean_path.name,
                "pesq": f"{pesq(16000, clean, enhanced, 'wb'):.4f}",
                "stoi": f"{stoi(clean, enhanced, 16000, extended=False):.4f}",
                "si_sdr": f"{si_sdr(clean, enhanced):.4f}",
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "pesq", "stoi", "si_sdr"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} with {len(rows)} rows")


if __name__ == "__main__":
    main()
