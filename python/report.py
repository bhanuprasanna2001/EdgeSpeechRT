#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_mapping(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected label=path/text, got {value!r}")
        key, raw = value.split("=", 1)
        result[key] = raw
    return result


def average_metrics(path: str | Path) -> tuple[int, dict[str, float]]:
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    if not rows:
        return 0, {"pesq": float("nan"), "stoi": float("nan"), "si_sdr": float("nan")}
    return len(rows), {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in ("pesq", "stoi", "si_sdr")
    }


def human_size(path: str | Path | None) -> tuple[str, str]:
    if not path:
        return "-", ""
    size = Path(path).stat().st_size
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB", str(size)
    return f"{size / 1024:.1f} KB", str(size)


def load_latency(path: str | Path | None) -> dict[str, dict[str, str]]:
    if not path or not Path(path).exists():
        return {}
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        model = row["model"]
        by_name[model] = row
        by_name[Path(model).name] = row
    return by_name


def markdown_table(rows: list[dict[str, str]]) -> str:
    columns = ["Model", "Params", "Size", "MAC/s", "PESQ", "STOI", "SI-SDR", "RTF", "p95 latency", "Notes"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["params"],
                    row["size"],
                    row["macs"],
                    row["pesq"],
                    row["stoi"],
                    row["si_sdr"],
                    row["rtf"],
                    row["p95_latency"],
                    row["notes"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark summary CSV/Markdown from measured metric and latency files.")
    parser.add_argument("--metric", action="append", required=True, help="label=metrics_csv")
    parser.add_argument("--model", action="append", default=[], help="label=onnx_or_checkpoint_path")
    parser.add_argument("--note", action="append", default=[], help="label=note")
    parser.add_argument("--metadata", default="artifacts/edgespeech_rt.json")
    parser.add_argument("--latency-csv", default="benchmarks/latency.csv")
    parser.add_argument("--output-csv", default="benchmarks/summary.csv")
    parser.add_argument("--output-md", default="benchmarks/summary.md")
    parser.add_argument("--latency-md", default="benchmarks/latency_report.md")
    args = parser.parse_args()

    metric_paths = parse_mapping(args.metric)
    model_paths = parse_mapping(args.model)
    notes = parse_mapping(args.note)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8")) if Path(args.metadata).exists() else {}
    params = metadata.get("parameters", "")
    macs = metadata.get("estimated_macs_per_second", "")
    latency_by_model = load_latency(args.latency_csv)

    rows: list[dict[str, str]] = []
    csv_rows: list[dict[str, str]] = []
    for label, metric_path in metric_paths.items():
        count, metrics = average_metrics(metric_path)
        model_path = model_paths.get(label)
        size_label, size_bytes = human_size(model_path)
        latency = latency_by_model.get(model_path or "") or latency_by_model.get(Path(model_path).name if model_path else "")
        row = {
            "model": label,
            "params": f"{int(params):,}" if params and model_path else "-",
            "size": size_label,
            "macs": f"{int(macs):,}" if macs and model_path else "-",
            "pesq": f"{metrics['pesq']:.4f}",
            "stoi": f"{metrics['stoi']:.4f}",
            "si_sdr": f"{metrics['si_sdr']:.4f}",
            "rtf": latency["rtf"] if latency else "-",
            "p95_latency": f"{latency['p95_ms']} ms" if latency else "-",
            "notes": notes.get(label, f"{count} files"),
        }
        rows.append(row)
        csv_rows.append(
            {
                "model": label,
                "files": str(count),
                "params": row["params"].replace(",", ""),
                "size": size_label,
                "size_bytes": size_bytes,
                "macs": row["macs"].replace(",", ""),
                "pesq": row["pesq"],
                "stoi": row["stoi"],
                "si_sdr": row["si_sdr"],
                "rtf": row["rtf"],
                "p95_latency_ms": latency["p95_ms"] if latency else "",
                "notes": row["notes"],
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    output_md = Path(args.output_md)
    output_md.write_text(markdown_table(rows), encoding="utf-8")

    latency_md = Path(args.latency_md)
    latency_rows = list(csv.DictReader(Path(args.latency_csv).open(newline="", encoding="utf-8"))) if Path(args.latency_csv).exists() else []
    latency_lines = [
        "# Latency Report",
        "",
        "ONNX Runtime CPU profiling uses one recurrent spectral frame per call after warmup.",
        "",
        "| Model | Threads | Frames | Mean ms | p50 ms | p95 ms | RTF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in latency_rows:
        latency_lines.append(
            f"| {Path(row['model']).name} | {row['threads']} | {row['frames']} | {row['mean_ms']} | "
            f"{row['p50_ms']} | {row['p95_ms']} | {row['rtf']} |"
        )
    latency_md.write_text("\n".join(latency_lines) + "\n", encoding="utf-8")

    print(f"wrote {output_csv}")
    print(f"wrote {output_md}")
    print(f"wrote {latency_md}")


if __name__ == "__main__":
    main()
