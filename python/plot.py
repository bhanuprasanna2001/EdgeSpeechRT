#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

from edgespeech_rt.dataset import read_audio_16k


def read_rows(path: str | Path) -> list[dict[str, str]]:
    return list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))


def plot_training(args: argparse.Namespace) -> None:
    rows = read_rows(args.csv)
    epochs = [int(row["epoch"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    val = [float(row["val_loss"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, marker="o", label="train")
    ax.plot(epochs, val, marker="o", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, args.output)


def plot_metrics(args: argparse.Namespace) -> None:
    rows = read_rows(args.summary_csv)
    labels = [row["model"].replace("_", " ") for row in rows]
    metrics = ["pesq", "stoi", "si_sdr"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, metrics, strict=True):
        values = [float(row[metric]) for row in rows]
        bars = ax.bar(labels, values, color=["#59656f", "#2f7d6d", "#4775b8", "#8a6f3f"][: len(labels)])
        ax.set_title(metric.upper().replace("_", "-"))
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    save(fig, args.output)


def spectrogram_db(samples: np.ndarray, sample_rate: int = 16000, n_fft: int = 512, hop_size: int = 320) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs, times, spec = stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_size,
        boundary=None,
        padded=False,
    )
    return freqs, times, 20.0 * np.log10(np.maximum(np.abs(spec), 1.0e-8))


def dbfs_rms(samples: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(samples))) + 1.0e-8))


def plot_spectrogram(args: argparse.Namespace) -> None:
    noisy = read_audio_16k(args.noisy)
    clean = read_audio_16k(args.clean)
    enhanced = read_audio_16k(args.enhanced)
    length = min(noisy.size, clean.size, enhanced.size)
    noisy = noisy[:length]
    clean = clean[:length]
    enhanced = enhanced[:length]

    if args.start_seconds > 0.0 or args.duration_seconds:
        start = int(args.start_seconds * 16000)
        end = length if args.duration_seconds is None else start + int(args.duration_seconds * 16000)
        noisy = noisy[start:end]
        clean = clean[start:end]
        enhanced = enhanced[start:end]

    noisy_residual = noisy - clean
    enhanced_residual = enhanced - clean
    panels = [
        ("Noisy", noisy),
        ("Enhanced", enhanced),
        ("Clean", clean),
        (f"Noisy residual ({dbfs_rms(noisy_residual):.1f} dBFS RMS)", noisy_residual),
        (f"Enhanced residual ({dbfs_rms(enhanced_residual):.1f} dBFS RMS)", enhanced_residual),
    ]
    specs = [spectrogram_db(samples) for _, samples in panels]
    top_values = np.concatenate([spec.ravel() for _, _, spec in specs[:3]])
    residual_values = np.concatenate([spec.ravel() for _, _, spec in specs[3:]])
    top_vmin, top_vmax = np.percentile(top_values, [1, 99])
    residual_vmin, residual_vmax = np.percentile(residual_values, [1, 99])

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    last_top = None
    last_residual = None
    for index, (ax, (title, _), (freqs, times, spec_db)) in enumerate(zip(axes, panels, specs, strict=True)):
        if index < 3:
            image = ax.pcolormesh(times, freqs, spec_db, shading="auto", cmap="magma", vmin=top_vmin, vmax=top_vmax)
            last_top = image
        else:
            image = ax.pcolormesh(times, freqs, spec_db, shading="auto", cmap="magma", vmin=residual_vmin, vmax=residual_vmax)
            last_residual = image
        ax.set_title(title)
        ax.set_ylabel("Hz")
        ax.set_ylim(0, 8000)
    axes[-1].set_xlabel("time [s]")
    if last_top is not None:
        fig.colorbar(last_top, ax=axes[:3], label="dB", shrink=0.86)
    if last_residual is not None:
        fig.colorbar(last_residual, ax=axes[3:], label="dB", shrink=0.75)
    save(fig, args.output)


def save(fig: plt.Figure, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EdgeSpeech-RT benchmark plots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    training = subparsers.add_parser("training", help="plot train/validation loss")
    training.add_argument("--csv", default="benchmarks/training_curve.csv")
    training.add_argument("--output", default="assets/plots/training_curve.png")
    training.set_defaults(func=plot_training)

    metrics = subparsers.add_parser("metrics", help="plot metric summary bars")
    metrics.add_argument("--summary-csv", default="benchmarks/summary.csv")
    metrics.add_argument("--output", default="assets/plots/metric_summary.png")
    metrics.set_defaults(func=plot_metrics)

    spectrogram = subparsers.add_parser("spectrogram", help="plot noisy/enhanced/clean/residual spectrograms")
    spectrogram.add_argument("--noisy", required=True)
    spectrogram.add_argument("--clean", required=True)
    spectrogram.add_argument("--enhanced", required=True)
    spectrogram.add_argument("--output", default="assets/spectrograms/comparison.png")
    spectrogram.add_argument("--start-seconds", type=float, default=0.0)
    spectrogram.add_argument("--duration-seconds", type=float, default=None)
    spectrogram.set_defaults(func=plot_spectrogram)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
