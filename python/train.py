#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edgespeech_rt.audio import stft_features
from edgespeech_rt.dataset import PairedSpeechDataset, discover_pairs, split_pairs
from edgespeech_rt.model import build_model, count_parameters, estimate_macs_per_second


def denoising_targets(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    n_fft: int,
    hop_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return features, noisy magnitude, clean magnitude, and a denoising mask target.

    A plain ideal-ratio-mask target is often too conservative for VoiceBank-DEMAND because
    speech and noise magnitudes overlap. The blended Wiener/IRM target encourages real
    attenuation in noise-dominant bins while the spectral reconstruction loss protects speech.
    """

    clean_features, clean_spec = stft_features(clean, n_fft=n_fft, hop_size=hop_size)
    noisy_features, noisy_spec = stft_features(noisy, n_fft=n_fft, hop_size=hop_size)
    _ = clean_features
    clean_mag = clean_spec.abs()
    noisy_mag = noisy_spec.abs()
    noise_mag = (noisy_spec - clean_spec).abs()
    irm = torch.clamp(clean_mag / (noisy_mag + 1.0e-6), 0.0, 1.0)
    wiener = clean_mag.square() / (clean_mag.square() + noise_mag.square() + 1.0e-6)
    target_mask = torch.clamp(0.35 * irm + 0.65 * wiener, 0.0, 1.0)
    return noisy_features, noisy_mag, clean_mag, target_mask


def denoising_loss(
    mask: torch.Tensor,
    noisy_mag: torch.Tensor,
    clean_mag: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    enhanced_mag = noisy_mag * mask
    log_mag_loss = F.smooth_l1_loss(torch.log1p(enhanced_mag), torch.log1p(clean_mag))
    mask_loss = F.mse_loss(mask, target_mask)
    noise_ratio = torch.clamp((noisy_mag - clean_mag).abs() / (noisy_mag + clean_mag + 1.0e-6), 0.0, 1.0)
    suppression_loss = torch.mean(mask * noise_ratio)
    return log_mag_loss + 0.5 * mask_loss + 0.02 * suppression_loss


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    n_fft: int,
    hop_size: int,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        features, noisy_mag, clean_mag, target_mask = denoising_targets(
            clean, noisy, n_fft=n_fft, hop_size=hop_size
        )
        h0 = torch.zeros(1, features.shape[0], model.config.hidden_size, device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        mask, _ = model(features, h0)
        loss = denoising_loss(mask, noisy_mag, clean_mag, target_mask)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += float(loss.detach()) * features.shape[0]
        count += features.shape[0]
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EdgeSpeech-RT on paired clean/noisy WAV files.")
    parser.add_argument("--clean-dir", default="datasets/vctk-demand/raw/clean_trainset_28spk_wav")
    parser.add_argument("--noisy-dir", default="datasets/vctk-demand/raw/noisy_trainset_28spk_wav")
    parser.add_argument("--output", default="artifacts/edgespeech_rt.pt")
    parser.add_argument("--history-csv", default="benchmarks/training_curve.csv")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--max-files", type=int, default=4096)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pairs = discover_pairs(args.clean_dir, args.noisy_dir, limit=args.max_files)
    train_pairs, val_pairs = split_pairs(pairs, val_fraction=args.val_fraction, seed=args.seed)
    if not val_pairs:
        val_pairs = train_pairs[:1]

    train_ds = PairedSpeechDataset(train_pairs, segment_seconds=args.segment_seconds, random_crop=True, seed=args.seed)
    val_ds = PairedSpeechDataset(val_pairs, segment_seconds=args.segment_seconds, random_crop=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(hidden_size=args.hidden_size, n_fft=args.n_fft).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_csv = Path(args.history_csv)
    history_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.n_fft, args.hop_size, device)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, args.n_fft, args.hop_size, device)
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()
    metadata = {
        "dataset": "VoiceBank-DEMAND",
        "train_files": len(train_pairs),
        "val_files": len(val_pairs),
        "epochs": args.epochs,
        "segment_seconds": args.segment_seconds,
        "parameters": count_parameters(model),
        "estimated_macs_per_second": estimate_macs_per_second(model.config),
        "best_val_loss": best_val,
        "loss": "log-magnitude reconstruction + blended Wiener/IRM mask + mild suppression regularizer",
    }
    torch.save({"model": model.state_dict(), "metadata": metadata}, output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    print(f"wrote {history_csv}")


if __name__ == "__main__":
    main()
