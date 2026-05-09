#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from edgespeech_rt.audio import stft_features
from edgespeech_rt.dataset import PairedSpeechDataset, discover_pairs, split_pairs
from edgespeech_rt.model import build_model, count_parameters, estimate_macs_per_second


class HybridLoss(torch.nn.Module):
    """GTCRN-style hybrid loss combining compressed complex STFT, compressed magnitude, and SI-SNR.

    Reference: Xiaobin Rong et al., GTCRN, ICASSP 2024.
    Loss = 30*(compressed_real_MSE + compressed_imag_MSE) + 70*compressed_mag_MSE + SI-SNR
    """

    def __init__(self, n_fft: int = 512, hop_size: int = 256) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size

    def forward(self, enhanced_spec: torch.Tensor, clean_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced_spec: complex [B, T, F] — noisy_spec * predicted_mask
            clean_spec:    complex [B, T, F] — clean reference
        """
        pred_real = enhanced_spec.real
        pred_imag = enhanced_spec.imag
        true_real = clean_spec.real
        true_imag = clean_spec.imag

        pred_mag = torch.sqrt(pred_real**2 + pred_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_real**2 + true_imag**2 + 1e-12)

        # Power-0.7 compression: phase-aware complex loss
        inv_pred_mag = pred_mag**0.7 + 1e-12
        pred_real_c = pred_real / inv_pred_mag
        pred_imag_c = pred_imag / inv_pred_mag
        inv_true_mag = true_mag**0.7 + 1e-12
        true_real_c = true_real / inv_true_mag
        true_imag_c = true_imag / inv_true_mag

        real_loss = F.mse_loss(pred_real_c, true_real_c)
        imag_loss = F.mse_loss(pred_imag_c, true_imag_c)

        # Power-0.3 compression: perceptually-weighted magnitude loss
        mag_loss = F.mse_loss(pred_mag**0.3, true_mag**0.3)

        spectral = 30.0 * (real_loss + imag_loss) + 70.0 * mag_loss

        sisnr = self._sisnr(enhanced_spec, clean_spec)
        return spectral + sisnr

    def _sisnr(self, enhanced_spec: torch.Tensor, clean_spec: torch.Tensor) -> torch.Tensor:
        """Time-domain SI-SNR via iSTFT. Falls back to zero if iSTFT is unsupported on device."""
        try:
            window = torch.hann_window(self.n_fft, periodic=False, device=enhanced_spec.device)
            # [B, T, F] -> [B, F, T] required by torch.istft
            y_pred = torch.istft(
                enhanced_spec.transpose(1, 2).contiguous(),
                self.n_fft, self.hop_size, self.n_fft,
                window=window, center=False,
            )
            y_true = torch.istft(
                clean_spec.transpose(1, 2).contiguous(),
                self.n_fft, self.hop_size, self.n_fft,
                window=window, center=False,
            )
            target = (
                torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true
                / (torch.sum(y_true**2, dim=-1, keepdim=True) + 1e-8)
            )
            return -torch.log10(
                torch.norm(target, dim=-1)**2
                / (torch.norm(y_pred - target, dim=-1)**2 + 1e-8)
                + 1e-8
            ).mean()
        except Exception:
            # Spectral losses still backpropagate; SI-SNR skipped on this device
            return enhanced_spec.real.mean() * 0.0


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: HybridLoss,
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

        noisy_features, noisy_spec = stft_features(noisy, n_fft=n_fft, hop_size=hop_size)
        _, clean_spec = stft_features(clean, n_fft=n_fft, hop_size=hop_size)

        h0 = torch.zeros(1, noisy.shape[0], model.config.hidden_size, device=device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        mask, _ = model(noisy_features, h0)

        # Apply real mask to complex spectrum — preserves phase, suppresses magnitude
        enhanced_spec = noisy_spec * mask

        loss = criterion(enhanced_spec, clean_spec)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total += float(loss.detach()) * noisy.shape[0]
        count += noisy.shape[0]

    return total / max(1, count)


def _auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EdgeSpeech-RT on paired clean/noisy WAV files.")
    parser.add_argument("--clean-dir", default="datasets/vctk-demand/raw/clean_trainset_28spk_wav")
    parser.add_argument("--noisy-dir", default="datasets/vctk-demand/raw/noisy_trainset_28spk_wav")
    parser.add_argument("--output", default="artifacts/edgespeech_rt.pt")
    parser.add_argument("--history-csv", default="benchmarks/training_curve.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--lr-min", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = _auto_device()
    else:
        device = torch.device(args.device)
    print(f"device: {device}")

    pairs = discover_pairs(args.clean_dir, args.noisy_dir)
    train_pairs, val_pairs = split_pairs(pairs, val_fraction=args.val_fraction, seed=args.seed)
    if not val_pairs:
        val_pairs = train_pairs[:1]
    print(f"train={len(train_pairs)}  val={len(val_pairs)}")

    train_ds = PairedSpeechDataset(
        train_pairs, segment_seconds=args.segment_seconds, random_crop=True, seed=args.seed
    )
    val_ds = PairedSpeechDataset(
        val_pairs, segment_seconds=args.segment_seconds, random_crop=False, seed=args.seed
    )
    # num_workers=0 is required for MPS stability on macOS
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(hidden_size=args.hidden_size, n_fft=args.n_fft).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    criterion = HybridLoss(n_fft=args.n_fft, hop_size=args.hop_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_csv = Path(args.history_csv)
    history_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    best_val = float("inf")
    best_state: dict | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, args.n_fft, args.hop_size, device
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model, val_loader, None, criterion, args.n_fft, args.hop_size, device
            )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })
        marker = " *" if val_loss < best_val else ""
        print(
            f"epoch={epoch:3d}/{args.epochs}"
            f"  train={train_loss:.5f}"
            f"  val={val_loss:.5f}"
            f"  lr={current_lr:.2e}"
            f"{marker}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()

    metadata = {
        "dataset": "VoiceBank-DEMAND (full 28-speaker set)",
        "train_files": len(train_pairs),
        "val_files": len(val_pairs),
        "epochs": args.epochs,
        "hop_size": args.hop_size,
        "n_fft": args.n_fft,
        "segment_seconds": args.segment_seconds,
        "parameters": count_parameters(model),
        "estimated_macs_per_second": estimate_macs_per_second(model.config, hop_size=args.hop_size),
        "best_val_epoch": next(
            (r["epoch"] for r in reversed(rows) if abs(r["val_loss"] - best_val) < 1e-9),
            args.epochs,
        ),
        "best_val_loss": best_val,
        "loss": "HybridLoss: 30*(compressed-complex real+imag MSE) + 70*(compressed-mag MSE) + SI-SNR",
        "optimizer": "AdamW(lr=8e-4, weight_decay=1e-4)",
        "scheduler": f"CosineAnnealingLR(T_max={args.epochs}, eta_min={args.lr_min})",
    }
    torch.save({"model": model.state_dict(), "metadata": metadata}, output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {output}")
    print(f"wrote {history_csv}")
    print(f"best val loss: {best_val:.6f} @ epoch {metadata['best_val_epoch']}")


if __name__ == "__main__":
    main()
