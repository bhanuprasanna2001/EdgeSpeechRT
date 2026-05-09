#!/usr/bin/env python
"""Compare our EdgeSpeech-RT model against the GTCRN pretrained checkpoint on the VCTK test set.

GTCRN reference: Xiaobin Rong et al., GTCRN, ICASSP 2024.
Checkpoint: model_trained_on_vctk.tar from https://github.com/Xiaobin-Rong/gtcrn
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

# Allow importing gtcrn.py from the gtcrn_reference directory
sys.path.insert(0, str(Path(__file__).parent.parent / "gtcrn_reference"))

from edgespeech_rt.audio import enhance_waveform, si_sdr
from edgespeech_rt.dataset import read_audio_16k
from edgespeech_rt.model import build_model


# ---------------------------------------------------------------------------
# Our model inference
# ---------------------------------------------------------------------------

def run_edgespeech(
    clean_dir: Path,
    noisy_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    n_fft: int = 512,
    hop_size: int = 256,
) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = state.get("metadata", {})
    hidden_size = meta.get("hidden_size", 48) if isinstance(meta, dict) else 48

    model = build_model(hidden_size=hidden_size, n_fft=n_fft)
    model.load_state_dict(state.get("model", state))
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    noisy_paths = sorted(noisy_dir.glob("*.wav"))
    for path in noisy_paths:
        clean_path = clean_dir / path.name
        if not clean_path.exists():
            continue
        wav = torch.from_numpy(read_audio_16k(path)).unsqueeze(0)
        enhanced = enhance_waveform(model, wav, n_fft=n_fft, hop_size=hop_size)
        sf.write(output_dir / path.name, enhanced.squeeze(0).numpy(), 16000)
    print(f"EdgeSpeech-RT enhanced {len(noisy_paths)} files -> {output_dir}")


# ---------------------------------------------------------------------------
# GTCRN inference
# ---------------------------------------------------------------------------

def run_gtcrn(
    clean_dir: Path,
    noisy_dir: Path,
    checkpoint: Path,
    output_dir: Path,
) -> None:
    from gtcrn import GTCRN  # type: ignore[import]

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = GTCRN().eval()
    model.load_state_dict(ckpt["model"])

    window = torch.hann_window(512).pow(0.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    noisy_paths = sorted(noisy_dir.glob("*.wav"))

    with torch.no_grad():
        for path in noisy_paths:
            clean_path = clean_dir / path.name
            if not clean_path.exists():
                continue

            wav_np = read_audio_16k(path)
            wav = torch.from_numpy(wav_np).unsqueeze(0)  # [1, T]

            # STFT: [1, F, T] complex
            spec = torch.stft(wav, 512, 256, 512, window, return_complex=True)
            # Pack as [B, F, T, 2] for GTCRN
            spec_in = torch.stack([spec.real, spec.imag], dim=-1)

            spec_enh = model(spec_in)  # [B, F, T, 2]

            # iSTFT
            spec_enh_c = torch.complex(spec_enh[..., 0], spec_enh[..., 1])  # [B, F, T]
            enhanced = torch.istft(spec_enh_c, 512, 256, 512, window, length=wav.shape[-1])

            sf.write(output_dir / path.name, enhanced.squeeze(0).numpy(), 16000)

    print(f"GTCRN enhanced {len(noisy_paths)} files -> {output_dir}")


# ---------------------------------------------------------------------------
# Metric evaluation
# ---------------------------------------------------------------------------

def evaluate_dir(clean_dir: Path, enhanced_dir: Path, label: str) -> dict:
    pesq_scores, stoi_scores, sisdr_scores = [], [], []
    for clean_path in sorted(clean_dir.glob("*.wav")):
        enh_path = enhanced_dir / clean_path.name
        if not enh_path.exists():
            continue
        clean = read_audio_16k(clean_path)
        enhanced = read_audio_16k(enh_path)
        length = min(clean.size, enhanced.size)
        clean = clean[:length]
        enhanced = enhanced[:length]
        pesq_scores.append(pesq(16000, clean, enhanced, "wb"))
        stoi_scores.append(stoi(clean, enhanced, 16000, extended=False))
        sisdr_scores.append(si_sdr(clean, enhanced))

    n = len(pesq_scores)
    result = {
        "model": label,
        "n_files": n,
        "pesq": round(float(np.mean(pesq_scores)), 4),
        "stoi": round(float(np.mean(stoi_scores)), 4),
        "si_sdr": round(float(np.mean(sisdr_scores)), 4),
    }
    print(f"[{label:20s}]  PESQ={result['pesq']:.4f}  STOI={result['stoi']:.4f}  SI-SDR={result['si_sdr']:.4f}  (n={n})")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Compare EdgeSpeech-RT vs GTCRN on VCTK test set.")
    parser.add_argument("--clean-dir", default=str(root / "datasets/vctk-demand/raw/clean_testset_wav"))
    parser.add_argument("--noisy-dir", default=str(root / "datasets/vctk-demand/raw/noisy_testset_wav"))
    parser.add_argument("--edgespeech-checkpoint", default=str(root / "artifacts/edgespeech_rt.pt"))
    parser.add_argument("--gtcrn-checkpoint", default=str(root / "gtcrn_reference/model_trained_on_vctk.tar"))
    parser.add_argument("--output-csv", default=str(root / "benchmarks/comparison.csv"))
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=256)
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)

    artifacts_root = root / "artifacts"

    # Enhance with our model
    edgespeech_out = artifacts_root / "enhanced_edgespeech"
    run_edgespeech(clean_dir, noisy_dir, Path(args.edgespeech_checkpoint),
                   edgespeech_out, n_fft=args.n_fft, hop_size=args.hop_size)

    # Enhance with GTCRN
    gtcrn_out = artifacts_root / "enhanced_gtcrn"
    run_gtcrn(clean_dir, noisy_dir, Path(args.gtcrn_checkpoint), gtcrn_out)

    # Evaluate
    print("\n--- Evaluation ---")
    rows = [
        evaluate_dir(clean_dir, noisy_dir, "Noisy"),
        evaluate_dir(clean_dir, edgespeech_out, "EdgeSpeech-RT (ours)"),
        evaluate_dir(clean_dir, gtcrn_out, "GTCRN (pretrained)"),
    ]

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "n_files", "pesq", "stoi", "si_sdr"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
