#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from edgespeech_rt.audio import enhance_waveform, istft_overlap_add, stft_features
from edgespeech_rt.dataset import discover_pairs, read_audio_16k
from edgespeech_rt.model import build_model


@torch.no_grad()
def enhance_onnx(
    waveform: torch.Tensor,
    session: ort.InferenceSession,
    hidden_size: int = 48,
    n_fft: int = 512,
    hop_size: int = 256,
) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    features, spec = stft_features(waveform, n_fft=n_fft, hop_size=hop_size)
    h0 = np.zeros((1, waveform.shape[0], hidden_size), dtype=np.float32)
    mask, _ = session.run(
        ["mask", "hn"],
        {"mag_log": features.cpu().numpy().astype(np.float32), "h0": h0},
    )
    mask_t = torch.from_numpy(mask).to(spec.device)
    return istft_overlap_add(spec * mask_t, length=waveform.shape[-1], n_fft=n_fft, hop_size=hop_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance VoiceBank-DEMAND WAV files with a checkpoint or ONNX model.")
    parser.add_argument("--clean-dir", default="datasets/vctk-demand/raw/clean_testset_wav")
    parser.add_argument("--noisy-dir", default="datasets/vctk-demand/raw/noisy_testset_wav")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=256)
    args = parser.parse_args()

    if bool(args.checkpoint) == bool(args.onnx):
        raise ValueError("provide exactly one of --checkpoint or --onnx")

    pairs = discover_pairs(args.clean_dir, args.noisy_dir, limit=args.max_files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model = build_model(hidden_size=args.hidden_size, n_fft=args.n_fft)
        model.load_state_dict(state.get("model", state))
        model.eval()

    session = None
    if args.onnx:
        session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    for index, pair in enumerate(pairs, start=1):
        waveform = torch.from_numpy(read_audio_16k(pair.noisy)).unsqueeze(0)
        if model is not None:
            enhanced = enhance_waveform(model, waveform, n_fft=args.n_fft, hop_size=args.hop_size)
        else:
            assert session is not None
            enhanced = enhance_onnx(
                waveform,
                session,
                hidden_size=args.hidden_size,
                n_fft=args.n_fft,
                hop_size=args.hop_size,
            )
        sf.write(output_dir / pair.name, enhanced.squeeze(0).cpu().numpy(), 16000)
        print(f"[{index}/{len(pairs)}] {pair.name}")


if __name__ == "__main__":
    main()
