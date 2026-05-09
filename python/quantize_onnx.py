#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from edgespeech_rt.config import load_yaml
from edgespeech_rt.audio import stft_features
from edgespeech_rt.dataset import read_audio_16k


class RandomSpectralCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        frames: int,
        num_bins: int = 257,
        hidden_size: int = 48,
        seed: int = 1337,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._remaining = frames
        self._num_bins = num_bins
        self._hidden_size = hidden_size

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        return {
            "mag_log": self._rng.random((1, 1, self._num_bins), dtype=np.float32),
            "h0": np.zeros((1, 1, self._hidden_size), dtype=np.float32),
        }


class WavSpectralCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        noisy_dir: str,
        max_files: int = 16,
        max_frames_per_file: int = 16,
        hidden_size: int = 48,
        n_fft: int = 512,
        hop_size: int = 256,
    ) -> None:
        self._hidden_size = hidden_size
        self._features: list[np.ndarray] = []
        for wav_path in sorted(Path(noisy_dir).glob("*.wav"))[:max_files]:
            wav = torch.from_numpy(read_audio_16k(wav_path)).unsqueeze(0)
            features, _ = stft_features(wav, n_fft=n_fft, hop_size=hop_size)
            frames = features.squeeze(0).cpu().numpy().astype(np.float32)
            step = max(1, len(frames) // max_frames_per_file)
            self._features.extend(frames[::step][:max_frames_per_file])
        if not self._features:
            raise FileNotFoundError(f"no calibration wav files found in {noisy_dir}")

    def get_next(self) -> dict[str, np.ndarray] | None:
        if not self._features:
            return None
        frame = self._features.pop(0)
        return {
            "mag_log": frame.reshape(1, 1, -1).astype(np.float32),
            "h0": np.zeros((1, 1, self._hidden_size), dtype=np.float32),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize exported ONNX models.")
    parser.add_argument("--config", default="configs/quant_int8_static.yaml")
    parser.add_argument("--mode", choices=["dynamic", "static"], default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--calibration-noisy-dir", default=None)
    parser.add_argument("--calibration-max-files", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    mode = args.mode or cfg.get("quantization", {}).get("mode", "static")
    input_path = Path(args.input or cfg["model_path"])
    output_path = Path(args.output or cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "dynamic":
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
    else:
        calibration = cfg.get("calibration", {})
        calibration_noisy_dir = args.calibration_noisy_dir or calibration.get("data_dir")
        if calibration_noisy_dir:
            reader = WavSpectralCalibrationReader(
                noisy_dir=calibration_noisy_dir,
                max_files=args.calibration_max_files,
                max_frames_per_file=max(1, int(args.frames or calibration.get("frames", 128)) // args.calibration_max_files),
                hidden_size=args.hidden_size,
                n_fft=args.n_fft,
                hop_size=args.hop_size,
            )
        else:
            reader = RandomSpectralCalibrationReader(
                frames=int(args.frames or calibration.get("frames", 128)),
                num_bins=int(cfg.get("num_bins", 257)),
                hidden_size=int(cfg.get("hidden_size", 48)),
                seed=int(calibration.get("seed", 1337)),
            )
        quantize_static(
            model_input=input_path,
            model_output=output_path,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
