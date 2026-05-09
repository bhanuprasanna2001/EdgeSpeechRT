from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AudioPair:
    clean: Path
    noisy: Path
    name: str


def discover_pairs(clean_dir: str | Path, noisy_dir: str | Path, limit: int | None = None) -> list[AudioPair]:
    clean_dir = Path(clean_dir)
    noisy_dir = Path(noisy_dir)
    noisy_by_name = {path.name: path for path in noisy_dir.glob("*.wav")}
    pairs = [
        AudioPair(clean=clean_path, noisy=noisy_by_name[clean_path.name], name=clean_path.name)
        for clean_path in sorted(clean_dir.glob("*.wav"))
        if clean_path.name in noisy_by_name
    ]
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise FileNotFoundError(f"no paired wav files found under {clean_dir} and {noisy_dir}")
    return pairs


def split_pairs(
    pairs: list[AudioPair],
    val_fraction: float = 0.1,
    seed: int = 1337,
) -> tuple[list[AudioPair], list[AudioPair]]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def read_audio_16k(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    if sr != sample_rate:
        gcd = np.gcd(sr, sample_rate)
        mono = resample_poly(mono, sample_rate // gcd, sr // gcd).astype(np.float32)
    return mono


class PairedSpeechDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        pairs: list[AudioPair],
        sample_rate: int = 16000,
        segment_seconds: float = 1.0,
        random_crop: bool = True,
        seed: int = 1337,
    ) -> None:
        self.pairs = pairs
        self.sample_rate = sample_rate
        self.segment_samples = int(round(segment_seconds * sample_rate))
        self.random_crop = random_crop
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        clean = read_audio_16k(pair.clean, self.sample_rate)
        noisy = read_audio_16k(pair.noisy, self.sample_rate)
        length = min(clean.size, noisy.size)
        clean = clean[:length]
        noisy = noisy[:length]
        clean, noisy = self._crop_or_pad(clean, noisy)
        return torch.from_numpy(noisy), torch.from_numpy(clean)

    def _crop_or_pad(self, clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if clean.size >= self.segment_samples:
            max_start = clean.size - self.segment_samples
            start = self.rng.randint(0, max_start) if self.random_crop and max_start else 0
            end = start + self.segment_samples
            return clean[start:end].astype(np.float32), noisy[start:end].astype(np.float32)
        pad = self.segment_samples - clean.size
        return (
            np.pad(clean, (0, pad)).astype(np.float32),
            np.pad(noisy, (0, pad)).astype(np.float32),
        )
