from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from edgespeech_rt.model import StreamingGTCRNStyleMasker


def make_window(n_fft: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.hann_window(n_fft, periodic=False, device=device)


def stft_features(
    waveform: torch.Tensor, n_fft: int = 512, hop_size: int = 320
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log-magnitude features and complex STFT for mono waveforms."""

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    window = make_window(n_fft, waveform.device)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    spec = spec.transpose(1, 2)
    mag_log = torch.log1p(spec.abs())
    return mag_log, spec


def istft_overlap_add(
    spec: torch.Tensor,
    length: int,
    n_fft: int = 512,
    hop_size: int = 320,
    center: bool = True,
) -> torch.Tensor:
    """Inverse STFT using explicit overlap-add normalization.

    This avoids backend-specific `torch.istft` NOLA checks for deployment hops
    such as 512/320 while preserving the same analysis window convention.
    """

    if spec.ndim != 3:
        raise ValueError("expected spec shape [batch, frames, bins]")
    batch, frames, _ = spec.shape
    window = make_window(n_fft, spec.device).to(spec.real.dtype)
    total = n_fft + hop_size * (frames - 1)
    output = torch.zeros(batch, total, dtype=spec.real.dtype, device=spec.device)
    norm = torch.zeros(total, dtype=spec.real.dtype, device=spec.device)
    for frame_index in range(frames):
        frame = torch.fft.irfft(spec[:, frame_index, :], n=n_fft)
        start = frame_index * hop_size
        output[:, start : start + n_fft] += frame * window
        norm[start : start + n_fft] += window * window
    output = output / torch.clamp(norm, min=1.0e-8)
    if center:
        offset = n_fft // 2
        output = output[:, offset : offset + length]
    return output[..., :length]


@torch.no_grad()
def enhance_waveform(
    model: StreamingGTCRNStyleMasker,
    waveform: torch.Tensor,
    n_fft: int = 512,
    hop_size: int = 320,
) -> torch.Tensor:
    """Enhance a waveform with the same mask contract used by the C++ SDK."""

    model.eval()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    features, spec = stft_features(waveform, n_fft=n_fft, hop_size=hop_size)
    h0 = torch.zeros(1, waveform.shape[0], model.config.hidden_size, device=waveform.device)
    mask, _ = model(features, h0)
    enhanced_spec = spec * mask
    return istft_overlap_add(enhanced_spec, length=waveform.shape[-1], n_fft=n_fft, hop_size=hop_size)


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB."""

    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    length = min(reference.size, estimate.size)
    reference = reference[:length]
    estimate = estimate[:length]
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    projection = np.dot(estimate, reference) * reference / (np.dot(reference, reference) + eps)
    noise = estimate - projection
    return float(10.0 * np.log10((np.sum(projection**2) + eps) / (np.sum(noise**2) + eps)))


def synthetic_clean_noisy(
    seconds: float = 1.0,
    sample_rate: int = 16000,
    snr_db: float = 5.0,
    seed: int = 1337,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic speech-like fixture for tests and smoke runs."""

    rng = np.random.default_rng(seed)
    samples = int(seconds * sample_rate)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    envelope = 0.55 + 0.45 * np.sin(2.0 * math.pi * 2.3 * t)
    clean = envelope * (
        0.45 * np.sin(2.0 * math.pi * 180.0 * t)
        + 0.25 * np.sin(2.0 * math.pi * 360.0 * t)
        + 0.12 * np.sin(2.0 * math.pi * 720.0 * t)
    )
    noise = rng.normal(0.0, 1.0, size=samples).astype(np.float32)
    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    scale = math.sqrt(clean_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    noisy = clean + scale * noise
    return clean.astype(np.float32), noisy.astype(np.float32)


def write_float_text(path: str | Path, values: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(values, dtype=np.float32), fmt="%.9g")
