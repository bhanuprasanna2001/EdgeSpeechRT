from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MaskNetConfig:
    """Configuration for the compact causal spectral masking network."""

    n_fft: int = 512
    hidden_size: int = 48

    @property
    def num_bins(self) -> int:
        return self.n_fft // 2 + 1


class StreamingGTCRNStyleMasker(nn.Module):
    """Small recurrent mask estimator exported to ONNX for C++ streaming.

    The C++ SDK owns STFT/iSTFT and overlap-add. This model consumes
    ``log1p(abs(STFT))`` frames with shape ``[batch, frames, bins]`` and returns
    a real-valued spectral mask plus the updated recurrent state.
    """

    def __init__(self, config: MaskNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or MaskNetConfig()
        bins = self.config.num_bins
        hidden = self.config.hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(bins, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, bins),
            nn.Sigmoid(),
        )

    def forward(
        self, mag_log: torch.Tensor, h0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one or more streaming frames.

        Args:
            mag_log: ``[batch, frames, bins]`` log-magnitude features.
            h0: ``[1, batch, hidden_size]`` recurrent state.
        """

        encoded = self.encoder(mag_log)
        recurrent, hn = self.gru(encoded, h0)
        mask = self.decoder(recurrent)
        return mask, hn


class FakeQuantizedMasker(nn.Module):
    """QAT-friendly wrapper that inserts fake quantization around activations."""

    def __init__(self, base: StreamingGTCRNStyleMasker) -> None:
        super().__init__()
        self.base = base
        self.input_quant = torch.ao.quantization.FakeQuantize()
        self.output_quant = torch.ao.quantization.FakeQuantize()

    def forward(
        self, mag_log: torch.Tensor, h0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantized_input = self.input_quant(mag_log)
        mask, hn = self.base(quantized_input, h0)
        return self.output_quant(mask), hn


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_macs_per_second(
    config: MaskNetConfig, sample_rate: int = 16000, hop_size: int = 256
) -> int:
    """Rough dense-layer/GRU MAC estimate for the model at streaming rate."""

    bins = config.num_bins
    hidden = config.hidden_size
    frames_per_second = sample_rate / hop_size

    encoder = bins * hidden + hidden * hidden
    gru = 3 * (hidden * hidden + hidden * hidden)
    decoder = hidden * hidden + hidden * bins
    return int((encoder + gru + decoder) * frames_per_second)


def build_model(hidden_size: int = 48, n_fft: int = 512) -> StreamingGTCRNStyleMasker:
    return StreamingGTCRNStyleMasker(MaskNetConfig(n_fft=n_fft, hidden_size=hidden_size))
