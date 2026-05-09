from __future__ import annotations

from edgespeech_rt.audio import si_sdr, synthetic_clean_noisy
from edgespeech_rt.model import build_model, count_parameters, estimate_macs_per_second


def test_si_sdr_detects_clean_reference_better_than_noisy():
    clean, noisy = synthetic_clean_noisy(seconds=0.5, snr_db=0.0)
    assert si_sdr(clean, clean) > si_sdr(clean, noisy) + 20.0


def test_model_size_stays_in_tiny_deployment_range():
    model = build_model(hidden_size=48, n_fft=512)
    params = count_parameters(model)
    macs = estimate_macs_per_second(model.config)
    assert params < 60_000
    assert macs < 4_000_000
