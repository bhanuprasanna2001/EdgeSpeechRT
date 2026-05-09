from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from edgespeech_rt.audio import synthetic_clean_noisy


def test_cpp_cli_matches_python_streaming_identity(tmp_path):
    cli = Path("build/edgespeech_wav_cli")
    if not cli.exists():
        pytest.skip("C++ CLI is not built yet")

    _, noisy = synthetic_clean_noisy(seconds=0.20, seed=7)
    input_wav = tmp_path / "input.wav"
    output_wav = tmp_path / "output.wav"
    sf.write(input_wav, noisy, 16000)

    subprocess.run(
        [str(cli), "--input", str(input_wav), "--output", str(output_wav)],
        check=True,
    )
    quantized_input, _ = sf.read(input_wav, dtype="float32")
    actual, _ = sf.read(output_wav, dtype="float32")
    expected = np.asarray(quantized_input, dtype=np.float32)

    np.testing.assert_allclose(actual[: expected.size], expected, atol=2e-4, rtol=2e-4)
