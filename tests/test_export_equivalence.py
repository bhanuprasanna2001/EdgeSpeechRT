from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch

from edgespeech_rt.export import export_onnx
from edgespeech_rt.model import build_model


def test_onnx_export_matches_torch(tmp_path):
    torch.manual_seed(1337)
    model = build_model(hidden_size=48, n_fft=512).eval()
    onnx_path = tmp_path / "masker.onnx"
    torch.save({"model": model.state_dict()}, tmp_path / "model.pt")
    export_onnx(onnx_path, checkpoint=tmp_path / "model.pt", hidden_size=48, n_fft=512)

    mag = torch.rand(1, 4, 257)
    h0 = torch.zeros(1, 1, 48)
    with torch.no_grad():
        expected_mask, expected_hn = model(mag, h0)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual_mask, actual_hn = session.run(
        ["mask", "hn"],
        {"mag_log": mag.numpy().astype(np.float32), "h0": h0.numpy().astype(np.float32)},
    )

    np.testing.assert_allclose(actual_mask, expected_mask.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(actual_hn, expected_hn.numpy(), rtol=1e-4, atol=1e-4)
