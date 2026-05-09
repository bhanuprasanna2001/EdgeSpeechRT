from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from edgespeech_rt.model import (
    build_model,
    count_parameters,
    estimate_macs_per_second,
)


def export_onnx(
    output_path: str | Path,
    checkpoint: str | Path | None = None,
    hidden_size: int = 48,
    n_fft: int = 512,
    opset: int = 17,
    seed: int = 1337,
) -> Path:
    torch.manual_seed(seed)
    model = build_model(hidden_size=hidden_size, n_fft=n_fft)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state.get("model", state))
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mag_log = torch.randn(1, 3, model.config.num_bins)
    h0 = torch.zeros(1, 1, model.config.hidden_size)

    torch.onnx.export(
        model,
        (mag_log, h0),
        output_path,
        input_names=["mag_log", "h0"],
        output_names=["mask", "hn"],
        dynamic_axes={
            "mag_log": {0: "batch", 1: "frames"},
            "h0": {1: "batch"},
            "mask": {0: "batch", 1: "frames"},
            "hn": {1: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    metadata = {
        "hidden_size": hidden_size,
        "n_fft": n_fft,
        "num_bins": model.config.num_bins,
        "parameters": count_parameters(model),
        "estimated_macs_per_second": estimate_macs_per_second(model.config),
        "opset": opset,
        "sha256": sha256_file(output_path),
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
