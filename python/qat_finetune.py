#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from edgespeech_rt.config import load_yaml
from edgespeech_rt.model import FakeQuantizedMasker, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="QAT-style fine-tuning smoke run with fake quantization.")
    parser.add_argument("--config", default="configs/qat_int8.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base = build_model(hidden_size=int(cfg.get("hidden_size", 48)), n_fft=int(cfg.get("n_fft", 512)))
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        base.load_state_dict(state.get("model", state))
    model = FakeQuantizedMasker(base)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("qat", {}).get("learning_rate", 3e-4)))

    batches = int(cfg.get("qat", {}).get("synthetic_batches", 64))
    epochs = int(cfg.get("qat", {}).get("epochs", 3))
    bins = base.config.num_bins
    h0 = torch.zeros(1, 4, base.config.hidden_size)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for _ in range(batches):
            mag = torch.rand(4, 8, bins)
            target = torch.clamp(1.0 / (1.0 + 0.35 * mag), 0.0, 1.0)
            optimizer.zero_grad(set_to_none=True)
            mask, _ = model(mag, h0)
            loss = F.mse_loss(mask, target)
            loss.backward()
            optimizer.step()
            total += float(loss)
        print(f"epoch={epoch} loss={total / batches:.6f}")

    output = Path(args.output or cfg.get("checkpoint", "artifacts/edgespeech_rt_qat.pt"))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": base.state_dict(), "config": cfg, "qat_fake_quant": True}, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
