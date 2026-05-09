#!/usr/bin/env python
from __future__ import annotations

import argparse

from edgespeech_rt.config import load_yaml
from edgespeech_rt.export import export_onnx


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the streaming mask model to ONNX.")
    parser.add_argument("--config", default="configs/gtcrn_base.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output = args.output or cfg.get("export", {}).get("output", "artifacts/edgespeech_rt_fp32.onnx")
    path = export_onnx(
        output_path=output,
        checkpoint=args.checkpoint,
        hidden_size=int(cfg.get("hidden_size", 48)),
        n_fft=int(cfg.get("n_fft", 512)),
        opset=int(cfg.get("export", {}).get("opset", 17)),
        seed=int(cfg.get("training", {}).get("seed", 1337)),
    )
    print(f"exported {path}")


if __name__ == "__main__":
    main()
