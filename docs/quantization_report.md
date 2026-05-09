# Quantization Guide

## Why Quantize?

The FP32 (32-bit float) model is 173 KB and runs in ~0.018 ms per frame. INT8 (8-bit integer) quantization reduces the model to **92–97 KB** with nearly identical quality. On edge hardware with hardware INT8 acceleration, inference can also become faster.

**Measured trade-off (824-file VCTK test set):**

| Variant | Size | PESQ | SI-SDR | p95 latency | vs FP32 |
|---|---:|---:|---:|---:|---|
| FP32 ONNX | 173 KB | 2.480 | 17.15 dB | 0.020 ms | reference |
| INT8 dynamic | **92 KB** | 2.454 | 17.16 dB | 0.020 ms | −47% size, −0.03 PESQ |
| INT8 static | 97 KB | 2.391 | 17.12 dB | 0.026 ms | −44% size, −0.09 PESQ |

For this CPU backend (macOS, ONNX Runtime), INT8 static did not reduce latency — it is slower because the QDQ operator graph has extra dequantize nodes that some ORT kernel paths do not fuse. Always measure before assuming INT8 = faster.

## Three Quantization Methods

### 1 · FP32 ONNX (baseline)

Export the trained PyTorch model to ONNX format with full 32-bit precision:

```bash
python python/export_onnx.py \
  --checkpoint artifacts/edgespeech_rt.pt \
  --output     artifacts/edgespeech_rt_fp32.onnx
```

### 2 · INT8 Dynamic PTQ

Quantizes **weights** to INT8 at export time. Activations are quantized to INT8 **at runtime** per-tensor. No calibration data needed.

```bash
python python/quantize_onnx.py \
  --mode   dynamic \
  --input  artifacts/edgespeech_rt_fp32.onnx \
  --output artifacts/edgespeech_rt_int8_dynamic.onnx
```

**Best for:** fast iteration, memory-constrained deployment, cases where you have no representative data. Nearly identical quality to FP32 in this model.

### 3 · INT8 Static PTQ

Quantizes both weights and activations to INT8. Requires running real audio through the model first (calibration) to measure the actual range of activation values, then bakes those ranges into the graph.

```bash
python python/quantize_onnx.py \
  --mode dynamic \
  --input  artifacts/edgespeech_rt_fp32.onnx \
  --output artifacts/edgespeech_rt_int8_static.onnx \
  --calibration-noisy-dir datasets/vctk-demand/raw/noisy_trainset_28spk_wav \
  --calibration-max-files 32 \
  --frames 256
```

**Best for:** hardware targets with dedicated INT8 accelerators (DSPs, NPUs). May not help on general-purpose CPUs.

## Profiling Latency

Always measure with fixed thread count and the actual target hardware:

```bash
# Profile each variant (500 frames, 50 warmup, 1 thread)
python python/profile_ort.py artifacts/edgespeech_rt_fp32.onnx       --frames 500 --warmup 50 --csv benchmarks/latency.csv
python python/profile_ort.py artifacts/edgespeech_rt_int8_dynamic.onnx --frames 500 --warmup 50 --csv benchmarks/latency.csv
python python/profile_ort.py artifacts/edgespeech_rt_int8_static.onnx  --frames 500 --warmup 50 --csv benchmarks/latency.csv
```

Output columns: `mean_ms`, `p50_ms`, `p95_ms`, `rtf` (real-time factor = latency / frame_duration).

**RTF < 1.0** means the model is faster than real time. At RTF 0.0011, it is **900× faster than real time** — the CPU is idle 99.9% of the time while processing audio.

## QAT (Quantization-Aware Training)

`python/qat_finetune.py` implements a fake-quantize smoke fine-tune using `FakeQuantizedMasker` — a thin wrapper that inserts fake quantize nodes around inputs and outputs. This is a starting point for hardware-specific INT8 fine-tuning rather than a complete QAT pipeline. Results are not benchmarked here.
