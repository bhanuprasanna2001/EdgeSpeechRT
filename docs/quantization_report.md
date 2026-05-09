# Quantization Report

This project supports three deployment tracks:

| Track | Script | Status |
| --- | --- | --- |
| INT8 dynamic PTQ | `python/quantize_onnx.py --mode dynamic` | implemented |
| INT8 static PTQ | `python/quantize_onnx.py --mode static` | implemented with calibration reader |
| QAT-style fine-tuning | `python/qat_finetune.py` | implemented as fake-quant smoke fine-tune |

Important: latency improvements are not assumed. Small recurrent models may become smaller after
INT8 conversion without becoming faster on a given CPU backend. Publish only measured RTF and p95
latency from `python/profile_ort.py` and C++ CLI profiling runs.

## Measured Result

Measured on the fresh VoiceBank-DEMAND checkpoint from May 9, 2026:

| Variant | Size | PESQ | STOI | SI-SDR | RTF | p95 latency | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 ONNX | 173.4 KB | 2.5192 | 0.9206 | 12.6877 | 0.00092 | 0.0208 ms | reference |
| INT8 dynamic | 92.3 KB | 2.5001 | 0.9206 | 12.5377 | 0.00085 | 0.0193 ms | smaller and slightly faster in this local ORT profile |
| INT8 static | 97.1 KB | 2.4766 | 0.9204 | 12.4410 | 0.00105 | 0.0218 ms | calibrated on real noisy train frames |

Static INT8 did not improve latency in this measurement, despite reducing size. That is expected
for small recurrent models on some CPU operator paths and is why the project reports measured
latency rather than assuming INT8 is faster.

Recommended measurement order:

1. Train or choose a checkpoint.
2. Export FP32 ONNX with `python/export_onnx.py`.
3. Quantize dynamic and static variants with `python/quantize_onnx.py`.
4. Use real noisy WAV calibration frames for static quantization.
5. Profile each variant with identical thread count and frame count.
6. Run metric regression on the same clean/noisy/enhanced file list.
