# Latency Report

ONNX Runtime CPU profiling uses one recurrent spectral frame per call after warmup.

| Model | Threads | Frames | Mean ms | p50 ms | p95 ms | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| edgespeech_rt_fp32.onnx | 1 | 500 | 0.0184 | 0.0175 | 0.0208 | 0.00092 |
| edgespeech_rt_int8_dynamic.onnx | 1 | 500 | 0.0170 | 0.0166 | 0.0193 | 0.00085 |
| edgespeech_rt_int8_static.onnx | 1 | 500 | 0.0209 | 0.0205 | 0.0218 | 0.00105 |
