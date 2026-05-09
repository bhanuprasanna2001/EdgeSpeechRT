# Latency Report

ONNX Runtime CPU profiling uses one recurrent spectral frame per call after warmup.

| Model | Threads | Frames | Mean ms | p50 ms | p95 ms | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| edgespeech_rt_fp32.onnx | 1 | 500 | 0.0178 | 0.0169 | 0.0202 | 0.00111 |
| edgespeech_rt_int8_dynamic.onnx | 1 | 500 | 0.0176 | 0.0165 | 0.0197 | 0.00110 |
| edgespeech_rt_int8_static.onnx | 1 | 500 | 0.0213 | 0.0200 | 0.0256 | 0.00133 |
