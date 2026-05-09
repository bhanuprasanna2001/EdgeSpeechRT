| Model | Params | Size | MAC/s | PESQ | STOI | SI-SDR | RTF | p95 latency | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Noisy | - | - | - | 1.9668 | 0.9211 | 8.4494 | - | - | 824 files, held-out test set |
| FP32 | 43,793 | 173.4 KB | 2,694,000 | 2.4801 | 0.9279 | 17.1538 | 0.00111 | 0.0202 ms | FP32 ONNX, full retrain |
| INT8_dynamic | 43,793 | 92.3 KB | 2,694,000 | 2.4542 | 0.9281 | 17.1648 | 0.00110 | 0.0197 ms | dynamic PTQ |
| INT8_static | 43,793 | 97.1 KB | 2,694,000 | 2.3911 | 0.9284 | 17.1244 | 0.00133 | 0.0256 ms | static PTQ, real calibration |
