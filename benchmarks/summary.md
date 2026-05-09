| Model | Params | Size | MAC/s | PESQ | STOI | SI-SDR | RTF | p95 latency | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Noisy | - | - | - | 2.1573 | 0.9181 | 8.4947 | - | - | 80-file held-out noisy baseline |
| FP32 | 43,793 | 173.4 KB | 2,155,200 | 2.5192 | 0.9206 | 12.6877 | 0.00092 | 0.0208 ms | fresh 4,096-file train subset |
| INT8_dynamic | 43,793 | 92.3 KB | 2,155,200 | 2.5001 | 0.9206 | 12.5377 | 0.00085 | 0.0193 ms | dynamic PTQ |
| INT8_static | 43,793 | 97.1 KB | 2,155,200 | 2.4766 | 0.9204 | 12.4410 | 0.00105 | 0.0218 ms | static PTQ, real calibration frames |
