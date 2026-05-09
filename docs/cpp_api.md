# C++ API

## Overview

`libedgespeech_rt` is a causal streaming speech enhancement library. It processes audio one 16 ms frame at a time and maintains state between calls — there is no buffering of the full audio, making it suitable for live microphone streams.

## Quick Example

```cpp
#include "edgespeech_rt/enhancer.hpp"

// Configure
edgespeech_rt::EnhancerConfig config;
config.sample_rate  = 16000;
config.frame_size   = 512;   // FFT window, samples
config.hop_size     = 256;   // Frame advance, samples = 16 ms at 16 kHz
config.model_path   = "artifacts/edgespeech_rt_fp32.onnx";  // or int8 variant

// Create (loads ONNX model once)
edgespeech_rt::SpeechEnhancer enhancer(config);

// Process live audio — call once per 16 ms chunk
while (audio_available) {
    std::vector<float> noisy_chunk = get_next_16ms_of_audio();  // exactly 256 samples
    std::vector<float> clean_chunk = enhancer.process_frame(noisy_chunk);
    send_to_speaker(clean_chunk);
}

// Reset between independent streams (clears GRU state and overlap buffers)
enhancer.reset();
```

## Build Options

### Default build (no ONNX Runtime — identity passthrough)

Useful for validating the STFT pipeline, buffering, and timing without pulling in ONNX Runtime.

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure

# WAV CLI (identity mask — cleans nothing, but validates I/O)
./build/edgespeech_wav_cli --input noisy_16k.wav --output passthrough.wav
```

### Full build with ONNX Runtime (neural inference)

```bash
cmake -S . -B build-ort \
  -DEDGESPEECH_ENABLE_ORT=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build-ort
ctest --test-dir build-ort --output-on-failure

# WAV CLI with neural model
./build-ort/edgespeech_wav_cli \
  --input  noisy_16k.wav \
  --output enhanced.wav \
  --model  artifacts/edgespeech_rt_fp32.onnx
```

Input must be **16 kHz mono WAV**. The CLI resamples nothing — run `python python/evaluate.py` to handle resampling automatically.

## Latency

The SDK has a fixed **256-sample (16 ms) algorithmic lookback latency** — the GRU needs to see the current frame, which requires the full 512-sample analysis window, so 256 past samples must be buffered. This is inherent to the causal STFT design and cannot be reduced without shrinking the FFT window.

The WAV CLI compensates this delay when writing output files so that sample-aligned metric comparisons (PESQ, STOI, SI-SDR) are valid.

## Choosing a Model Variant

| Variant | File | Size | PESQ | Notes |
|---|---|---:|---:|---|
| FP32 | `edgespeech_rt_fp32.onnx` | 173 KB | 2.48 | Reference, highest quality |
| INT8 dynamic | `edgespeech_rt_int8_dynamic.onnx` | 92 KB | 2.45 | 47% smaller, similar quality |
| INT8 static | `edgespeech_rt_int8_static.onnx` | 97 KB | 2.39 | Calibrated, may be faster on some targets |

For memory-constrained deployment, `INT8 dynamic` is the best trade-off: nearly half the size with only 0.03 PESQ degradation.
