# EdgeSpeech-RT

Quantized C++/ONNX Runtime SDK for real-time speech enhancement. This project turns a compact
PyTorch spectral-mask denoising prototype into a deployment-oriented library with real dataset
training, ONNX export, INT8 quantization, profiling, C++ inference, tests, metrics, plots, and
MATLAB validation helpers.

The current model is intentionally tiny: **43,793 parameters** and an estimated **2.16M MAC/s**
at 16 kHz with 20 ms hops. It is inspired by GTCRN/CRN-style causal recurrent masking, but it is
not the official GTCRN architecture. The goal here is a clean edge-deployment project, not an
offline SOTA denoiser.

## What Is Implemented

- VoiceBank-DEMAND download/extraction tooling.
- Real paired clean/noisy training on VoiceBank-DEMAND WAVs.
- FP32 ONNX export with PyTorch/ONNX Runtime equivalence tests.
- INT8 dynamic and INT8 static post-training quantization.
- Static INT8 calibration from real noisy training frames.
- Python ONNX Runtime latency profiling.
- C++ `libedgespeech_rt` with `SpeechEnhancer`, causal STFT/iSTFT, overlap-add, recurrent state, and WAV CLI.
- Optional C++ ONNX Runtime backend via `-DEDGESPEECH_ENABLE_ORT=ON`.
- PESQ, STOI, SI-SDR evaluation.
- Training curve, metric summary, and noisy/clean/enhanced/residual spectrogram plots.
- CTest and pytest regression tests.
- MATLAB STFT and comparison scripts.

## Fresh Measured Result

This table is a real local measurement from **May 9, 2026** after cleaning stale artifacts and
rerunning the experiment from scratch.

- Dataset: VoiceBank-DEMAND, 28-speaker train split and official test split.
- Training: first 4,096 paired train files, 12 epochs, 1-second random crops, batch size 8, CPU.
- Checkpoint selection: best validation loss, not last epoch.
- Evaluation: first 80 paired files from the official test split.
- Metrics: audio resampled to 16 kHz.
- Latency: ONNX Runtime Python CPU profiling, 1 thread, 500 measured recurrent spectral frames.

| Model | Params | Size | MAC/s | PESQ | STOI | SI-SDR | RTF | p95 latency | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Noisy | - | - | - | 2.1573 | 0.9181 | 8.4947 | - | - | 80-file held-out noisy baseline |
| FP32 | 43,793 | 173.4 KB | 2,155,200 | 2.5192 | 0.9206 | 12.6877 | 0.00092 | 0.0208 ms | fresh 4,096-file train subset |
| INT8_dynamic | 43,793 | 92.3 KB | 2,155,200 | 2.5001 | 0.9206 | 12.5377 | 0.00085 | 0.0193 ms | dynamic PTQ |
| INT8_static | 43,793 | 97.1 KB | 2,155,200 | 2.4766 | 0.9204 | 12.4410 | 0.00105 | 0.0218 ms | static PTQ, real calibration frames |

Generated reports:

- Summary CSV/Markdown: `benchmarks/summary.csv`, `benchmarks/summary.md`
- Latency report: `benchmarks/latency_report.md`
- Training curve: `assets/plots/training_curve.png`
- Metric summary: `assets/plots/metric_summary.png`
- Spectrogram/residual comparison: `assets/spectrograms/p232_080_fp32_comparison.png`
- Audio samples: `assets/audio_samples/p232_080_noisy.wav`, `assets/audio_samples/p232_080_enhanced_fp32.wav`, `assets/audio_samples/p232_080_clean.wav`

## Why The Earlier Plot Looked Wrong

The denoiser looked nearly identical to noisy because the earlier run had three problems:

- The loss over-emphasized matching a mask target, so the safest solution was a conservative
  near-identity mask that avoided speech distortion.
- The plotted example was weak, and separate color scales can hide residual changes.
- The C++ WAV CLI wrote the SDK's causal output without compensating its fixed 192-sample
  STFT lookback latency. That made unaligned quality checks look much worse than the actual signal.

The current training loss combines log-magnitude reconstruction, a blended Wiener/IRM mask target,
and a mild suppression regularizer. The spectrogram plot now uses shared color scales and residual
RMS labels, so the noise reduction is visible instead of implied.

## Quick Start

```bash
conda activate learning
python -m pip install -e .
python -m pytest
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Reproduce The Pipeline

Download VoiceBank-DEMAND:

```bash
python python/download_vctk_demand.py --mode all28
```

Train:

```bash
python python/train.py \
  --max-files 4096 \
  --epochs 12 \
  --batch-size 8 \
  --segment-seconds 1.0 \
  --lr 0.0008 \
  --output artifacts/edgespeech_rt.pt \
  --history-csv benchmarks/training_curve.csv
```

Export and quantize:

```bash
python python/export_onnx.py \
  --checkpoint artifacts/edgespeech_rt.pt \
  --output artifacts/edgespeech_rt_fp32.onnx

python python/quantize_onnx.py \
  --mode dynamic \
  --input artifacts/edgespeech_rt_fp32.onnx \
  --output artifacts/edgespeech_rt_int8_dynamic.onnx

python python/quantize_onnx.py \
  --mode static \
  --input artifacts/edgespeech_rt_fp32.onnx \
  --output artifacts/edgespeech_rt_int8_static.onnx \
  --calibration-noisy-dir datasets/vctk-demand/raw/noisy_trainset_28spk_wav \
  --calibration-max-files 32 \
  --frames 256
```

Profile, enhance, evaluate, summarize, and plot:

```bash
rm -f benchmarks/latency.csv
python python/profile_ort.py artifacts/edgespeech_rt_fp32.onnx --frames 500 --warmup 50 --csv benchmarks/latency.csv
python python/profile_ort.py artifacts/edgespeech_rt_int8_dynamic.onnx --frames 500 --warmup 50 --csv benchmarks/latency.csv
python python/profile_ort.py artifacts/edgespeech_rt_int8_static.onnx --frames 500 --warmup 50 --csv benchmarks/latency.csv

python python/enhance.py --onnx artifacts/edgespeech_rt_fp32.onnx --output-dir artifacts/enhanced_fp32 --max-files 80
python python/enhance.py --onnx artifacts/edgespeech_rt_int8_dynamic.onnx --output-dir artifacts/enhanced_int8_dynamic --max-files 80
python python/enhance.py --onnx artifacts/edgespeech_rt_int8_static.onnx --output-dir artifacts/enhanced_int8_static --max-files 80

python python/evaluate.py --clean-dir datasets/vctk-demand/raw/clean_testset_wav --enhanced-dir datasets/vctk-demand/raw/noisy_testset_wav --output benchmarks/metrics_noisy.csv --max-files 80
python python/evaluate.py --clean-dir datasets/vctk-demand/raw/clean_testset_wav --enhanced-dir artifacts/enhanced_fp32 --output benchmarks/metrics_fp32.csv --max-files 80
python python/evaluate.py --clean-dir datasets/vctk-demand/raw/clean_testset_wav --enhanced-dir artifacts/enhanced_int8_dynamic --output benchmarks/metrics_int8_dynamic.csv --max-files 80
python python/evaluate.py --clean-dir datasets/vctk-demand/raw/clean_testset_wav --enhanced-dir artifacts/enhanced_int8_static --output benchmarks/metrics_int8_static.csv --max-files 80

python python/report.py \
  --metric Noisy=benchmarks/metrics_noisy.csv \
  --metric FP32=benchmarks/metrics_fp32.csv \
  --metric INT8_dynamic=benchmarks/metrics_int8_dynamic.csv \
  --metric INT8_static=benchmarks/metrics_int8_static.csv \
  --model FP32=artifacts/edgespeech_rt_fp32.onnx \
  --model INT8_dynamic=artifacts/edgespeech_rt_int8_dynamic.onnx \
  --model INT8_static=artifacts/edgespeech_rt_int8_static.onnx

python python/plot.py training --csv benchmarks/training_curve.csv --output assets/plots/training_curve.png
python python/plot.py metrics --summary-csv benchmarks/summary.csv --output assets/plots/metric_summary.png
python python/plot.py spectrogram \
  --noisy datasets/vctk-demand/raw/noisy_testset_wav/p232_080.wav \
  --clean datasets/vctk-demand/raw/clean_testset_wav/p232_080.wav \
  --enhanced artifacts/enhanced_fp32/p232_080.wav \
  --output assets/spectrograms/p232_080_fp32_comparison.png
```

## C++ SDK

Default build, no ONNX Runtime dependency:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/edgespeech_wav_cli --input noisy_16k.wav --output enhanced.wav
```

ONNX Runtime C++ build:

```bash
cmake -S . -B build-ort \
  -DEDGESPEECH_ENABLE_ORT=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build-ort
ctest --test-dir build-ort --output-on-failure
./build-ort/edgespeech_wav_cli \
  --input noisy_16k.wav \
  --output enhanced.wav \
  --model artifacts/edgespeech_rt_fp32.onnx
```

The C++ CLI expects **16 kHz mono WAV** input. The SDK is causal and has a fixed
`frame_size - hop_size` lookback latency; the WAV CLI compensates that latency when writing files
so metric comparisons stay sample-aligned.

## Repository Layout

```text
configs/      model, pruning, PTQ, and QAT configs
python/       dataset, training, export, quantization, profiling, metrics, reports, plots
cpp/          C++ SDK headers/sources and WAV CLI
tests/        pytest and CTest regression tests
benchmarks/   measured CSV/Markdown reports
docs/         design, API, quantization, and debugging notes
matlab/       reference STFT and validation helpers
```

## Notes On Model Size

43.8K parameters is small, and that is intentional. The project is aligned with real-time speech
deployment work: deterministic inference, C++ SDK packaging, quantization, profiling, and
regression testing. A larger model could improve quality, but it should be justified with measured
RTF, latency, memory, and quality tradeoffs.

## Resume Framing

**EdgeSpeech-RT: Quantized C++ SDK for Real-Time Speech Enhancement**  
*C++ · Python · PyTorch · torchaudio · ONNX Runtime · CMake · STFT · INT8 Quantization · QAT*

- Built a modular real-time speech enhancement SDK converting a PyTorch causal recurrent spectral-mask prototype into a streaming C++/ONNX Runtime library with STFT/iSTFT processing, recurrent state caching, 20 ms frame inference, WAV CLI tooling, and Python/C++ validation hooks.
- Trained and benchmarked FP32, INT8 dynamic, and INT8 static variants on VoiceBank-DEMAND using PESQ, STOI, SI-SDR, model size, MAC/s, RTF, and p95 CPU frame latency.
- Added automated regression tests for STFT reconstruction, streaming state reset, latency budget checks, reproducible ONNX export, Python/ONNX numerical equivalence, and Python/C++ file-output equivalence.
