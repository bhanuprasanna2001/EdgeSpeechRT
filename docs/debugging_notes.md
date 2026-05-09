# Debugging Guide

## Start Here

When the enhanced audio sounds wrong or metrics don't match expectations, work through this checklist before diving into model weights or training code.

---

## Python vs C++ Output Diverges

**Check 1 — Sample rate.** Both Python and C++ expect 16 kHz mono. The VoiceBank-DEMAND source files are 48 kHz. Python scripts resample automatically; the C++ CLI does not.

```bash
# Resample before passing to C++ CLI
ffmpeg -i noisy.wav -ar 16000 -ac 1 noisy_16k.wav
./build-ort/edgespeech_wav_cli --input noisy_16k.wav --output enhanced.wav --model artifacts/edgespeech_rt_fp32.onnx
```

**Check 2 — Hop size.** Both sides must use `hop_size = 256`. If Python uses 256 and C++ uses a different value, the number of frames and the mask alignment will differ.

**Check 3 — Window.** Both sides use a 512-sample Hann window (`periodic=False` in PyTorch, `torch.hann_window(512, periodic=False)`).

**Check 4 — Recurrent state.** Comparing isolated clips? Call `enhancer.reset()` in C++ and start with `h0 = zeros(...)` in Python before each clip.

---

## Enhanced Audio Sounds Like Noisy Audio

If the spectrogram of the enhanced output looks almost identical to the noisy input:

1. **Check the mask values.** Print `mask.min()` and `mask.max()` during inference. If both are near 1.0, the model never suppresses anything.
2. **Use shared color scales in spectrograms.** The `plot.py spectrogram` command already does this — residual plots with labelled RMS show suppression even when the raw spectrogram looks similar.
3. **Verify the checkpoint.** Load `artifacts/edgespeech_rt.pt` and check `metadata["best_val_epoch"]` and `metadata["best_val_loss"]`. A very high loss (>10) may indicate a failed training run.

---

## PESQ / STOI / SI-SDR Lower Than Expected

**Check 1 — Test set alignment.** `evaluate.py` matches files by filename. If the enhanced directory has different filenames than the clean directory, rows will be skipped silently.

**Check 2 — Length mismatch.** `evaluate.py` trims both signals to the shorter length before computing metrics. If enhanced files are much shorter than clean (e.g., missing the tail), scores will be biased.

**Check 3 — Resampling.** All metrics assume 16 kHz. `read_audio_16k` resamples automatically; if you use a different loader, verify sample rate.

**Check 4 — Latency compensation.** The C++ WAV CLI compensates the 256-sample lookback latency when writing files. Raw `process_frame` output is intentionally delayed by 256 samples — metrics on uncompensated C++ output will be poor due to misalignment.

---

## ONNX vs PyTorch Outputs Differ

Run the equivalence test:

```bash
python -m pytest tests/test_export_equivalence.py -v
```

If it fails:
1. Check ONNX opset. Export uses opset 17; older ORT versions may not support all ops.
2. GRU initial state — ensure `h0` is zeros and not carried over from a previous run.
3. Check `return_complex` — torch.stft in PyTorch 2.x always returns complex; old export paths may differ.

---

## INT8 Model Quality Significantly Worse Than FP32

Large quality gaps (>0.1 PESQ) after INT8 quantization usually mean the calibration data for static quantization was not representative.

- Use real noisy audio (not random noise) for calibration: `--calibration-noisy-dir datasets/vctk-demand/raw/noisy_trainset_28spk_wav`
- Use at least 32 files and 256 frames: `--calibration-max-files 32 --frames 256`
- If static PTQ degrades heavily, try dynamic PTQ — it quantizes activations at runtime and requires no calibration.

---

## Latency Is Higher Than Expected

- Fix `--threads 1` in `profile_ort.py` for reproducible results. Multiple threads can improve latency but also introduce variance.
- On macOS, ONNX Runtime uses CPUExecutionProvider. MPS/CoreML providers are possible but not configured here.
- INT8 static may be *slower* than FP32 on some CPU ORT paths due to QDQ dequantize overhead — always measure, never assume.
- The profiler generates a fresh random input each frame (no caching). Real audio has temporal correlation that may affect prefetch and branch prediction differently.

---

## Training Loss Does Not Decrease

- Learning rate too high: start with 8e-4 (AdamW default here) and reduce if loss oscillates.
- Gradient explosion: `clip_grad_norm_(model.parameters(), 5.0)` is already applied. If gradients are still large, check for NaN in inputs (corrupted audio files).
- MPS fallback: if `torch.istft` is not supported on MPS for the current PyTorch version, the SI-SNR term in HybridLoss falls back to zero. Training will still converge on the spectral terms alone — check the console for any `Exception` output from `_sisnr`.
