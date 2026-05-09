# Debugging Notes

Useful checks when Python and C++ outputs diverge:

- Compare `log1p(abs(STFT))` frame dumps before debugging model output.
- Verify that both sides use 16 kHz, 512-point Hann windows, and 320-sample hops.
- Resample VoiceBank-DEMAND raw WAVs before C++ CLI runs; the CLI rejects non-16 kHz input.
- Python uses explicit overlap-add reconstruction because `torch.istft` can reject some
  512/320 Hann-window lengths through its NOLA check.
- Reset recurrent state before comparing isolated clips.
- Keep ORT thread count fixed while comparing latency.
- Compare FP32 ONNX against PyTorch before investigating INT8 output.
- For WAV-file C++ metrics, account for the causal SDK's fixed `frame_size - hop_size` latency.
  The CLI compensates this delay; raw frame-by-frame API captures are intentionally delayed.
- If enhanced and noisy spectrograms look identical, verify shared color scales and residual plots.
  A mask can improve SI-SDR while the speech-dominant bins remain visually similar.

For quantized models, inspect whether the exported graph is QDQ or operator-oriented quantized
format. Backend operator support determines whether INT8 reduces latency or only model size.
