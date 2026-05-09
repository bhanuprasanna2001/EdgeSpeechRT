# Design

EdgeSpeech-RT separates signal processing from neural inference. The model is a compact
GTCRN/CRN-inspired recurrent spectral masker, not the official GTCRN implementation.

## Streaming Contract

- Input: 16 kHz mono audio.
- Frame hop: 320 samples, equal to 20 ms.
- Analysis window: 512-sample Hann window.
- C++ owns frame history, causal STFT, inverse STFT, overlap-add normalization, and recurrent state.
- The ONNX model consumes one or more `log1p(abs(STFT))` frames and returns a spectral mask plus GRU state.
- Python enhancement uses explicit overlap-add normalization instead of relying on `torch.istft`,
  so 512/320 deployment hops are handled consistently across Python and C++.
- The SDK stream has `frame_size - hop_size` samples of algorithmic lookback latency. The WAV CLI
  compensates this fixed 192-sample delay when writing files for sample-aligned metric checks.

## Model Contract

Inputs:

- `mag_log`: float32 `[batch, frames, 257]`
- `h0`: float32 `[1, batch, hidden_size]`

Outputs:

- `mask`: float32 `[batch, frames, 257]`
- `hn`: float32 `[1, batch, hidden_size]`

This keeps the exported graph small and makes Python/C++ debugging easier because spectral
features and masks can be dumped independently.

## C++ Runtime

`SpeechEnhancer::process_frame` accepts exactly one 20 ms frame. The no-model build uses an
identity mask, which is useful for validating STFT/iSTFT and buffering without requiring ONNX
Runtime C++ headers. When built with `EDGESPEECH_ENABLE_ORT=ON`, the same class loads the ONNX
model and caches recurrent state across calls.

The WAV CLI is intentionally strict: input must be 16 kHz mono. Dataset scripts resample raw
benchmark audio to 16 kHz before metric computation or C++ CLI runs.

## Training Objective

The first mask-only objective produced conservative near-identity masks. The current objective uses:

- log-magnitude reconstruction between enhanced and clean spectra
- a blended Wiener/IRM target mask
- a mild suppression penalty in noise-dominant bins

This keeps speech distortion low while forcing measurable attenuation of residual noise.

## Training/Evaluation Split

The measured local run uses:

- VoiceBank-DEMAND 28-speaker train split: first 4,096 paired files for training/validation.
- Official VoiceBank-DEMAND test split: first 80 paired files for reported subset metrics.
- 1-second random crops during training.
- 16 kHz resampling for all metrics.
