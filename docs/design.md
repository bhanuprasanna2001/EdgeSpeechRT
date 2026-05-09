# Architecture & Design

## The Core Problem

Noise reduction must happen frame-by-frame as audio arrives. You cannot buffer the whole recording, process it, and play it back — that adds unacceptable delay for live calls. This project is built around that constraint.

## Signal Processing Pipeline

Audio at 16 kHz is split into overlapping 32 ms windows (512 samples), advanced by 16 ms (256 samples) each step. This is the Short-Time Fourier Transform (STFT). Each window becomes 257 complex frequency bins.

```
Raw audio (16 kHz)
  └─ STFT (512-pt Hann window, 256-sample hop)
       └─ 257 complex frequency bins per frame
            ├─ log₁₊|magnitude|  →  Neural network input
            └─ Phase              →  Preserved, applied at output
```

**Why 256-sample hop?** A 256-sample hop is exactly 50% of the 512-sample window. This satisfies the NOLA (Nonzero Overlap-Add) condition, which guarantees that the inverse STFT perfectly reconstructs the original signal. The previous 320-sample (62.5%) hop did not guarantee this and required a custom workaround.

## Neural Network Contract

The model is exported to ONNX and lives entirely in the frequency domain. It never touches raw audio samples — the C++ SDK handles all STFT/iSTFT operations.

**Inputs:**
- `mag_log`: `float32 [batch, frames, 257]` — log₁₊|STFT magnitude|
- `h0`: `float32 [1, batch, 48]` — GRU recurrent state from the previous frame

**Outputs:**
- `mask`: `float32 [batch, frames, 257]` — values in [0, 1], one weight per frequency bin
- `hn`: `float32 [1, batch, 48]` — updated recurrent state passed to the next frame

**Applying the mask:**
```
enhanced_STFT = noisy_STFT × mask
```
Bins where mask ≈ 1: speech passes through unchanged.  
Bins where mask ≈ 0: noise is suppressed.

## Model Architecture

```
Input: log₁₊|STFT| — shape [B, T, 257]

Encoder
  Linear(257 → 48)  +  Tanh
  Linear(48  → 48)  +  Tanh

Recurrent (causal)
  GRU(48 → 48, batch_first=True)

Decoder
  Linear(48  → 48)  +  Tanh
  Linear(48  → 257) +  Sigmoid  →  mask ∈ [0, 1]
```

- **43,793 parameters** — fits in L1 cache of most modern CPUs
- **~2.69M multiply-accumulate operations per second** at 16 kHz
- **Causal** — the GRU only depends on past frames, never future ones

The model predicts a **real-valued** mask. Phase is preserved from the noisy input (not corrected). A future upgrade to complex ratio masking would close the remaining gap to GTCRN.

## Loss Function: HybridLoss

Training with a simple magnitude target tends to produce conservative models that barely suppress noise. This project uses the loss function from GTCRN (Xiaobin Rong et al., ICASSP 2024):

```
HybridLoss = 30 × (real_loss + imag_loss) + 70 × mag_loss + SI-SNR

where:
  real_loss  = MSE( pred_real / |pred|^0.7,  clean_real / |clean|^0.7 )
  imag_loss  = MSE( pred_imag / |pred|^0.7,  clean_imag / |clean|^0.7 )
  mag_loss   = MSE( |pred|^0.3,              |clean|^0.3              )
  SI-SNR     = time-domain scale-invariant SNR via inverse STFT
```

The power-law compressions (0.7 and 0.3) are perceptually motivated — they de-emphasise loud frequency bins where small errors matter less. The SI-SNR term directly optimises what the ear hears, not just what the spectrogram looks like.

**Result:** SI-SDR improved from +4.2 dB (old log-magnitude loss) to +8.7 dB (HybridLoss) on the same test set.

## C++ Streaming Contract

`SpeechEnhancer::process_frame` is the one function a C++ audio pipeline calls:

```
process_frame(256 samples of noisy audio)
  1. Append to 512-sample history buffer (sliding window)
  2. Apply 512-pt Hann window
  3. Forward STFT → 257 complex bins
  4. Compute log₁₊|magnitude|
  5. Run ONNX model → spectral mask
  6. Apply mask to complex STFT
  7. Inverse STFT + overlap-add
  8. Return 256 samples of enhanced audio
```

**Latency:** The model carries `512 − 256 = 256 samples` (16 ms) of algorithmic lookback. The WAV CLI compensates this fixed delay when writing to files so metrics stay sample-aligned.

## Training Setup

| Setting | Value | Reason |
|---|---|---|
| Dataset | VoiceBank-DEMAND 28-speaker set | Standard benchmark, 15 noise conditions |
| Training files | 10,415 paired recordings | Full dataset, no artificial cap |
| Validation files | 1,157 recordings | 10% held out, same seed every run |
| Segment length | 1.0 second (random crop) | Fits in memory, varied context |
| Batch size | 16 | Efficient on MPS |
| Optimizer | AdamW (lr=8e-4, decay=1e-4) | Standard for small recurrent models |
| LR schedule | CosineAnnealingLR → 1e-5 | Smooth decay prevents late-epoch divergence |
| Best checkpoint | Epoch 20 / 30 | Saved by lowest validation loss |
| Device | MPS (Apple Silicon GPU) | `--device auto` selects MPS → CUDA → CPU |

`num_workers=0` in DataLoader is intentional — MPS on macOS requires single-process data loading to avoid multiprocessing conflicts.
