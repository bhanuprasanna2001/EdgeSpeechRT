#include "edgespeech_rt/stft.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace edgespeech_rt {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

std::vector<float> hann_window(int size) {
  std::vector<float> window(static_cast<std::size_t>(size));
  if (size <= 1) {
    std::fill(window.begin(), window.end(), 1.0F);
    return window;
  }
  for (int i = 0; i < size; ++i) {
    window[static_cast<std::size_t>(i)] =
        static_cast<float>(0.5 - 0.5 * std::cos((2.0 * kPi * i) / (size - 1)));
  }
  return window;
}

}  // namespace

Stft::Stft(StftConfig config) : config_(config), window_(hann_window(config.frame_size)) {
  if (config_.frame_size <= 0 || config_.hop_size <= 0 || config_.hop_size > config_.frame_size) {
    throw std::invalid_argument("invalid STFT frame/hop size");
  }

  const int bins = num_bins();
  const int frame = frame_size();
  cos_forward_.resize(static_cast<std::size_t>(bins * frame));
  sin_forward_.resize(static_cast<std::size_t>(bins * frame));
  cos_inverse_.resize(static_cast<std::size_t>(frame * frame));
  sin_inverse_.resize(static_cast<std::size_t>(frame * frame));

  for (int k = 0; k < bins; ++k) {
    for (int n = 0; n < frame; ++n) {
      const double angle = (2.0 * kPi * k * n) / frame;
      cos_forward_[static_cast<std::size_t>(twiddle_index(k, n))] = static_cast<float>(std::cos(angle));
      sin_forward_[static_cast<std::size_t>(twiddle_index(k, n))] = static_cast<float>(std::sin(angle));
    }
  }

  for (int k = 0; k < frame; ++k) {
    for (int n = 0; n < frame; ++n) {
      const double angle = (2.0 * kPi * k * n) / frame;
      const auto index = static_cast<std::size_t>(k * frame + n);
      cos_inverse_[index] = static_cast<float>(std::cos(angle));
      sin_inverse_[index] = static_cast<float>(std::sin(angle));
    }
  }
}

int Stft::twiddle_index(int k, int n) const { return k * frame_size() + n; }

Stft::Spectrum Stft::forward(const std::vector<float>& frame) const {
  if (static_cast<int>(frame.size()) != frame_size()) {
    throw std::invalid_argument("STFT input frame has wrong size");
  }

  Spectrum spectrum(static_cast<std::size_t>(num_bins()));
  for (int k = 0; k < num_bins(); ++k) {
    double real = 0.0;
    double imag = 0.0;
    for (int n = 0; n < frame_size(); ++n) {
      const auto idx = static_cast<std::size_t>(twiddle_index(k, n));
      const double sample = static_cast<double>(frame[static_cast<std::size_t>(n)] *
                                                window_[static_cast<std::size_t>(n)]);
      real += sample * cos_forward_[idx];
      imag -= sample * sin_forward_[idx];
    }
    spectrum[static_cast<std::size_t>(k)] = {
        static_cast<float>(real), static_cast<float>(imag)};
  }
  return spectrum;
}

std::vector<float> Stft::inverse(const Spectrum& spectrum) const {
  if (static_cast<int>(spectrum.size()) != num_bins()) {
    throw std::invalid_argument("STFT spectrum has wrong bin count");
  }

  std::vector<std::complex<float>> full(static_cast<std::size_t>(frame_size()));
  for (int k = 0; k < num_bins(); ++k) {
    full[static_cast<std::size_t>(k)] = spectrum[static_cast<std::size_t>(k)];
  }
  for (int k = num_bins(); k < frame_size(); ++k) {
    full[static_cast<std::size_t>(k)] = std::conj(full[static_cast<std::size_t>(frame_size() - k)]);
  }

  std::vector<float> frame(static_cast<std::size_t>(frame_size()));
  for (int n = 0; n < frame_size(); ++n) {
    double sample = 0.0;
    for (int k = 0; k < frame_size(); ++k) {
      const auto twiddle = static_cast<std::size_t>(k * frame_size() + n);
      const auto value = full[static_cast<std::size_t>(k)];
      sample += static_cast<double>(value.real()) * cos_inverse_[twiddle] -
                static_cast<double>(value.imag()) * sin_inverse_[twiddle];
    }
    sample /= static_cast<double>(frame_size());
    frame[static_cast<std::size_t>(n)] =
        static_cast<float>(sample) * window_[static_cast<std::size_t>(n)];
  }
  return frame;
}

std::vector<float> Stft::offline_roundtrip(const std::vector<float>& input) const {
  const int frame = frame_size();
  const int hop = hop_size();
  std::vector<float> padded(static_cast<std::size_t>(frame), 0.0F);
  padded.insert(padded.end(), input.begin(), input.end());
  padded.resize(padded.size() + static_cast<std::size_t>(frame), 0.0F);

  while ((static_cast<int>(padded.size()) - frame) % hop != 0) {
    padded.push_back(0.0F);
  }

  std::vector<float> output(padded.size(), 0.0F);
  std::vector<float> norm(padded.size(), 0.0F);
  for (std::size_t pos = 0; pos + static_cast<std::size_t>(frame) <= padded.size();
       pos += static_cast<std::size_t>(hop)) {
    std::vector<float> chunk(padded.begin() + static_cast<long>(pos),
                             padded.begin() + static_cast<long>(pos + static_cast<std::size_t>(frame)));
    const auto spectrum = forward(chunk);
    const auto reconstructed = inverse(spectrum);
    for (int n = 0; n < frame; ++n) {
      const auto index = pos + static_cast<std::size_t>(n);
      output[index] += reconstructed[static_cast<std::size_t>(n)];
      const float w = window_[static_cast<std::size_t>(n)];
      norm[index] += w * w;
    }
  }

  std::vector<float> cropped(input.size(), 0.0F);
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto index = static_cast<std::size_t>(frame) + i;
    cropped[i] = norm[index] > 1.0e-8F ? output[index] / norm[index] : 0.0F;
  }
  return cropped;
}

}  // namespace edgespeech_rt
