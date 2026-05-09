#pragma once

#include <complex>
#include <vector>

namespace edgespeech_rt {

struct StftConfig {
  int frame_size = 512;
  int hop_size = 320;
};

class Stft {
 public:
  using Spectrum = std::vector<std::complex<float>>;

  explicit Stft(StftConfig config = {});

  [[nodiscard]] int frame_size() const { return config_.frame_size; }
  [[nodiscard]] int hop_size() const { return config_.hop_size; }
  [[nodiscard]] int num_bins() const { return config_.frame_size / 2 + 1; }
  [[nodiscard]] const std::vector<float>& window() const { return window_; }

  [[nodiscard]] Spectrum forward(const std::vector<float>& frame) const;
  [[nodiscard]] std::vector<float> inverse(const Spectrum& spectrum) const;
  [[nodiscard]] std::vector<float> offline_roundtrip(const std::vector<float>& input) const;

 private:
  StftConfig config_;
  std::vector<float> window_;
  std::vector<float> cos_forward_;
  std::vector<float> sin_forward_;
  std::vector<float> cos_inverse_;
  std::vector<float> sin_inverse_;

  [[nodiscard]] int twiddle_index(int k, int n) const;
};

}  // namespace edgespeech_rt
