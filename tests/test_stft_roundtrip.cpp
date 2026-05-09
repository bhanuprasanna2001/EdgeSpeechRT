#include "edgespeech_rt/stft.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

int main() {
  edgespeech_rt::Stft stft({512, 320});
  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> dist(-0.8F, 0.8F);
  std::vector<float> input(16000);
  for (float& value : input) {
    value = dist(rng);
  }

  const auto reconstructed = stft.offline_roundtrip(input);
  double mse = 0.0;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const double error = static_cast<double>(input[i] - reconstructed[i]);
    mse += error * error;
  }
  const double rmse = std::sqrt(mse / static_cast<double>(input.size()));
  if (rmse > 1.0e-3) {
    std::cerr << "STFT roundtrip RMSE too high: " << rmse << "\n";
    return 1;
  }
  std::cout << "STFT roundtrip RMSE=" << rmse << "\n";
  return 0;
}
