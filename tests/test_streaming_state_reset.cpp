#include "edgespeech_rt/enhancer.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
  edgespeech_rt::SpeechEnhancer enhancer;
  std::vector<float> frame(320);
  for (std::size_t i = 0; i < frame.size(); ++i) {
    frame[i] = 0.25F * std::sin(0.01F * static_cast<float>(i));
  }

  const auto first = enhancer.process_frame(frame);
  (void)enhancer.process_frame(frame);
  enhancer.reset();
  const auto after_reset = enhancer.process_frame(frame);

  if (first.size() != frame.size() || after_reset.size() != frame.size()) {
    std::cerr << "unexpected output frame size\n";
    return 1;
  }
  for (std::size_t i = 0; i < first.size(); ++i) {
    if (std::abs(first[i] - after_reset[i]) > 1.0e-6F) {
      std::cerr << "reset did not restore deterministic state at index " << i << "\n";
      return 1;
    }
  }
  std::cout << "streaming reset deterministic\n";
  return 0;
}
