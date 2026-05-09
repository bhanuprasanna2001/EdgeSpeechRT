#include "edgespeech_rt/enhancer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
  edgespeech_rt::SpeechEnhancer enhancer;
  std::vector<float> frame(320);
  std::vector<double> latencies;
  latencies.reserve(80);

  for (int index = 0; index < 100; ++index) {
    for (std::size_t i = 0; i < frame.size(); ++i) {
      frame[i] = 0.2F * std::sin(0.02F * static_cast<float>(i + index));
    }
    const auto start = std::chrono::steady_clock::now();
    const auto output = enhancer.process_frame(frame);
    const auto end = std::chrono::steady_clock::now();
    if (output.size() != frame.size()) {
      std::cerr << "unexpected output size\n";
      return 1;
    }
    if (index >= 20) {
      latencies.push_back(
          std::chrono::duration<double, std::milli>(end - start).count());
    }
  }

  std::sort(latencies.begin(), latencies.end());
  const auto p95_index = static_cast<std::size_t>(0.95 * static_cast<double>(latencies.size() - 1));
  const double p95 = latencies[p95_index];
  std::cout << "p95 frame latency=" << p95 << " ms\n";
  if (p95 > 20.0) {
    std::cerr << "p95 latency exceeds 20 ms frame budget\n";
    return 1;
  }
  return 0;
}
