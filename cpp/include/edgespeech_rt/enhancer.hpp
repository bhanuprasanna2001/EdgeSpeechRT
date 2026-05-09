#pragma once

#include <memory>
#include <string>
#include <vector>

namespace edgespeech_rt {

struct EnhancerConfig {
  int sample_rate = 16000;
  int frame_size = 512;
  int hop_size = 320;
  int hidden_size = 48;
  int intra_op_threads = 1;
  bool enable_onnx = true;
  std::string model_path;
};

class SpeechEnhancer {
 public:
  explicit SpeechEnhancer(const EnhancerConfig& config = {});
  ~SpeechEnhancer();

  SpeechEnhancer(SpeechEnhancer&&) noexcept;
  SpeechEnhancer& operator=(SpeechEnhancer&&) noexcept;
  SpeechEnhancer(const SpeechEnhancer&) = delete;
  SpeechEnhancer& operator=(const SpeechEnhancer&) = delete;

  void reset();
  [[nodiscard]] std::vector<float> process_frame(const std::vector<float>& input_20ms);
  [[nodiscard]] const EnhancerConfig& config() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace edgespeech_rt
