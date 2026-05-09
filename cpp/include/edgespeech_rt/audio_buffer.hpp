#pragma once

#include <cstddef>
#include <vector>

namespace edgespeech_rt {

class FrameBuffer {
 public:
  void append(const std::vector<float>& samples);
  [[nodiscard]] bool can_pop(std::size_t frame_size) const;
  [[nodiscard]] std::vector<float> pop(std::size_t frame_size);
  [[nodiscard]] std::size_t size() const { return samples_.size(); }
  void clear();

 private:
  std::vector<float> samples_;
};

[[nodiscard]] std::vector<std::vector<float>> split_frames(
    const std::vector<float>& samples, std::size_t frame_size, bool pad_last = true);

}  // namespace edgespeech_rt
