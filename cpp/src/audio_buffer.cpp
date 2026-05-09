#include "edgespeech_rt/audio_buffer.hpp"

#include <algorithm>
#include <stdexcept>

namespace edgespeech_rt {

void FrameBuffer::append(const std::vector<float>& samples) {
  samples_.insert(samples_.end(), samples.begin(), samples.end());
}

bool FrameBuffer::can_pop(std::size_t frame_size) const { return samples_.size() >= frame_size; }

std::vector<float> FrameBuffer::pop(std::size_t frame_size) {
  if (!can_pop(frame_size)) {
    throw std::runtime_error("not enough samples in FrameBuffer");
  }
  std::vector<float> frame(samples_.begin(), samples_.begin() + static_cast<long>(frame_size));
  samples_.erase(samples_.begin(), samples_.begin() + static_cast<long>(frame_size));
  return frame;
}

void FrameBuffer::clear() { samples_.clear(); }

std::vector<std::vector<float>> split_frames(
    const std::vector<float>& samples, std::size_t frame_size, bool pad_last) {
  std::vector<std::vector<float>> frames;
  for (std::size_t pos = 0; pos < samples.size(); pos += frame_size) {
    const std::size_t end = std::min(pos + frame_size, samples.size());
    if (end - pos < frame_size && !pad_last) {
      break;
    }
    std::vector<float> frame(samples.begin() + static_cast<long>(pos),
                             samples.begin() + static_cast<long>(end));
    frame.resize(frame_size, 0.0F);
    frames.push_back(std::move(frame));
  }
  return frames;
}

}  // namespace edgespeech_rt
