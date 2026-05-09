#pragma once

#include <string>
#include <vector>

namespace edgespeech_rt {

struct WavData {
  int sample_rate = 0;
  std::vector<float> samples;
};

[[nodiscard]] WavData read_wav_mono(const std::string& path);
void write_wav_mono_16bit(const std::string& path, int sample_rate, const std::vector<float>& samples);

}  // namespace edgespeech_rt
