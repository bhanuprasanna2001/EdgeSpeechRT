#include "edgespeech_rt/wav_io.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace edgespeech_rt {
namespace {

uint16_t read_u16(std::istream& input) {
  uint8_t bytes[2];
  input.read(reinterpret_cast<char*>(bytes), 2);
  return static_cast<uint16_t>(bytes[0] | (bytes[1] << 8));
}

uint32_t read_u32(std::istream& input) {
  uint8_t bytes[4];
  input.read(reinterpret_cast<char*>(bytes), 4);
  return static_cast<uint32_t>(bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24));
}

void write_u16(std::ostream& output, uint16_t value) {
  const uint8_t bytes[2] = {static_cast<uint8_t>(value & 0xFF),
                            static_cast<uint8_t>((value >> 8) & 0xFF)};
  output.write(reinterpret_cast<const char*>(bytes), 2);
}

void write_u32(std::ostream& output, uint32_t value) {
  const uint8_t bytes[4] = {static_cast<uint8_t>(value & 0xFF),
                            static_cast<uint8_t>((value >> 8) & 0xFF),
                            static_cast<uint8_t>((value >> 16) & 0xFF),
                            static_cast<uint8_t>((value >> 24) & 0xFF)};
  output.write(reinterpret_cast<const char*>(bytes), 4);
}

std::string read_tag(std::istream& input) {
  char tag[4];
  input.read(tag, 4);
  return std::string(tag, 4);
}

}  // namespace

WavData read_wav_mono(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open WAV: " + path);
  }
  if (read_tag(input) != "RIFF") {
    throw std::runtime_error("not a RIFF file: " + path);
  }
  (void)read_u32(input);
  if (read_tag(input) != "WAVE") {
    throw std::runtime_error("not a WAVE file: " + path);
  }

  uint16_t audio_format = 0;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  std::vector<uint8_t> data;

  while (input && (!sample_rate || data.empty())) {
    const std::string chunk = read_tag(input);
    const uint32_t chunk_size = read_u32(input);
    if (chunk == "fmt ") {
      audio_format = read_u16(input);
      channels = read_u16(input);
      sample_rate = read_u32(input);
      (void)read_u32(input);
      (void)read_u16(input);
      bits_per_sample = read_u16(input);
      if (chunk_size > 16) {
        input.seekg(static_cast<std::streamoff>(chunk_size - 16), std::ios::cur);
      }
    } else if (chunk == "data") {
      data.resize(chunk_size);
      input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    } else {
      input.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
    }
    if (chunk_size % 2 == 1) {
      input.seekg(1, std::ios::cur);
    }
  }

  if (audio_format != 1 || bits_per_sample != 16 || channels == 0) {
    throw std::runtime_error("only PCM 16-bit WAV files are supported");
  }

  const std::size_t total_samples = data.size() / sizeof(int16_t);
  const std::size_t frames = total_samples / channels;
  std::vector<float> mono(frames, 0.0F);
  const auto* pcm = reinterpret_cast<const int16_t*>(data.data());
  for (std::size_t frame = 0; frame < frames; ++frame) {
    float sum = 0.0F;
    for (std::size_t ch = 0; ch < channels; ++ch) {
      sum += static_cast<float>(pcm[frame * channels + ch]) / 32768.0F;
    }
    mono[frame] = sum / static_cast<float>(channels);
  }
  return {static_cast<int>(sample_rate), std::move(mono)};
}

void write_wav_mono_16bit(const std::string& path, int sample_rate, const std::vector<float>& samples) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to write WAV: " + path);
  }
  const uint16_t channels = 1;
  const uint16_t bits = 16;
  const uint32_t byte_rate = static_cast<uint32_t>(sample_rate * channels * bits / 8);
  const uint16_t block_align = static_cast<uint16_t>(channels * bits / 8);
  const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));

  output.write("RIFF", 4);
  write_u32(output, 36 + data_size);
  output.write("WAVE", 4);
  output.write("fmt ", 4);
  write_u32(output, 16);
  write_u16(output, 1);
  write_u16(output, channels);
  write_u32(output, static_cast<uint32_t>(sample_rate));
  write_u32(output, byte_rate);
  write_u16(output, block_align);
  write_u16(output, bits);
  output.write("data", 4);
  write_u32(output, data_size);
  for (float sample : samples) {
    const float clipped = std::clamp(sample, -1.0F, 1.0F);
    const auto value = static_cast<int16_t>(clipped * 32767.0F);
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
  }
}

}  // namespace edgespeech_rt
