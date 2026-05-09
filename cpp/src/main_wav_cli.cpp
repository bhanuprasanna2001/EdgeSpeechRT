#include "edgespeech_rt/audio_buffer.hpp"
#include "edgespeech_rt/enhancer.hpp"
#include "edgespeech_rt/wav_io.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --input noisy.wav --output enhanced.wav [--model model.onnx] [--threads 1]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string input_path;
    std::string output_path;
    std::string model_path;
    int threads = 1;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--input" && i + 1 < argc) {
        input_path = argv[++i];
      } else if (arg == "--output" && i + 1 < argc) {
        output_path = argv[++i];
      } else if (arg == "--model" && i + 1 < argc) {
        model_path = argv[++i];
      } else if (arg == "--threads" && i + 1 < argc) {
        threads = std::stoi(argv[++i]);
      } else if (arg == "--help" || arg == "-h") {
        usage(argv[0]);
        return 0;
      } else {
        usage(argv[0]);
        return 2;
      }
    }

    if (input_path.empty() || output_path.empty()) {
      usage(argv[0]);
      return 2;
    }

    const auto wav = edgespeech_rt::read_wav_mono(input_path);
    edgespeech_rt::EnhancerConfig config;
    config.sample_rate = wav.sample_rate;
    config.model_path = model_path;
    config.enable_onnx = !model_path.empty();
    config.intra_op_threads = threads;
    edgespeech_rt::SpeechEnhancer enhancer(config);

    std::vector<float> enhanced;
    enhanced.reserve(wav.samples.size() + static_cast<std::size_t>(config.hop_size));
    for (const auto& frame : edgespeech_rt::split_frames(
             wav.samples, static_cast<std::size_t>(config.hop_size), true)) {
      const auto output = enhancer.process_frame(frame);
      enhanced.insert(enhanced.end(), output.begin(), output.end());
    }
    const std::vector<float> flush(static_cast<std::size_t>(config.hop_size), 0.0F);
    const auto tail = enhancer.process_frame(flush);
    enhanced.insert(enhanced.end(), tail.begin(), tail.end());

    const auto latency = static_cast<std::size_t>(config.frame_size - config.hop_size);
    std::vector<float> aligned(wav.samples.size(), 0.0F);
    if (enhanced.size() > latency) {
      const auto available = std::min(aligned.size(), enhanced.size() - latency);
      std::copy(enhanced.begin() + static_cast<long>(latency),
                enhanced.begin() + static_cast<long>(latency + available),
                aligned.begin());
    }
    edgespeech_rt::write_wav_mono_16bit(output_path, wav.sample_rate, aligned);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "edgespeech_wav_cli: " << error.what() << "\n";
    return 1;
  }
}
