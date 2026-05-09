#include "edgespeech_rt/enhancer.hpp"

#include "edgespeech_rt/stft.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#if EDGESPEECH_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace edgespeech_rt {

class SpeechEnhancer::Impl {
 public:
  explicit Impl(EnhancerConfig config)
      : config_(std::move(config)),
        stft_({config_.frame_size, config_.hop_size}),
        history_(static_cast<std::size_t>(config_.frame_size - config_.hop_size), 0.0F),
        ola_(static_cast<std::size_t>(config_.frame_size), 0.0F),
        norm_(static_cast<std::size_t>(config_.frame_size), 0.0F),
        hidden_(static_cast<std::size_t>(config_.hidden_size), 0.0F) {
    if (config_.sample_rate != 16000) {
      throw std::invalid_argument("EdgeSpeech-RT currently expects 16 kHz mono audio");
    }
    if (config_.frame_size <= 0 || config_.hop_size <= 0 || config_.hop_size > config_.frame_size) {
      throw std::invalid_argument("invalid enhancer frame/hop configuration");
    }
    init_onnx();
  }

  void reset() {
    std::fill(history_.begin(), history_.end(), 0.0F);
    std::fill(ola_.begin(), ola_.end(), 0.0F);
    std::fill(norm_.begin(), norm_.end(), 0.0F);
    std::fill(hidden_.begin(), hidden_.end(), 0.0F);
  }

  std::vector<float> process_frame(const std::vector<float>& input_20ms) {
    if (static_cast<int>(input_20ms.size()) != config_.hop_size) {
      throw std::invalid_argument("process_frame expects exactly hop_size samples");
    }

    std::vector<float> analysis(static_cast<std::size_t>(config_.frame_size), 0.0F);
    std::copy(history_.begin(), history_.end(), analysis.begin());
    std::copy(input_20ms.begin(), input_20ms.end(),
              analysis.begin() + static_cast<long>(history_.size()));

    auto spectrum = stft_.forward(analysis);
    const auto mask = infer_mask(spectrum);
    for (std::size_t i = 0; i < spectrum.size(); ++i) {
      spectrum[i] *= mask[i];
    }
    const auto synthesis = stft_.inverse(spectrum);
    const auto& window = stft_.window();
    for (int i = 0; i < config_.frame_size; ++i) {
      const auto index = static_cast<std::size_t>(i);
      ola_[index] += synthesis[index];
      norm_[index] += window[index] * window[index];
    }

    std::vector<float> output(static_cast<std::size_t>(config_.hop_size), 0.0F);
    for (int i = 0; i < config_.hop_size; ++i) {
      const auto index = static_cast<std::size_t>(i);
      output[index] = norm_[index] > 1.0e-8F ? ola_[index] / norm_[index] : 0.0F;
    }

    shift_left(ola_, config_.hop_size);
    shift_left(norm_, config_.hop_size);
    std::copy(analysis.end() - static_cast<long>(history_.size()), analysis.end(), history_.begin());
    return output;
  }

  const EnhancerConfig& config() const { return config_; }

 private:
  EnhancerConfig config_;
  Stft stft_;
  std::vector<float> history_;
  std::vector<float> ola_;
  std::vector<float> norm_;
  std::vector<float> hidden_;

#if EDGESPEECH_HAS_ONNXRUNTIME
  std::unique_ptr<Ort::Env> ort_env_;
  std::unique_ptr<Ort::Session> ort_session_;
  std::unique_ptr<Ort::MemoryInfo> ort_memory_;
#endif

  static void shift_left(std::vector<float>& values, int amount) {
    const auto shift = static_cast<std::size_t>(amount);
    if (shift >= values.size()) {
      std::fill(values.begin(), values.end(), 0.0F);
      return;
    }
    std::move(values.begin() + static_cast<long>(shift), values.end(), values.begin());
    std::fill(values.end() - static_cast<long>(shift), values.end(), 0.0F);
  }

  void init_onnx() {
#if EDGESPEECH_HAS_ONNXRUNTIME
    if (!config_.enable_onnx || config_.model_path.empty()) {
      return;
    }
    ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "edgespeech_rt");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(config_.intra_op_threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    ort_session_ = std::make_unique<Ort::Session>(*ort_env_, config_.model_path.c_str(), options);
    ort_memory_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
#else
    if (config_.enable_onnx && !config_.model_path.empty()) {
      throw std::runtime_error(
          "this build was compiled without ONNX Runtime; rebuild with EDGESPEECH_ENABLE_ORT=ON");
    }
#endif
  }

  std::vector<float> infer_mask(const Stft::Spectrum& spectrum) {
    std::vector<float> mask(spectrum.size(), 1.0F);
#if EDGESPEECH_HAS_ONNXRUNTIME
    if (!ort_session_) {
      return mask;
    }

    std::vector<float> mag_log(spectrum.size(), 0.0F);
    for (std::size_t i = 0; i < spectrum.size(); ++i) {
      mag_log[i] = std::log1p(std::abs(spectrum[i]));
    }

    std::array<int64_t, 3> mag_shape{1, 1, static_cast<int64_t>(spectrum.size())};
    std::array<int64_t, 3> hidden_shape{1, 1, static_cast<int64_t>(hidden_.size())};
    auto mag_tensor = Ort::Value::CreateTensor<float>(
        *ort_memory_, mag_log.data(), mag_log.size(), mag_shape.data(), mag_shape.size());
    auto h_tensor = Ort::Value::CreateTensor<float>(
        *ort_memory_, hidden_.data(), hidden_.size(), hidden_shape.data(), hidden_shape.size());

    const char* input_names[] = {"mag_log", "h0"};
    const char* output_names[] = {"mask", "hn"};
    std::array<Ort::Value, 2> inputs{std::move(mag_tensor), std::move(h_tensor)};
    auto outputs = ort_session_->Run(
        Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(), output_names, 2);

    const float* mask_data = outputs[0].GetTensorData<float>();
    std::copy(mask_data, mask_data + static_cast<long>(mask.size()), mask.begin());
    const float* hn_data = outputs[1].GetTensorData<float>();
    std::copy(hn_data, hn_data + static_cast<long>(hidden_.size()), hidden_.begin());
    for (float& value : mask) {
      value = std::clamp(value, 0.0F, 1.5F);
    }
#endif
    return mask;
  }
};

SpeechEnhancer::SpeechEnhancer(const EnhancerConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

SpeechEnhancer::~SpeechEnhancer() = default;
SpeechEnhancer::SpeechEnhancer(SpeechEnhancer&&) noexcept = default;
SpeechEnhancer& SpeechEnhancer::operator=(SpeechEnhancer&&) noexcept = default;

void SpeechEnhancer::reset() { impl_->reset(); }

std::vector<float> SpeechEnhancer::process_frame(const std::vector<float>& input_20ms) {
  return impl_->process_frame(input_20ms);
}

const EnhancerConfig& SpeechEnhancer::config() const { return impl_->config(); }

}  // namespace edgespeech_rt
