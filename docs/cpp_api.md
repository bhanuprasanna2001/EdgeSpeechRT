# C++ API

```cpp
#include "edgespeech_rt/enhancer.hpp"

edgespeech_rt::EnhancerConfig config;
config.sample_rate = 16000;
config.hop_size = 320;
config.frame_size = 512;
config.model_path = "artifacts/edgespeech_rt_fp32.onnx";

edgespeech_rt::SpeechEnhancer enhancer(config);
std::vector<float> enhanced_20ms = enhancer.process_frame(input_20ms);
```

`process_frame` requires exactly `hop_size` samples and returns exactly `hop_size` samples.
Call `reset()` between independent streams to clear overlap-add buffers and recurrent state.
The streaming API is causal and carries `frame_size - hop_size` samples of lookback latency. The
WAV CLI writes sample-aligned files by flushing one extra zero frame and trimming that fixed delay.

The default build compiles without ONNX Runtime and uses an identity mask. Build with
`-DEDGESPEECH_ENABLE_ORT=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime` to enable neural inference.
