#include "models/gemma4/31b/sm120/mtp/head.h"
#include "models/gemma4/31b/sm120/mtp/cuda.cuh"
#include "gewell/models/gemma4/31b/model.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

template<class T> std::vector<T> read(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() <= 0 || std::size_t(f.tellg()) % sizeof(T)) throw std::runtime_error("invalid fixture " + path);
  std::vector<T> result(std::size_t(f.tellg()) / sizeof(T)); f.seekg(0);
  if (!f.read(reinterpret_cast<char*>(result.data()), result.size() * sizeof(T))) throw std::runtime_error("fixture read " + path);
  return result;
}
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: gewell_mtp_head_oracle CHECKPOINT FIXTURE_DIRECTORY");
    gewell::mtp_head::Head head(argv[1], 64);
    const auto hidden = read<__nv_bfloat16>(std::string(argv[2]) + "/hidden.bf16");
    const auto context = read<float>(std::string(argv[2]) + "/context.f32");
    const auto expected = read<float>(std::string(argv[2]) + "/survival.f32");
    const auto layers = head.layers().size() + 1;
    const auto width = gewell::gemma4_31b::kHiddenSize;
    if (hidden.size() != 64 * layers * width || context.size() != 64 * 9 || expected.size() != 64 * head.max_depth())
      throw std::runtime_error("oracle requires 64 examples in checkpoint feature order");
    gewell::mtp_cuda::Buffer device(hidden.size() * sizeof(hidden[0]));
    gewell::mtp_cuda::check(cudaMemcpy(device.data(), hidden.data(), device.size(), cudaMemcpyHostToDevice), "upload oracle states");
    for (unsigned batch : {1, 2, 8, 32, 64}) {
      std::vector<gewell::mtp_head::Input> inputs(batch);
      for (unsigned b = 0; b < batch; ++b) {
        inputs[b].probes = device.at<__nv_bfloat16>() + b * layers * width;
        inputs[b].final_hidden = inputs[b].probes + (layers - 1) * width;
        std::copy_n(context.data() + b * 9, 9, inputs[b].context);
      }
      const auto result = head.predict(inputs, nullptr);
      float error = 0;
      for (unsigned i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i])) throw std::runtime_error("nonfinite prediction");
        error = std::max(error, std::abs(result[i] - expected[i]));
      }
      if (error > 3e-5F) throw std::runtime_error("PyTorch survival mismatch: " + std::to_string(error));
      for (int i = 0; i < 5; ++i) head.predict(inputs, nullptr);
      const auto begin = std::chrono::steady_clock::now();
      for (int i = 0; i < 40; ++i) head.predict(inputs, nullptr);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / 40;
      std::cout << "{\"batch\":" << batch << ",\"max_survival_error\":" << error << ",\"wall_ms\":" << ms << "}\n";
    }
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
