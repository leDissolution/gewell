#include "gewell/models/gemma4/26b_a4b/executor.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 4) throw std::runtime_error("usage: forward ARTIFACT OUTPUT_DIRECTORY [--boundaries] [--greedy N] TOKEN...");
    namespace fs = std::filesystem;
    const fs::path output(argv[2]);
    if (fs::exists(output)) throw std::runtime_error("refusing existing capture directory");
    std::vector<unsigned> tokens;
    bool boundaries = false;
    unsigned greedy = 0;
    for (int i = 3; i < argc; ++i) {
      std::string arg(argv[i]);
      if (arg == "--boundaries") { boundaries = true; continue; }
      bool count = arg == "--greedy";
      if (count) {
        if (++i == argc) throw std::runtime_error("missing greedy count");
        arg = argv[i];
      }
      std::size_t consumed = 0;
      auto value = std::stoul(arg, &consumed);
      if (consumed != arg.size() || (count ? value > 262144 : value >= 262144)) throw std::runtime_error("invalid token/count");
      if (count) greedy = value; else tokens.push_back(value);
    }
    const auto fed_tokens = tokens.size() + (greedy ? greedy-1 : 0);
    if (tokens.empty() || fed_tokens > 262144) throw std::runtime_error("invalid context length");
    gewell::gemma4_26b_a4b::sm120::Executor executor(argv[1], std::max<std::size_t>(fed_tokens, 1026));
    fs::create_directories(output);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      auto token = tokens[i];
      bool capture = !boundaries || i == 0 || i == 254 || i == 255 || i == 256 ||
                     i == 1022 || i == 1023 || i == 1024 || i == 1025;
      const fs::path step = output / std::to_string(i);
      gewell::gemma4_26b_a4b::sm120::Capture sink;
      if (capture) {
        fs::create_directory(step);
        sink = [&](const auto& name, const auto& bits) {
          std::ofstream file(step / (name + ".bf16"), std::ios::binary);
          file.write(reinterpret_cast<const char*>(bits.data()), bits.size()*2);
          if (!file) throw std::runtime_error("capture write failed");
        };
      }
      auto result = executor.ForwardToken(token,sink);
      float maximum = -1e30f;
      unsigned best = 0;
      for (unsigned j = 0; j < result.size(); ++j) {
        std::uint32_t bits = std::uint32_t(result[j]) << 16;
        float value; std::memcpy(&value, &bits, 4);
        if (value > maximum) { maximum = value; best = j; }
      }
      if (i+1 == tokens.size() && tokens.size() < fed_tokens) tokens.push_back(best);
      if (capture || i % 128 == 0) std::cout << "position=" << i << " input=" << token << " argmax=" << best << " logit=" << maximum << std::endl;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
