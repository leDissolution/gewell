#include "bos_runner.h"
#include "fixtures.h"
#include "vision_runner.h"
#include "gewell/console.h"

#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace console = gewell::console;
namespace pair = gewell::diagnostics;

int main(int argc, char** argv) {
  try {
    int remaining = 1;
    for (int index = 1; index < argc; ++index) {
      if (std::string_view(argv[index]) == "--log-format") {
        if (index + 1 == argc) throw std::runtime_error("--log-format requires a value");
        console::set_format(argv[++index]);
      } else argv[remaining++] = argv[index];
    }
    argc = remaining;
    argv[argc] = nullptr;
    if (argc == 3 || argc == 4) {
      const std::string_view command = argv[1];
      const char* output = argc == 4 ? argv[3] : "-";
      if (command == "bos") return gewell::bos_runner::run(argv[2], output);
      if (command == "pair") return pair::run(argv[2], output);
      if (command == "cached-pair") return pair::run_cached(argv[2], output);
      if (command == "short-decode") return pair::run_short_decode(argv[2], output);
      if (command == "local-boundary") return pair::run_local_boundary(argv[2], output);
      if (command == "local-boundary-prefill") return pair::run_local_boundary_prefill(argv[2], output);
      if (command == "graph-decode") return pair::run_graph_decode(argv[2], output);
      if (argc == 3 && command == "profile-decode") return pair::run_profile_decode(argv[2]);
    }
    if (argc == 7 && (std::string_view(argv[1]) == "vision" || std::string_view(argv[1]) == "vision26")) {
      const std::string_view text = argv[5];
      std::uint32_t tokens = 0;
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), tokens);
      if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || tokens == 0)
        throw std::runtime_error("SOFT_TOKENS must be an integer in 1..4294967295");
      return gewell::vision_runner::run(std::string_view(argv[1]) == "vision26"
          ? gewell::vision_engine::Model::gemma4_26b_a4b : gewell::vision_engine::Model::gemma4_31b, argv[2], argv[3], argv[4], tokens, argv[6]);
    }
    console::message(std::string("Usage: ") + argv[0] + R"( [--log-format human|json] COMMAND ...
  bos | pair | cached-pair | short-decode PATH [CAPTURE_DIR|-]
  local-boundary | local-boundary-prefill PATH [CAPTURE_DIR|-]
  graph-decode PATH [OUTPUT_DIR|-]
  profile-decode PATH
  vision | vision26 VISION.safetensors PIXELS.f32 POSITIONS.i32 SOFT_TOKENS CAPTURE_DIR
)");
    return argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h") ? 0 : 2;
  } catch (const std::exception& error) {
    console::message(error.what(), true);
    return 1;
  }
}
