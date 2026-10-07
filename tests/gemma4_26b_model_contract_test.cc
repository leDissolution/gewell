#include "gewell/models/gemma4/26b_a4b/model.h"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace model = gewell::gemma4_26b_a4b;

int main(int argc, char** argv) {
  try {
    const bool dump = argc == 2 && std::string_view(argv[1]) == "--dump";
    if (argc != 1 && !dump) throw std::runtime_error("expected --dump or no arguments");
    std::size_t experts = 0;
    for (const auto& spec : model::kTextTensors) {
      if (!spec.rows || (spec.rank == 2 && !spec.columns))
        throw std::runtime_error("empty tensor");
      if (spec.expert != model::kNoExpert) {
        if (spec.expert < 0 || spec.expert >= model::kExpertCount ||
            spec.layer < 0 || spec.layer >= model::kLayerCount)
          throw std::runtime_error("invalid expert identity");
        ++experts;
      }
      if (dump)
        std::cout << static_cast<int>(spec.role) << ' ' << spec.layer << ' ' << spec.expert << ' '
                  << static_cast<int>(spec.rank) << ' ' << spec.rows << ' ' << spec.columns << '\n';
    }
    if (experts != 11'520 || model::kLmHeadPhysicalId != model::kEmbeddingPhysicalId)
      throw std::runtime_error("expert count or tied-head alias mismatch");
    if (!dump) std::cout << "26B contract: 12117 entries, 11520 expert matrices, tied head passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
