#pragma once
#include "gewell/models/gemma4/26b_a4b/artifact.h"
#include "gewell/tokenizer.h"

namespace gewell::gemma4_26b_a4b {
struct ServingAssets {
  ArtifactFile weights;
  text::Tokenizer tokenizer;
  static ServingAssets Open(const std::string& model_directory);
};
}
