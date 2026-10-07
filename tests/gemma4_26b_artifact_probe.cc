#include "gewell/models/gemma4/26b_a4b/artifact.h"
#include <iostream>
#include <stdexcept>
#include <string_view>

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string_view(argv[2]) != "--verify"))
      throw std::runtime_error("usage: artifact_probe FILE [--verify]");
    auto file = gewell::gemma4_26b_a4b::ArtifactFile::Open(argv[1]);
    if (argc == 3) file.VerifyFull();
    std::cout << "26B artifact: " << file.entries().size() << " tensors, " << file.file_bytes()
              << " bytes; " << (argc == 3 ? "full verification" : "metadata validation") << " passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
