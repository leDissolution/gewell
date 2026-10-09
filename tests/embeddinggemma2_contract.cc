#include "gewell/models/embeddinggemma2/model.h"
#include "gewell/models/embeddinggemma2/vision.h"
#include "gewell/models/embeddinggemma2/audio.h"
#include <openssl/sha.h>
#include <iomanip>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
  try {
    namespace eg = gewell::embeddinggemma2;
    const bool vision = argc == 3 && std::string(argv[1]) == "--vision";
    const bool audio = argc == 3 && std::string(argv[1]) == "--audio";
    if (vision || audio) { --argc; ++argv; }
    const auto specs = vision ? eg::vision::tensor_specs() : audio ? eg::audio::tensor_specs() : eg::tensor_specs();
    nlohmann::json rows = nlohmann::json::array();
    if (argc == 2 && std::string(argv[1]) == "--dump") {
      for (const auto& spec : specs)
        rows.push_back({{"id", spec.physical_id}, {"name", spec.name}, {"shape", spec.shape}});
    } else if (argc == 2) {
      eg::validate_bundle_config(argv[1]);
      if (vision) eg::vision::validate_bundle_config(argv[1]);
      if (audio) eg::audio::validate_bundle_config(argv[1]);
      gewell::component::File weights(std::string(argv[1]) + (vision ? "/vision.safetensors" : audio ? "/audio.safetensors" : "/text.safetensors"), specs);
      for (const auto& tensor : weights.tensors()) {
        unsigned char digest[SHA256_DIGEST_LENGTH];
        SHA256(tensor.data, tensor.bytes, digest);
        std::ostringstream hex;
        for (auto byte : digest) hex << std::hex << std::setfill('0') << std::setw(2) << unsigned(byte);
        rows.push_back({{"name", specs[tensor.physical_id].name}, {"bytes", tensor.bytes}, {"sha256", hex.str()}});
      }
    } else throw std::runtime_error("usage: embeddinggemma2_contract [--vision|--audio] --dump | BUNDLE");
    std::cout << rows.dump() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
