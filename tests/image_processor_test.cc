#include "gewell/models/gemma4/image_processor.h"

#include <openssl/evp.h>
#include <json.hpp>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, 32> digest;
  unsigned int size = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), digest.data(), &size, EVP_sha256(), nullptr) != 1 || size != digest.size()) {
    throw std::runtime_error("SHA256 failed");
  }
  std::ostringstream result;
  for (const auto byte : digest) result << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
  return result.str();
}

// Pinned HF fixtures hash FP32 pixel_values; uint8 pixels rescale exactly.
std::string pixels_sha256(const std::vector<std::uint8_t>& pixels) {
  std::vector<std::uint8_t> bytes(pixels.size() * sizeof(float));
  for (std::size_t i = 0; i < pixels.size(); ++i) {
    const float value = pixels[i] * (1.0F / 255.0F);
    std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(float));
  }
  return sha256(bytes);
}

void require_invalid(std::string_view data, std::string_view name) {
  try {
    (void)gewell::gemma4::prepare_image_data_url(data);
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(std::string(name) + " was accepted");
}

std::string base64(const std::vector<unsigned char>& bytes) {
  std::string encoded(4 * ((bytes.size() + 2) / 3), '\0');
  // EVP writes an additional NUL byte.
  encoded.resize(encoded.size() + 1);
  const int size = EVP_EncodeBlock(reinterpret_cast<unsigned char*>(encoded.data()), bytes.data(), bytes.size());
  encoded.resize(size);
  return encoded;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) throw std::runtime_error("expected fixture JSON path");
    std::ifstream stream(argv[1]);
    const auto fixture = nlohmann::json::parse(stream);
    for (const auto& test : fixture.at("cases")) {
      const std::string name = test.at("name");
      for (const auto& expected : test.at("budgets")) {
        const auto budget = expected.at("max_soft_tokens").get<std::uint32_t>();
        const auto image = gewell::gemma4::prepare_image_data_url(test.at("data_url").get<std::string>(), budget);
        if (image.padded_patch_rows != budget * 9 || image.soft_token_count != expected.at("soft_token_count") ||
            image.pixels.size() != std::size_t(budget) * 9 * 768 || image.positions.size() != budget * 9 * 8 ||
            pixels_sha256(image.pixels) != expected.at("pixels_sha256") || sha256(image.positions) != expected.at("positions_sha256")) {
          std::cerr << name << " budget=" << budget << ": pixels=" << pixels_sha256(image.pixels)
                    << " positions=" << sha256(image.positions) << " soft_tokens=" << image.soft_token_count << '\n';
          throw std::runtime_error(name + " differs from pinned processor");
        }
        if (budget == 280) {
          const auto defaulted = gewell::gemma4::prepare_image_data_url(test.at("data_url").get<std::string>());
          if (defaulted.pixels != image.pixels || defaulted.positions != image.positions)
            throw std::runtime_error(name + " default differs from explicit 280 budget");
        }
        std::cout << name << " budget=" << budget << " exact\n";
      }
    }
    for (const auto& test : fixture.at("rejected")) {
      require_invalid(test.at("data_url").get<std::string>(), test.at("name").get<std::string>());
    }
    for (const auto* bad : {"https://example.com/a.png", "file:///tmp/a.png", "data:image/gif;base64,AAAA",
                           "data:image/png;base64,", "data:image/png;base64,A", "data:image/png;base64,AB==",
                           "data:image/png;base64,AAB=", "data:image/png;base64,AA=A", "data:image/png;base64,====",
                           "data:image/png;base64,AA==AAAA", "data:image/png;base64,AAAA\nAAA", "data:image/png;base64,AAAA"}) {
      require_invalid(bad, "malformed URL");
    }
    auto valid = fixture.at("cases").at(0).at("data_url").get<std::string>();
    for (const auto budget : {0U, 1U, 71U, 281U, 1280U, 0xffffffffU}) {
      bool rejected = false;
      try { (void)gewell::gemma4::prepare_image_data_url(valid, budget); }
      catch (const std::invalid_argument&) { rejected = true; }
      if (!rejected) throw std::runtime_error("unsupported image budget was accepted");
    }
    auto mime = valid;
    mime.replace(11, 3, "jpeg");
    require_invalid(mime, "MIME mismatch");
    valid.resize(valid.size() - 16);
    require_invalid(valid, "truncated PNG");
    auto jpeg = fixture.at("cases").at(2).at("data_url").get<std::string>();
    jpeg.resize(jpeg.size() - 16);
    require_invalid(jpeg, "truncated JPEG");
    require_invalid(std::string(gewell::gemma4::kImageMaxDataUrlBytes + 1, 'x'), "encoded size");

    // Optional real caption source: compare the complete prepared tensor ABI.
    if (argc == 5) {
      std::ifstream encoded_stream(argv[2], std::ios::binary);
      const std::vector<unsigned char> encoded{std::istreambuf_iterator<char>(encoded_stream), {}};
      const auto image = gewell::gemma4::prepare_image_data_url("data:image/png;base64," + base64(encoded));
      if (pixels_sha256(image.pixels) != argv[3] || sha256(image.positions) != argv[4]) {
        std::cerr << "caption pixels=" << pixels_sha256(image.pixels) << " positions=" << sha256(image.positions) << '\n';
        throw std::runtime_error("caption source differs from prepared-image reference");
      }
      std::cout << "caption source exact (" << image.soft_token_count << " soft tokens)\n";
    }
    std::cout << "image processor tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
