#pragma once
#include "json.hpp"
#include <openssl/evp.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace gewell::gemma4::serving_detail {
using nlohmann::json;
namespace fs = std::filesystem;

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("serving assets: " + message);
}

inline void regular_file(const fs::path& path) {
  std::error_code error;
  const auto status = fs::symlink_status(path, error);
  require(!error && fs::is_regular_file(status),
          "expected a local regular file (no symlinks): " + path.string());
}

inline std::string read_file(const fs::path& path, std::uintmax_t limit) {
  regular_file(path);
  const auto size = fs::file_size(path);
  require(size <= limit, "file is too large: " + path.string());
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot open " + path.string());
  std::string bytes(static_cast<std::size_t>(size), '\0');
  input.read(bytes.data(), static_cast<std::streamsize>(size));
  require(input.good() && input.peek() == std::char_traits<char>::eof(),
          "file changed or could not be read: " + path.string());
  return bytes;
}

inline std::string digest_hex(const std::array<unsigned char, 32>& digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(64, '0');
  for (std::size_t i = 0; i < digest.size(); ++i) {
    result[2*i] = digits[digest[i] >> 4];
    result[2*i+1] = digits[digest[i] & 15];
  }
  return result;
}

inline std::string sha256(const std::string& bytes) {
  std::array<unsigned char, 32> digest;
  unsigned int size = 0;
  require(EVP_Digest(bytes.data(), bytes.size(), digest.data(), &size, EVP_sha256(), nullptr) == 1 &&
              size == digest.size(), "SHA-256 failed");
  return digest_hex(digest);
}

inline void keys(const json& value, const std::initializer_list<const char*> expected,
          const std::string& label) {
  require(value.is_object() && value.size() == expected.size(), label + " has unexpected fields");
  for (const char* key : expected) require(value.contains(key), label + " is missing " + key);
}

inline std::string read_tokenizer(const fs::path& directory, const json& manifest) {
    const auto& serving = manifest.at("serving");
    keys(serving, {"repository", "revision", "assets"}, "serving");
    require(serving.at("repository").is_string() && serving.at("revision").is_string(),
            "serving provenance must be strings");
    const auto& assets = serving.at("assets");
    require(assets.is_array() && assets.size() == 1, "wrong serving asset inventory");
    const auto& tokenizer = assets[0];
    keys(tokenizer, {"file", "byte_length", "sha256"}, "asset entry");
    require(tokenizer.at("file") == "tokenizer.json", "wrong asset filename for tokenizer.json");
    auto tokenizer_data = read_file(directory / "tokenizer.json", 128 * 1024 * 1024);
    require(tokenizer.at("byte_length").is_number_unsigned() &&
                tokenizer.at("byte_length").get<std::uint64_t>() == tokenizer_data.size(),
            "wrong byte length for tokenizer.json");
    require(tokenizer.at("sha256").is_string() && sha256(tokenizer_data) == tokenizer.at("sha256"),
            "SHA-256 mismatch for tokenizer.json");
    return tokenizer_data;
}

}  // namespace gewell::gemma4::serving_detail
