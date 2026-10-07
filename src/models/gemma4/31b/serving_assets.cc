#include "gewell/models/gemma4/31b/serving_assets.h"
#include "gewell/models/gemma4/text_contract.h"

#include "../serving_assets_detail.h"

#include <openssl/evp.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace gewell::gemma4_31b {
using namespace gemma4::serving_detail;

ServingAssets ServingAssets::Open(const std::string& model_directory) {
  const fs::path directory(model_directory);
  const auto& contract = gemma4::text_contract_31b();
  try {
    const auto manifest = json::parse(read_file(directory / "manifest.json", 4 * 1024 * 1024));
    keys(manifest, {"aliases", "artifact", "format", "layout", "model",
                    "schema_version", "serving", "source", "tensors"}, "manifest");
    require(manifest.at("schema_version").is_number_unsigned() && manifest.at("schema_version") == 6,
            "manifest schema must be integer 6; convert the model first");
    const bool native_nvfp4 = manifest.at("format") == "gemma4-31b-nvfp4-w4a4-v2";
    const bool native_mixed = manifest.at("format") == "gemma4-31b-mixed-v2";
    require(native_nvfp4 || native_mixed || manifest.at("format") == "gemma4-31b-bf16-v4", "wrong artifact format");
    auto tokenizer_data = read_tokenizer(directory, manifest);
    const auto& metadata = manifest.at("artifact");
    const auto filename = metadata.at("file").get<std::string>();
    require(!filename.empty() && fs::path(filename).filename() == filename &&
                filename != "." && filename != "..", "artifact filename must be a local basename");
    regular_file(directory / filename);
    auto weights = artifact::ArtifactFile::Open((directory / filename).string());
    const auto& header = weights.header();
    require(header.native_nvfp4 == native_nvfp4 && header.native_mixed == native_mixed,
            "manifest format does not match artifact");
    require(metadata.at("file_bytes").is_number_unsigned() &&
                metadata.at("file_bytes") == header.file_bytes &&
                metadata.at("header_sha256") == artifact::digest_hex(header.header_sha256) &&
                metadata.at("entry_table_sha256") == artifact::digest_hex(header.entry_table_sha256) &&
                metadata.at("payload_sha256") == artifact::digest_hex(header.payload_sha256),
            "weight manifest does not match artifact metadata");
    require(manifest.at("model").at("config_sha256") == artifact::digest_hex(header.config_sha256) &&
                manifest.at("model").at("index_sha256") == artifact::digest_hex(header.source_index_sha256),
            "model manifest does not match artifact metadata");
    return {std::move(weights), text::Tokenizer::FromJson(tokenizer_data, contract)};
  } catch (const json::exception& error) {
    throw std::runtime_error("serving assets: invalid manifest in " + model_directory + ": " + error.what());
  } catch (const fs::filesystem_error& error) {
    throw std::runtime_error("serving assets: " + std::string(error.what()));
  }
}

}  // namespace gewell::gemma4_31b
