#include "gewell/models/gemma4/26b_a4b/serving_assets.h"
#include "gewell/models/gemma4/text_contract.h"
#include "../serving_assets_detail.h"

namespace gewell::gemma4_26b_a4b {
using namespace gemma4::serving_detail;
ServingAssets ServingAssets::Open(const std::string& model_directory) {
  const fs::path directory(model_directory);
  try {
    const auto manifest = json::parse(read_file(directory / "manifest.json", 8 * 1024 * 1024));
    keys(manifest, {"aliases", "architecture", "artifact", "format", "layout", "model",
                   "schema_version", "serving", "tensors"}, "manifest");
    require(manifest.at("schema_version").is_number_unsigned() && manifest.at("schema_version") == 6,
            "manifest schema must be integer 6; convert the model first");
    require((manifest.at("format") == "gemma4-26b-a4b-bf16-v1" ||
             manifest.at("format") == "gemma4-26b-a4b-mixed-v1") &&
            manifest.at("architecture") == "gemma4_26b_a4b", "wrong 26B artifact format");
    auto tokenizer_data = read_tokenizer(directory, manifest);
    const auto& metadata = manifest.at("artifact");
    const auto filename = metadata.at("file").get<std::string>();
    require(!filename.empty() && fs::path(filename).filename() == filename &&
            filename != "." && filename != "..", "artifact filename must be a local basename");
    const auto path = (directory / filename).string();
    regular_file(path);
    auto weights = ArtifactFile::Open(path);
    require(manifest.at("format") == (weights.is_mixed() ? "gemma4-26b-a4b-mixed-v1" : "gemma4-26b-a4b-bf16-v1"),
            "manifest format does not match artifact identity");
    require(metadata.at("file_bytes").is_number_unsigned() &&
            metadata.at("file_bytes") == weights.file_bytes() &&
            metadata.at("header_sha256") == digest_hex(weights.header_hash()) &&
            metadata.at("entry_table_sha256") == digest_hex(weights.entry_table_hash()) &&
            metadata.at("payload_sha256") == digest_hex(weights.payload_hash()),
            "weight manifest does not match artifact metadata");
    require(manifest.at("model").at("config_sha256") == digest_hex(weights.config_hash()) &&
            manifest.at("model").at("index_sha256") == digest_hex(weights.source_index_hash()),
            "model manifest does not match artifact metadata");
    return {std::move(weights), text::Tokenizer::FromJson(tokenizer_data, gemma4::text_contract_26b_a4b())};
  } catch (const json::exception& error) {
    throw std::runtime_error("serving assets: invalid manifest in " + model_directory + ": " + error.what());
  } catch (const fs::filesystem_error& error) {
    throw std::runtime_error("serving assets: " + std::string(error.what()));
  }
}
}
