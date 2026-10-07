#pragma once

#include "gewell/models/gemma4/31b/model.h"
#include "gewell/storage_type.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gewell::artifact {

inline constexpr std::uint32_t kFormatVersion = 4;
inline constexpr std::uint32_t kNvfp4FormatVersion = 2;
inline constexpr std::uint32_t kMixedFormatVersion = 2;
inline constexpr std::uint64_t kHeaderBytes = 4'096;
inline constexpr std::uint64_t kEntryBytes = 72;
inline constexpr std::uint64_t kEntriesOffset = kHeaderBytes;
inline constexpr std::uint64_t kDataOffset = 65'536;
inline constexpr std::uint64_t kLogicalDataBytes = 61'394'690'680ULL;
inline constexpr std::uint64_t kPayloadBytes = 61'395'726'336ULL;
inline constexpr std::uint64_t kFileBytes = 61'395'791'872ULL;
inline constexpr std::size_t kIoChunkBytes = 8 * 1'024 * 1'024;

using Digest = std::array<std::uint8_t, 32>;

[[nodiscard]] constexpr std::uint64_t fp8_packed_bytes(
    std::uint32_t rows, std::uint32_t columns) {
  return static_cast<std::uint64_t>(rows) * columns;
}

[[nodiscard]] constexpr std::uint64_t nvfp4_packed_bytes(
    std::uint32_t rows, std::uint32_t columns) {
  return static_cast<std::uint64_t>(rows) * columns / 2;
}

[[nodiscard]] constexpr std::uint64_t nvfp4_scale_bytes(
    std::uint32_t rows, std::uint32_t columns) {
  return gemma4_31b::align_up(rows, 128) *
         gemma4_31b::align_up(columns / 16, 4);
}

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct Header {
  bool native_nvfp4{};
  bool native_mixed{};
  std::uint32_t format_version{};
  std::uint32_t physical_tensor_count{};
  std::uint32_t logical_tensor_count{};
  std::uint32_t alias_count{};
  std::uint64_t data_offset{};
  std::uint64_t logical_data_bytes{};
  std::uint64_t payload_bytes{};
  std::uint64_t file_bytes{};
  std::uint32_t lm_head_logical_id{};
  std::uint32_t lm_head_target_id{};
  Digest config_sha256{};
  Digest source_index_sha256{};
  Digest entry_table_sha256{};
  Digest payload_sha256{};
  Digest header_sha256{};

};

// Wire entry: HhHBBIIIQQ32s4x. Byte 7 stores StorageType in the native profile.
struct TensorEntry {
  std::uint16_t physical_id{};
  std::int16_t layer{};
  gemma4_31b::TensorRole role{};
  std::uint8_t rank{};
  StorageType storage_type{StorageType::bf16};
  std::uint32_t dim0{};
  std::uint32_t dim1{};
  std::uint32_t dim2{};
  std::uint64_t file_offset{};
  std::uint64_t byte_length{};
  Digest sha256{};
  float weight_scale_2{};  // Weight dequantization scale for both packed formats.
  float input_scale{};
};

struct Verification {
  Digest payload_sha256{};
};

class ArtifactFile {
 public:
  static ArtifactFile Open(const std::string& path);

  ArtifactFile() = default;
  ~ArtifactFile();
  ArtifactFile(const ArtifactFile&) = delete;
  ArtifactFile& operator=(const ArtifactFile&) = delete;
  ArtifactFile(ArtifactFile&& other) noexcept;
  ArtifactFile& operator=(ArtifactFile&& other) noexcept;

  [[nodiscard]] const std::string& path() const { return path_; }
  [[nodiscard]] const Header& header() const { return header_; }
  [[nodiscard]] const std::array<TensorEntry,
                                 gemma4_31b::kTextPhysicalTensorCount>&
  entries() const {
    return entries_;
  }
  [[nodiscard]] const std::uint8_t* payload_data() const;
  [[nodiscard]] const std::uint8_t* tensor_data(std::size_t physical_id) const;
  [[nodiscard]] Verification VerifyFull() const;

 private:
  ArtifactFile(std::string path, int descriptor, const std::uint8_t* mapping,
               std::size_t mapping_bytes);
  void ValidateMetadata();
  void Reset() noexcept;

  std::string path_;
  int descriptor_{-1};
  const std::uint8_t* mapping_{nullptr};
  std::size_t mapping_bytes_{};
  Header header_{};
  std::array<TensorEntry, gemma4_31b::kTextPhysicalTensorCount> entries_{};
};

[[nodiscard]] std::string digest_hex(const Digest& digest);

}  // namespace gewell::artifact
