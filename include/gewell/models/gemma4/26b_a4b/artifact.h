#pragma once
#include "gewell/models/gemma4/26b_a4b/model.h"
#include "gewell/storage_type.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gewell::gemma4_26b_a4b {
using Digest = std::array<std::uint8_t, 32>;
using artifact::StorageType;
inline constexpr std::uint64_t kArtifactHeaderBytes = 4096;
inline constexpr std::uint64_t kArtifactEntryBytes = 72;
inline constexpr std::uint64_t kArtifactDataOffset =
    align_up(kArtifactHeaderBytes + kTextPhysicalTensorCount * kArtifactEntryBytes);

struct TensorEntry {
  std::uint64_t offset{};
  std::uint64_t bytes{};
  Digest sha256{};
  StorageType storage_type{StorageType::bf16};
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
  [[nodiscard]] const std::vector<TensorEntry>& entries() const { return entries_; }
  [[nodiscard]] const std::uint8_t* tensor_data(std::size_t id) const;
  [[nodiscard]] std::uint64_t file_bytes() const { return size_; }
  [[nodiscard]] const Digest& payload_hash() const { return payload_hash_; }
  [[nodiscard]] Digest header_hash() const;
  [[nodiscard]] Digest entry_table_hash() const;
  [[nodiscard]] Digest config_hash() const;
  [[nodiscard]] Digest source_index_hash() const;
  [[nodiscard]] bool is_mixed() const;
  void VerifyFull() const;
 private:
  void ValidateMetadata();
  void Reset() noexcept;
  int fd_{-1};
  const std::uint8_t* data_{};
  std::size_t size_{};
  std::vector<TensorEntry> entries_;
  Digest payload_hash_{};
};
}  // namespace gewell::gemma4_26b_a4b
