#include "gewell/models/gemma4/26b_a4b/artifact.h"
#include <openssl/evp.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

namespace gewell::gemma4_26b_a4b {
namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
std::uint64_t integer(const std::uint8_t* bytes, unsigned width) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < width; ++i) value |= std::uint64_t(bytes[i]) << (8 * i);
  return value;
}
Digest digest_at(const std::uint8_t* p) {
  Digest value{};
  std::copy_n(p, value.size(), value.begin());
  return value;
}
bool zero(const std::uint8_t* data, std::size_t size) {
  return std::all_of(data, data + size, [](std::uint8_t v) { return v == 0; });
}
class Hasher {
 public:
  Hasher() : context_(EVP_MD_CTX_new(), EVP_MD_CTX_free) {
    require(context_ && EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) == 1, "SHA256 init failed");
  }
  void Add(const std::uint8_t* p, std::size_t size) {
    require(EVP_DigestUpdate(context_.get(), p, size) == 1, "SHA256 update failed");
  }
  Digest Finish() {
    Digest result{};
    unsigned length = 0;
    require(EVP_DigestFinal_ex(context_.get(), result.data(), &length) == 1 && length == result.size(),
            "SHA256 final failed");
    return result;
  }
 private:
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_;
};
Digest hash(const std::uint8_t* data, std::size_t size) {
  Hasher state;
  state.Add(data, size);
  return state.Finish();
}
bool projection(TensorRole role) {
  switch (role) {
    case TensorRole::q_proj: case TensorRole::k_proj: case TensorRole::v_proj:
    case TensorRole::o_proj: case TensorRole::gate_proj: case TensorRole::up_proj:
    case TensorRole::down_proj: case TensorRole::expert_gate_proj:
    case TensorRole::expert_up_proj: case TensorRole::expert_down_proj: return true;
    default: return false;
  }
}
std::uint64_t tensor_bytes(const TensorSpec& spec, StorageType storage) {
  if (storage == StorageType::bf16) return spec.byte_count();
  require(projection(spec.role), "quantized 26B tensor is not a projection");
  if (storage == StorageType::fp8_w8a8) return spec.element_count() + 8;
  require(storage == StorageType::nvfp4_w4a4, "invalid 26B storage type");
  const auto rows = align_up(spec.rows, 128);
  return rows * spec.columns / 2 + rows * align_up(spec.columns / 16, 4) + 8;
}
void validate_globals(const std::uint8_t* bytes) {
  float weight, input;
  const std::uint32_t w = integer(bytes, 4), a = integer(bytes + 4, 4);
  std::memcpy(&weight, &w, 4);
  std::memcpy(&input, &a, 4);
  require(std::isfinite(weight) && weight > 0 && std::isfinite(input) && input > 0 &&
          std::isfinite(1.f / input) && std::isfinite(weight * input) && weight * input > 0,
          "invalid 26B global scales");
}
void validate_quantized_payload(const TensorSpec& spec, const TensorEntry& entry,
                                const std::uint8_t* data) {
  if (entry.storage_type == StorageType::fp8_w8a8) {
    require(std::none_of(data, data + spec.element_count(),
                        [](std::uint8_t v) { return (v & 0x7f) == 0x7f; }), "nonfinite 26B FP8 payload");
  } else if (entry.storage_type == StorageType::nvfp4_w4a4) {
    const auto rows = align_up(spec.rows, 128);
    const auto logical_bytes = spec.element_count() / 2;
    const auto packed_bytes = rows * spec.columns / 2;
    require(zero(data + logical_bytes, packed_bytes - logical_bytes), "nonzero 26B NVFP4 weight padding");
    const auto columns = align_up(spec.columns / 16, 4);
    const auto* scales = data + packed_bytes;
    // Each 128-row tile permutes [4,32,Ktiles,4] to [Ktiles,32,4,4].
    for (std::uint64_t row = 0; row < rows; ++row) {
      for (std::uint64_t column = 0; column < columns; ++column) {
        const auto local = row % 128;
        const auto offset = (row / 128) * 128 * columns +
            (((column / 4) * 32 + local % 32) * 4 + local / 32) * 4 + column % 4;
        const auto value = scales[offset];
        if (row >= spec.rows || column >= spec.columns / 16)
          require(value == 0, "nonzero 26B NVFP4 scale padding");
        else require(value < 0x7f, "invalid 26B NVFP4 block scale");
      }
    }
  }
}
}  // namespace

Digest ArtifactFile::header_hash() const { return digest_at(data_ + 296); }
Digest ArtifactFile::entry_table_hash() const { return digest_at(data_ + 232); }
Digest ArtifactFile::config_hash() const { return digest_at(data_ + 168); }
Digest ArtifactFile::source_index_hash() const { return digest_at(data_ + 200); }
bool ArtifactFile::is_mixed() const { return std::memcmp(data_, "G4A4MIX1", 8) == 0; }

ArtifactFile::~ArtifactFile() { Reset(); }
void ArtifactFile::Reset() noexcept {
  if (data_) munmap(const_cast<std::uint8_t*>(data_), size_);
  if (fd_ >= 0) close(fd_);
  data_ = nullptr;
  fd_ = -1;
  size_ = 0;
}
ArtifactFile::ArtifactFile(ArtifactFile&& other) noexcept { *this = std::move(other); }
ArtifactFile& ArtifactFile::operator=(ArtifactFile&& other) noexcept {
  if (this != &other) {
    Reset();
    fd_ = std::exchange(other.fd_, -1);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    entries_ = std::move(other.entries_);
    payload_hash_ = other.payload_hash_;
  }
  return *this;
}
ArtifactFile ArtifactFile::Open(const std::string& path) {
  ArtifactFile file;
  file.fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  require(file.fd_ >= 0, "cannot open 26B artifact");
  struct stat status{};
  require(fstat(file.fd_, &status) == 0 && S_ISREG(status.st_mode) &&
              status.st_size >= static_cast<off_t>(kArtifactDataOffset), "truncated 26B artifact");
  file.size_ = static_cast<std::size_t>(status.st_size);
  void* mapping = mmap(nullptr, file.size_, PROT_READ, MAP_PRIVATE, file.fd_, 0);
  require(mapping != MAP_FAILED, "cannot map 26B artifact");
  file.data_ = static_cast<const std::uint8_t*>(mapping);
  file.ValidateMetadata();
  return file;
}
void ArtifactFile::ValidateMetadata() {
  const bool mixed = std::memcmp(data_, "G4A4MIX1", 8) == 0;
  require(mixed || std::memcmp(data_, "G4A4BF16", 8) == 0, "wrong 26B artifact identity");
  const std::uint32_t header_fields[] = {1, 4096, 72, kTextPhysicalTensorCount,
      kTextPhysicalTensorCount + 1, 1, 4096, 1, 1, 1};
  for (unsigned i = 0; i < 10; ++i)
    require(integer(data_ + 8 + i * 4, 4) == header_fields[i], "invalid 26B header geometry");
  require(integer(data_ + 88, 4) == kLmHeadLogicalId && integer(data_ + 92, 4) == kLmHeadPhysicalId,
          "invalid 26B tied-head alias");
  require(zero(data_ + 96, 72) && zero(data_ + 328, kArtifactHeaderBytes - 328), "nonzero 26B header reserved bytes");
  std::array<std::uint8_t, kArtifactHeaderBytes> header{};
  std::copy_n(data_, header.size(), header.begin());
  std::fill_n(header.begin() + 296, 32, 0);
  require(hash(header.data(), header.size()) == digest_at(data_ + 296), "26B header checksum mismatch");
  require(hash(data_ + kArtifactHeaderBytes, kTextPhysicalTensorCount * kArtifactEntryBytes) == digest_at(data_ + 232),
          "26B entry table checksum mismatch");
  payload_hash_ = digest_at(data_ + 264);
  std::uint64_t offset = kArtifactDataOffset;
  std::uint64_t logical_bytes = 0;
  entries_.reserve(kTextPhysicalTensorCount);
  for (std::size_t i = 0; i < kTextPhysicalTensorCount; ++i) {
    const auto& spec = kTextTensors[i];
    const auto* entry = data_ + kArtifactHeaderBytes + i * kArtifactEntryBytes;
    require(entry[7] <= 2 && (mixed || entry[7] == 0), "invalid 26B storage type");
    const auto storage = static_cast<StorageType>(entry[7]);
    const auto bytes = tensor_bytes(spec, storage);
    require(integer(entry, 2) == i && integer(entry + 2, 2) == static_cast<std::uint16_t>(spec.layer) &&
            integer(entry + 4, 2) == static_cast<std::uint16_t>(spec.role) &&
            entry[6] == spec.rank && integer(entry + 8, 4) == spec.rows &&
            integer(entry + 12, 4) == spec.columns && integer(entry + 16, 4) == 0 &&
            integer(entry + 20, 8) == offset && integer(entry + 28, 8) == bytes &&
            zero(entry + 68, 4), "invalid 26B tensor entry");
    entries_.push_back({offset, bytes, digest_at(entry + 36), storage});
    offset += align_up(bytes);
    logical_bytes += bytes;
  }
  require(integer(data_ + 48, 8) == kArtifactHeaderBytes &&
          integer(data_ + 56, 8) == kArtifactDataOffset &&
          integer(data_ + 64, 8) == logical_bytes &&
          integer(data_ + 72, 8) == offset - kArtifactDataOffset &&
          integer(data_ + 80, 8) == offset && size_ == offset, "invalid 26B artifact size");
  for (const auto& entry : entries_)
    if (entry.storage_type != StorageType::bf16)
      validate_globals(data_ + entry.offset + entry.bytes - 8);
}
const std::uint8_t* ArtifactFile::tensor_data(std::size_t id) const {
  return data_ + entries_.at(id).offset;
}
void ArtifactFile::VerifyFull() const {
  require(data_ != nullptr, "26B artifact is not open");
  const auto table_end = kArtifactHeaderBytes + kTextPhysicalTensorCount * kArtifactEntryBytes;
  require(zero(data_ + table_end, kArtifactDataOffset - table_end), "nonzero 26B table padding");
  Hasher payload;
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    const auto& entry = entries_[i];
    validate_quantized_payload(kTextTensors[i], entry, data_ + entry.offset);
    Hasher tensor;
    for (std::uint64_t consumed = 0; consumed < entry.bytes;) {
      const auto size = std::min<std::uint64_t>(entry.bytes - consumed, 8 * 1024 * 1024);
      const auto* bytes = data_ + entry.offset + consumed;
      tensor.Add(bytes, size);
      payload.Add(bytes, size);
      consumed += size;
    }
    require(tensor.Finish() == entry.sha256, "26B tensor checksum mismatch");
    const auto padding = align_up(entry.bytes) - entry.bytes;
    const auto* bytes = data_ + entry.offset + entry.bytes;
    require(zero(bytes, padding), "nonzero 26B tensor padding");
    payload.Add(bytes, padding);
  }
  require(payload.Finish() == payload_hash_, "26B payload checksum mismatch");
}
}  // namespace gewell::gemma4_26b_a4b
