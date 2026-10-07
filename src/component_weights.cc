#include "gewell/component_weights.h"
#include "json.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace gewell::component {
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("component weights: " + message);
}
constexpr std::size_t kMaxHeaderBytes = 4 * 1024 * 1024;
}

File::File(const std::string& path, const std::vector<TensorSpec>& specs) {
  const auto count = specs.size();
  require(count > 0, "empty component contract");
  std::size_t logical_bytes = 0;
  for (const auto& spec : specs) {
    logical_bytes += spec.byte_count();
    device_bytes_ += (spec.byte_count() + 4095) / 4096 * 4096;
  }
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  require(fd >= 0, "cannot open " + path);
  struct stat status{};
  const bool valid = ::fstat(fd, &status) == 0 && S_ISREG(status.st_mode) &&
      status.st_size >= 8 && std::uint64_t(status.st_size) <= logical_bytes + kMaxHeaderBytes + 8;
  if (!valid) {
    ::close(fd);
    require(false, "invalid safetensors file size/type: " + path);
  }
  mapping_bytes_ = static_cast<std::size_t>(status.st_size);
  void* mapped = ::mmap(nullptr, mapping_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  require(mapped != MAP_FAILED, "cannot map " + path);
  mapping_ = static_cast<const std::uint8_t*>(mapped);
  try {
    std::uint64_t header_bytes = 0;
    for (unsigned i = 0; i < 8; ++i) header_bytes |= std::uint64_t(mapping_[i]) << (8 * i);
    require(header_bytes > 0 && header_bytes <= kMaxHeaderBytes && header_bytes <= mapping_bytes_ - 8,
            "invalid safetensors header length");
    const auto data_offset = 8 + header_bytes;
    using Json = nlohmann::json;
    std::vector<std::set<std::string>> keys;
    const auto header = Json::parse(mapping_ + 8, mapping_ + data_offset,
        [&](int, Json::parse_event_t event, Json& value) {
          if (event == Json::parse_event_t::object_start) keys.emplace_back();
          if (event == Json::parse_event_t::key)
            require(keys.back().insert(value.get<std::string>()).second, "duplicate safetensors key");
          if (event == Json::parse_event_t::object_end) keys.pop_back();
          return true;
        });
    require(header.is_object() && header.size() == count + header.count("__metadata__"),
            "wrong tensor inventory");
    if (header.contains("__metadata__")) {
      require(header["__metadata__"].is_object(), "invalid safetensors metadata");
      for (const auto& value : header["__metadata__"])
        require(value.is_string(), "safetensors metadata values must be strings");
    }
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    for (const auto& spec : specs) {
      const auto& name = spec.name;
      require(header.contains(name), "missing tensor " + name);
      const auto& entry = header[name];
      require(entry.is_object() && entry.size() == 3 && entry.contains("dtype") &&
                  entry.contains("shape") && entry.contains("data_offsets"), "invalid tensor entry: " + name);
      require(entry["dtype"] == "BF16", "only BF16 component weights are supported: " + name);
      const auto& shape = entry["shape"];
      require(shape.is_array() && shape.size() == spec.shape.size(), "wrong shape: " + name);
      for (std::size_t axis = 0; axis < shape.size(); ++axis)
        require(shape[axis].is_number_unsigned() && shape[axis] == spec.shape[axis], "wrong shape: " + name);
      const auto& offsets = entry["data_offsets"];
      require(offsets.is_array() && offsets.size() == 2 && offsets[0].is_number_unsigned() &&
                  offsets[1].is_number_unsigned(), "invalid offsets: " + name);
      const auto begin = offsets[0].get<std::uint64_t>();
      const auto end = offsets[1].get<std::uint64_t>();
      require(begin <= end && end <= mapping_bytes_ - data_offset && end - begin == spec.byte_count(),
              "wrong byte range: " + name);
      spans.emplace_back(begin, end);
      tensors_.push_back({spec.physical_id, mapping_ + data_offset + begin, spec.byte_count()});
    }
    std::sort(spans.begin(), spans.end());
    std::size_t cursor = 0;
    for (const auto& [begin, end] : spans) {
      require(begin == cursor, "safetensors payload has gaps or overlapping tensors");
      cursor = end;
    }
    require(cursor == mapping_bytes_ - data_offset, "safetensors payload has trailing bytes");
  } catch (...) {
    ::munmap(const_cast<std::uint8_t*>(mapping_), mapping_bytes_);
    mapping_ = nullptr;
    throw;
  }
}

File::~File() {
  if (mapping_) ::munmap(const_cast<std::uint8_t*>(mapping_), mapping_bytes_);
}
}  // namespace gewell::component
