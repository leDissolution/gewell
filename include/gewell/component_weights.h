#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gewell::component {
// Model-owned names, shapes and IDs; the reader supplies no model defaults.
struct TensorSpec {
  std::size_t physical_id;
  std::string name;
  std::vector<std::uint32_t> shape;
  std::size_t byte_count() const {
    std::size_t bytes = 2;
    for (auto width : shape) bytes *= width;
    return bytes;
  }
};
struct Tensor {
  std::size_t physical_id;
  const std::uint8_t* data;
  std::size_t bytes;
};

// Strict BF16 safetensors inventory. Payload is mapped read-only; returned
// pointers remain valid for this File's lifetime. Device sizes align to 4KiB.
class File {
 public:
  File(const std::string& path, const std::vector<TensorSpec>& specs);
  ~File();
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  const std::vector<Tensor>& tensors() const { return tensors_; }
  std::size_t device_bytes() const { return device_bytes_; }
  // Unmaps resident pages from this process; later reads fault back from the page cache.
  void drop_resident_pages() const;
 private:
  const std::uint8_t* mapping_{};
  std::size_t mapping_bytes_{};
  std::size_t device_bytes_{};
  std::vector<Tensor> tensors_;
};
}  // namespace gewell::component
