#include "gewell/console.h"
#include "vision_runner.h"

#include "gewell/models/gemma4/31b/component_weights.h"
#include "gewell/models/gemma4/26b_a4b/component_weights.h"
#include "gewell/models/gemma4/31b/model.h"
#include "gewell/vision_engine.h"
#include "gewell/vision_executor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gewell::vision_runner {
namespace {

namespace executor = gewell::vision_executor;
namespace model = gewell::gemma4_31b;

using BFloat16 = __nv_bfloat16;

[[noreturn]] void fail(std::string_view operation,
                       std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(operation, cudaGetErrorString(status));
  }
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    if (bytes == 0) {
      fail("vision cudaMalloc", "zero-sized allocation");
    }
    check_cuda(cudaMalloc(&pointer_, bytes), "vision cudaMalloc");
  }

  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  [[nodiscard]] void* data() const { return pointer_; }
  [[nodiscard]] std::size_t size() const { return bytes_; }

 private:
  void* pointer_{};
  std::size_t bytes_{};
};

std::vector<std::uint8_t> read_file(const std::string& path,
                                    std::size_t maximum_bytes,
                                    std::string_view label) {
  std::error_code error;
  const std::uintmax_t actual_bytes = std::filesystem::file_size(path, error);
  if (error) {
    fail(label, "cannot stat " + path + ": " + error.message());
  }
  if (actual_bytes == 0 || actual_bytes > maximum_bytes) {
    fail(label, "unsupported byte length for " + path);
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    fail(label, "cannot open " + path);
  }
  std::vector<std::uint8_t> result(static_cast<std::size_t>(actual_bytes));
  input.read(reinterpret_cast<char*>(result.data()),
             static_cast<std::streamsize>(result.size()));
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    fail(label, "short read or trailing data in " + path);
  }
  return result;
}

void write_exclusive(const std::filesystem::path& path,
                     const std::uint8_t* data, std::size_t bytes) {
  const int descriptor = ::open(path.c_str(),
                                O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                    O_NOFOLLOW,
                                0644);
  if (descriptor < 0) {
    fail("vision capture write", path.string() + ": " + std::strerror(errno));
  }
  std::size_t written = 0;
  while (written < bytes) {
    const ssize_t result = ::write(descriptor, data + written, bytes - written);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      const std::string message = path.string() + ": " + std::strerror(errno);
      ::close(descriptor);
      fail("vision capture write", message);
    }
    written += static_cast<std::size_t>(result);
  }
  if (::close(descriptor) != 0) {
    fail("vision capture close", path.string() + ": " + std::strerror(errno));
  }
}

std::size_t dtype_bytes(executor::CaptureDType dtype) {
  switch (dtype) {
    case executor::CaptureDType::bf16:
      return sizeof(BFloat16);
    case executor::CaptureDType::f32:
      return sizeof(float);
    case executor::CaptureDType::i32:
      return sizeof(std::int32_t);
  }
  fail("vision capture", "unknown dtype");
}

std::string_view dtype_extension(executor::CaptureDType dtype) {
  switch (dtype) {
    case executor::CaptureDType::bf16:
      return ".bf16";
    case executor::CaptureDType::f32:
      return ".f32";
    case executor::CaptureDType::i32:
      return ".i32";
  }
  fail("vision capture", "unknown dtype");
}

struct CaptureRecord {
  std::string name;
  executor::CaptureDType dtype{};
  std::uint32_t rows{};
  std::uint32_t columns{};
  std::size_t offset{};
  bool captured{};
};

class CapturePlan final : public executor::CaptureSink {
 public:
  CapturePlan(std::filesystem::path directory, std::uint32_t padded_patch_rows,
              std::uint32_t soft_tokens, std::uint32_t output_width)
      : directory_(std::move(directory)) {
    std::error_code error;
    if (std::filesystem::exists(directory_, error) || error) {
      fail("vision capture directory",
           error ? error.message() : "path already exists (overwrite refused)");
    }
    const std::uint32_t patches = soft_tokens * 9;
    add("input.pixel_values", executor::CaptureDType::f32,
        padded_patch_rows, model::kVisionPatchWidth);
    add("input.image_position_ids", executor::CaptureDType::i32,
        padded_patch_rows, 2);
    add("vision.patch_embedding", executor::CaptureDType::bf16, patches,
        model::kVisionHiddenSize);
    for (std::uint32_t layer = 0; layer < model::kVisionLayerCount; ++layer) {
      char name[64];
      const int length = std::snprintf(name, sizeof(name),
                                       "vision.layer.%02u.output", layer);
      if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
        fail("vision capture", "layer name exceeds fixed buffer");
      }
      add(std::string(name, static_cast<std::size_t>(length)),
          executor::CaptureDType::bf16, patches,
          model::kVisionHiddenSize);
    }
    add("vision.post_pool_scaled", executor::CaptureDType::f32, soft_tokens,
        model::kVisionHiddenSize);
    add("vision.post_standardization", executor::CaptureDType::bf16,
        soft_tokens, model::kVisionHiddenSize);
    add("vision.bridge_norm", executor::CaptureDType::bf16, soft_tokens,
        model::kVisionHiddenSize);
    add("vision.soft_features", executor::CaptureDType::bf16, soft_tokens,
        output_width);
    if (records_.size() != 34) {
      fail("vision capture", "capture inventory must contain 34 tensors");
    }
    check_cuda(cudaMallocHost(&host_data_, bytes_),
               "allocate pinned vision captures");
  }

  ~CapturePlan() override {
    if (host_data_ != nullptr) {
      cudaFreeHost(host_data_);
    }
  }

  CapturePlan(const CapturePlan&) = delete;
  CapturePlan& operator=(const CapturePlan&) = delete;

  void capture(const executor::CaptureTensor& tensor,
               cudaStream_t stream) override {
    CaptureRecord& record = find(tensor.name);
    if (record.captured) {
      fail("vision capture", record.name + " was produced twice");
    }
    if (record.dtype != tensor.dtype || record.rows != tensor.rows ||
        record.columns != tensor.columns || tensor.device_data == nullptr) {
      fail("vision capture", record.name + " has a wrong tensor contract");
    }
    const std::size_t bytes = static_cast<std::size_t>(record.rows) *
                              record.columns * dtype_bytes(record.dtype);
    check_cuda(cudaMemcpyAsync(
                   static_cast<std::uint8_t*>(host_data_) + record.offset,
                   tensor.device_data, bytes, cudaMemcpyDeviceToHost, stream),
               "copy vision capture");
    record.captured = true;
  }

  void write() {
    for (const CaptureRecord& record : records_) {
      if (!record.captured) {
        fail("vision capture", record.name + " was not produced");
      }
    }
    std::error_code error;
    if (!std::filesystem::create_directory(directory_, error)) {
      fail("vision capture directory",
           error ? error.message() : "path already exists (overwrite refused)");
    }
    for (const CaptureRecord& record : records_) {
      const std::size_t bytes = static_cast<std::size_t>(record.rows) *
                                record.columns * dtype_bytes(record.dtype);
      const std::filesystem::path path =
          directory_ / (record.name + std::string(dtype_extension(record.dtype)));
      write_exclusive(path,
                      static_cast<const std::uint8_t*>(host_data_) +
                          record.offset,
                      bytes);
    }
  }

  [[nodiscard]] std::size_t bytes() const { return bytes_; }

 private:
  void add(std::string name, executor::CaptureDType dtype, std::uint32_t rows,
           std::uint32_t columns) {
    const std::size_t tensor_bytes = static_cast<std::size_t>(rows) * columns *
                                     dtype_bytes(dtype);
    records_.push_back(
        {std::move(name), dtype, rows, columns, bytes_, false});
    bytes_ += tensor_bytes;
  }

  CaptureRecord& find(std::string_view name) {
    for (CaptureRecord& record : records_) {
      if (record.name == name) {
        return record;
      }
    }
    fail("vision capture", std::string(name) + " is not in the inventory");
  }

  std::filesystem::path directory_;
  std::vector<CaptureRecord> records_;
  void* host_data_{};
  std::size_t bytes_{};
};

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "create vision event"); }
  ~CudaEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  [[nodiscard]] cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{};
};

}  // namespace

int run(vision_engine::Model selected, const std::string& artifact_path, const std::string& pixel_values_path,
        const std::string& position_ids_path,
        std::uint32_t soft_token_count,
        const std::string& capture_directory) {
  if (soft_token_count == 0 ||
      soft_token_count > model::kVisionMaxSoftTokenCount) {
    fail("vision runner", "soft-token count must be in 1..1120");
  }

  const std::vector<std::uint8_t> pixels =
      read_file(pixel_values_path,
                vision_engine::prepared_pixel_bytes(
                    vision_engine::padded_patch_rows_for_capacity(
                        model::kVisionMaxSoftTokenCount)),
                "vision pixel input");
  const std::vector<std::uint8_t> positions =
      read_file(position_ids_path,
                vision_engine::prepared_position_bytes(
                    vision_engine::padded_patch_rows_for_capacity(
                        model::kVisionMaxSoftTokenCount)),
                "vision position input");
  vision_engine::validate_prepared_image_bytes(
      pixels.data(), pixels.size(), positions.data(), positions.size(),
      soft_token_count);
  const std::uint32_t padded_patch_rows = static_cast<std::uint32_t>(
      pixels.size() / (model::kVisionPatchWidth * sizeof(float)));

  const auto output_width = vision_engine::output_width(selected);
  gewell::component::File file(artifact_path, selected == vision_engine::Model::gemma4_26b_a4b
      ? gemma4_26b_a4b::vision_tensor_specs() : model::component_specs(model::Component::vision));
  console::section("Vision prefill diagnostic");
  DeviceAllocation weights(file.device_bytes());
  auto* destination = static_cast<std::uint8_t*>(weights.data());
  check_cuda(cudaMemset(destination, 0, weights.size()), "zero vision padding");
  std::size_t offset = 0;
  for (const auto& tensor : file.tensors()) {
    check_cuda(cudaMemcpy(destination + offset, tensor.data, tensor.bytes, cudaMemcpyHostToDevice),
               "copy vision weights to device");
    offset += model::align_up(tensor.bytes, model::kStorageAlignment);
  }

  executor::VisionExecutor tower(selected,
      {static_cast<const BFloat16*>(weights.data()), weights.size()},
      soft_token_count);
  DeviceAllocation device_pixels(pixels.size());
  DeviceAllocation device_positions(positions.size());
  DeviceAllocation output(static_cast<std::size_t>(soft_token_count) *
                          output_width * sizeof(BFloat16));
  DeviceAllocation scratch(tower.scratch_bytes());
  check_cuda(cudaMemcpy(device_pixels.data(), pixels.data(), pixels.size(),
                        cudaMemcpyHostToDevice),
             "copy vision pixels to device");
  check_cuda(cudaMemcpy(device_positions.data(), positions.data(),
                        positions.size(), cudaMemcpyHostToDevice),
             "copy vision positions to device");

  vision_engine::PrefillRequest request{
      {static_cast<const float*>(device_pixels.data()),
       static_cast<const std::int32_t*>(device_positions.data()),
       padded_patch_rows, soft_token_count},
      output.data(), soft_token_count};
  CapturePlan captures(capture_directory, padded_patch_rows,
                       soft_token_count, output_width);
  CudaEvent begin;
  CudaEvent end;
  check_cuda(cudaEventRecord(begin.get()), "record vision start event");
  tower.run(request, scratch.data(), scratch.size(), nullptr, &captures);
  check_cuda(cudaEventRecord(end.get()), "record vision end event");
  check_cuda(cudaEventSynchronize(end.get()), "synchronize vision execution");
  float milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&milliseconds, begin.get(), end.get()),
             "measure vision execution");
  captures.write();

  console::section("Vision prefill results");
  console::field("vision_prefill", "ok");
  console::field("soft_tokens", soft_token_count);
  console::field("valid_patch_rows", tower.patch_rows());
  console::field("vision_weight_bytes", weights.size());
  console::field("vision_scratch_bytes", scratch.size());
  console::field("capture_host_bytes", captures.bytes());
  console::field("vision_weights", artifact_path);
  console::field("vision_capture_run_milliseconds", milliseconds);
  console::section("Capture files");
  console::field("capture_directory", capture_directory);
  return 0;
}

}  // namespace gewell::vision_runner
