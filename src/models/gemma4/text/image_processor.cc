#include "gewell/models/gemma4/image_processor.h"
#include "gewell/vision_engine.h"

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <jpeglib.h>
#include <png.h>

namespace gewell::gemma4 {
namespace {

constexpr std::uint32_t kPatchSize = 16;
constexpr std::uint32_t kPoolSize = 3;

[[noreturn]] void invalid(const char* message) {
  throw std::invalid_argument(message);
}

void validate_size(std::uint32_t width, std::uint32_t height) {
  if (!width || !height || width > kImageMaxDimension ||
      height > kImageMaxDimension ||
      static_cast<std::size_t>(width) * height > kImageMaxPixels) {
    invalid("image exceeds the 8192-pixel side or 16-Mpixel decoded limit");
  }
}

int base64_digit(unsigned char value) {
  if (value >= 'A' && value <= 'Z') return value - 'A';
  if (value >= 'a' && value <= 'z') return value - 'a' + 26;
  if (value >= '0' && value <= '9') return value - '0' + 52;
  if (value == '+') return 62;
  if (value == '/') return 63;
  return -1;
}

std::vector<std::uint8_t> decode_base64(std::string_view data) {
  if (data.empty() || data.size() % 4 != 0) invalid("invalid image base64");
  std::vector<std::uint8_t> bytes;
  bytes.reserve(data.size() / 4 * 3);
  for (std::size_t i = 0; i < data.size(); i += 4) {
    const int a = base64_digit(data[i]);
    const int b = base64_digit(data[i + 1]);
    const int c = base64_digit(data[i + 2]);
    const int d = base64_digit(data[i + 3]);
    const bool last = i + 4 == data.size();
    if (a < 0 || b < 0 ||
        (c < 0 && !(last && data[i + 2] == '=' && data[i + 3] == '=' && (b & 15) == 0)) ||
        (d < 0 && !(last && data[i + 3] == '=' &&
                    (c < 0 || (c & 3) == 0)))) {
      invalid("invalid image base64");
    }
    bytes.push_back(static_cast<std::uint8_t>((a << 2) | (b >> 4)));
    if (c >= 0) bytes.push_back(static_cast<std::uint8_t>((b << 4) | (c >> 2)));
    if (d >= 0) bytes.push_back(static_cast<std::uint8_t>((c << 6) | d));
  }
  return bytes;
}

struct PngState {
  png_structp png{};
  png_infop info{};
  const std::vector<std::uint8_t>* encoded{};
  std::size_t offset{};
  RgbImage image;
  std::vector<std::uint8_t> raw;
  std::vector<png_bytep> rows;
  ~PngState() { if (png) png_destroy_read_struct(&png, &info, nullptr); }
};

void png_read_memory(png_structp png, png_bytep output, png_size_t size) {
  auto& state = *static_cast<PngState*>(png_get_io_ptr(png));
  if (size > state.encoded->size() - state.offset) png_error(png, "truncated image");
  std::memcpy(output, state.encoded->data() + state.offset, size);
  state.offset += size;
}

void png_error_quiet(png_structp png, const char*) { png_longjmp(png, 1); }
void png_warning_quiet(png_structp, const char*) {}

RgbImage decode_png(const std::vector<std::uint8_t>& encoded) {
  // Heap ownership keeps C longjmp error recovery outside C++ object lifetimes.
  const auto state = std::make_unique<PngState>();
  state->encoded = &encoded;
  state->png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                     png_error_quiet, png_warning_quiet);
  if (!state->png) throw std::bad_alloc();
  state->info = png_create_info_struct(state->png);
  if (!state->info) throw std::bad_alloc();
  if (setjmp(png_jmpbuf(state->png))) invalid("invalid or truncated PNG image");
  png_set_read_fn(state->png, state.get(), png_read_memory);
  png_set_user_limits(state->png, kImageMaxDimension, kImageMaxDimension);
  png_set_chunk_malloc_max(state->png, 1024 * 1024);
  png_set_chunk_cache_max(state->png, 32);
  png_set_crc_action(state->png, PNG_CRC_ERROR_QUIT, PNG_CRC_ERROR_QUIT);
  png_read_info(state->png, state->info);
  state->image.width = png_get_image_width(state->png, state->info);
  state->image.height = png_get_image_height(state->png, state->info);
  validate_size(state->image.width, state->image.height);
  const int color = png_get_color_type(state->png, state->info);
  const int depth = png_get_bit_depth(state->png, state->info);
  const bool gray16 = color == PNG_COLOR_TYPE_GRAY && depth == 16;
  if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(state->png);
  if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(state->png);
  // Pillow RGB conversion drops alpha, including transparent palette colors.
  png_set_strip_alpha(state->png);
  if (depth == 16 && !gray16) png_set_strip_16(state->png);
  if ((color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA) && !gray16) {
    png_set_gray_to_rgb(state->png);
  }
  png_set_interlace_handling(state->png);
  png_read_update_info(state->png, state->info);
  const std::size_t row_bytes = png_get_rowbytes(state->png, state->info);
  if (row_bytes != state->image.width * (gray16 ? 2U : 3U)) invalid("unsupported PNG channel layout");
  state->raw.resize(row_bytes * state->image.height);
  state->rows.resize(state->image.height);
  for (std::size_t y = 0; y < state->rows.size(); ++y) state->rows[y] = state->raw.data() + y * row_bytes;
  png_read_image(state->png, state->rows.data());
  png_read_end(state->png, nullptr);
  if (gray16) {
    // Pillow loads 16-bit grayscale as an integer image, then clamps to RGB8.
    state->image.pixels.resize(static_cast<std::size_t>(state->image.width) * state->image.height * 3);
    for (std::size_t i = 0; i < state->raw.size() / 2; ++i) {
      const auto value = static_cast<std::uint8_t>(std::min(255U,
          (static_cast<unsigned>(state->raw[2 * i]) << 8) | state->raw[2 * i + 1]));
      std::fill_n(state->image.pixels.data() + 3 * i, 3, value);
    }
  } else {
    state->image.pixels = std::move(state->raw);
  }
  return std::move(state->image);
}

struct JpegError {
  jpeg_error_mgr base;
  std::jmp_buf jump;
};

struct JpegState {
  jpeg_decompress_struct decoder{};
  JpegError error{};
  bool created{};
  RgbImage image;
  std::vector<std::uint8_t> row;
  ~JpegState() { if (created) jpeg_destroy_decompress(&decoder); }
};

void jpeg_error_quiet(j_common_ptr common) {
  auto* error = reinterpret_cast<JpegError*>(common->err);
  std::longjmp(error->jump, 1);
}
void jpeg_message_quiet(j_common_ptr common, int level) {
  // libjpeg normally fabricates an EOI for truncated files; reject that input.
  if (level < 0) jpeg_error_quiet(common);
}

RgbImage decode_jpeg(const std::vector<std::uint8_t>& encoded) {
  const auto state = std::make_unique<JpegState>();
  auto& decoder = state->decoder;
  decoder.err = jpeg_std_error(&state->error.base);
  state->error.base.error_exit = jpeg_error_quiet;
  state->error.base.emit_message = jpeg_message_quiet;
  if (setjmp(state->error.jump)) invalid("invalid or truncated JPEG image");
  jpeg_create_decompress(&decoder);
  state->created = true;
  decoder.mem->max_memory_to_use = 128 * 1024 * 1024;
  jpeg_mem_src(&decoder, encoded.data(), encoded.size());
  jpeg_read_header(&decoder, TRUE);
  validate_size(decoder.image_width, decoder.image_height);
  const bool cmyk = decoder.jpeg_color_space == JCS_CMYK || decoder.jpeg_color_space == JCS_YCCK;
  decoder.out_color_space = cmyk ? JCS_CMYK : JCS_RGB;
  // These are Pillow's normal (non-draft) libjpeg decoder settings.
  decoder.dct_method = JDCT_ISLOW;
  decoder.do_fancy_upsampling = TRUE;
  jpeg_start_decompress(&decoder);
  state->image.width = decoder.output_width;
  state->image.height = decoder.output_height;
  state->image.pixels.resize(static_cast<std::size_t>(decoder.output_width) * decoder.output_height * 3);
  state->row.resize(static_cast<std::size_t>(decoder.output_width) * decoder.output_components);
  while (decoder.output_scanline < decoder.output_height) {
    auto* output = state->image.pixels.data() + static_cast<std::size_t>(decoder.output_scanline) * decoder.output_width * 3;
    JSAMPROW row = cmyk ? state->row.data() : output;
    if (jpeg_read_scanlines(&decoder, &row, 1) != 1) invalid("truncated JPEG image");
    if (cmyk) {
      // Pillow's JPEG loader assumes Adobe inverted CMYK, then converts to RGB.
      for (std::size_t x = 0; x < decoder.output_width; ++x) {
        for (std::size_t c = 0; c < 3; ++c) {
          const unsigned product = state->row[4 * x + c] * state->row[4 * x + 3] + 128;
          output[3 * x + c] = static_cast<std::uint8_t>((product + (product >> 8)) >> 8);
        }
      }
    }
  }
  jpeg_finish_decompress(&decoder);
  return std::move(state->image);
}

std::pair<std::uint32_t, std::uint32_t> resized_shape(
    const RgbImage& image, std::uint32_t max_soft_tokens) {
  constexpr std::uint32_t multiple = kPatchSize * kPoolSize;
  const double factor = std::sqrt(static_cast<double>(max_soft_tokens * multiple * multiple) /
                                  (static_cast<double>(image.width) * image.height));
  auto width = static_cast<std::uint32_t>(std::floor(factor * image.width / multiple)) * multiple;
  auto height = static_cast<std::uint32_t>(std::floor(factor * image.height / multiple)) * multiple;
  if (!width) {
    width = multiple;
    height = std::min(image.height / image.width, max_soft_tokens) * multiple;
  } else if (!height) {
    height = multiple;
    width = std::min(image.width / image.height, max_soft_tokens) * multiple;
  }
  return {width, height};
}

// Integer taps for output position x are values[x*stride, x*stride+count[x]),
// applied to inputs starting at first[x].
struct ResampleWeights {
  unsigned precision{};
  std::size_t stride{};
  std::vector<std::uint32_t> first, count;
  std::vector<std::int16_t> values;
};

double cubic(double x) {
  x = std::abs(x);
  if (x < 1.0) return ((1.5 * x - 2.5) * x) * x + 1.0;
  if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
  return 0.0;
}

// Matches PyTorch 2.11's uint8 antialiased bicubic coefficients and rounding:
// aten/src/ATen/native/cpu/UpSampleKernel.cpp, HelperInterpCubic. The separable
// passes clamp and round to uint8 separately. See vendor/pytorch-resize-LICENSE.
ResampleWeights resample_weights(std::uint32_t input, std::uint32_t output) {
  ResampleWeights weights;
  weights.first.resize(output);
  weights.count.resize(output);
  std::vector<std::vector<double>> floating(output);
  const double scale = static_cast<double>(input) / output;
  const double support = 2.0 * std::max(1.0, scale);
  const double inverse_scale = scale >= 1.0 ? 1.0 / scale : 1.0;
  double maximum = 0.0;
  for (std::uint32_t x = 0; x < output; ++x) {
    const double center = scale * (x + 0.5);
    const auto first = std::max(0, static_cast<int>(center - support + 0.5));
    const auto end = std::min(static_cast<int>(input), static_cast<int>(center + support + 0.5));
    weights.first[x] = static_cast<std::uint32_t>(first);
    weights.count[x] = static_cast<std::uint32_t>(end - first);
    weights.stride = std::max<std::size_t>(weights.stride, end - first);
    floating[x].resize(end - first);
    double total = 0.0;
    for (int i = first; i < end; ++i) {
      total += floating[x][i - first] = cubic((i - center + 0.5) * inverse_scale);
    }
    for (double& value : floating[x]) {
      value /= total;
      maximum = std::max(maximum, value);
    }
  }
  while (weights.precision < 22 &&
         static_cast<int>(0.5 + maximum * (1 << (weights.precision + 1))) < (1 << 15)) {
    ++weights.precision;
  }
  weights.values.resize(output * weights.stride);
  for (std::uint32_t x = 0; x < output; ++x) {
    for (std::size_t i = 0; i < floating[x].size(); ++i) {
      const double value = floating[x][i] * (1 << weights.precision);
      weights.values[x * weights.stride + i] = static_cast<std::int16_t>(value < 0 ? value - 0.5 : value + 0.5);
    }
  }
  return weights;
}

std::uint8_t resampled(int sum, unsigned precision) {
  return static_cast<std::uint8_t>(std::clamp(sum >> precision, 0, 255));
}

// Integer sums are order-independent, so the loop structure is free to change
// without altering any output byte.
RgbImage resize(RgbImage image, std::uint32_t width, std::uint32_t height) {
  if (image.width != width) {
    const auto weights = resample_weights(image.width, width);
    const int half = 1 << (weights.precision - 1);
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * image.height * 3);
    for (std::uint32_t y = 0; y < image.height; ++y) {
      const auto* row = image.pixels.data() + static_cast<std::size_t>(y) * image.width * 3;
      auto* output = pixels.data() + static_cast<std::size_t>(y) * width * 3;
      for (std::uint32_t x = 0; x < width; ++x) {
        const auto* source = row + static_cast<std::size_t>(weights.first[x]) * 3;
        const auto* taps = weights.values.data() + x * weights.stride;
        int red = half, green = half, blue = half;
        for (std::uint32_t i = 0; i < weights.count[x]; ++i) {
          red += source[3 * i] * taps[i];
          green += source[3 * i + 1] * taps[i];
          blue += source[3 * i + 2] * taps[i];
        }
        output[3 * x] = resampled(red, weights.precision);
        output[3 * x + 1] = resampled(green, weights.precision);
        output[3 * x + 2] = resampled(blue, weights.precision);
      }
    }
    image.width = width;
    image.pixels = std::move(pixels);
  }
  if (image.height != height) {
    const auto weights = resample_weights(image.height, height);
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3;
    std::vector<std::uint8_t> pixels(row_bytes * height);
    std::vector<int> sums(row_bytes);
    for (std::uint32_t y = 0; y < height; ++y) {
      std::fill(sums.begin(), sums.end(), 1 << (weights.precision - 1));
      for (std::uint32_t i = 0; i < weights.count[y]; ++i) {
        const auto* source = image.pixels.data() + (weights.first[y] + i) * row_bytes;
        const int tap = weights.values[y * weights.stride + i];
        for (std::size_t x = 0; x < row_bytes; ++x) sums[x] += source[x] * tap;
      }
      auto* output = pixels.data() + y * row_bytes;
      for (std::size_t x = 0; x < row_bytes; ++x) output[x] = resampled(sums[x], weights.precision);
    }
    image.height = height;
    image.pixels = std::move(pixels);
  }
  return image;
}

void write_u32(std::uint8_t* bytes, std::uint32_t value) {
  bytes[0] = static_cast<std::uint8_t>(value);
  bytes[1] = static_cast<std::uint8_t>(value >> 8);
  bytes[2] = static_cast<std::uint8_t>(value >> 16);
  bytes[3] = static_cast<std::uint8_t>(value >> 24);
}

PreparedImage patchify(const RgbImage& image, std::uint32_t max_soft_tokens) {
  PreparedImage prepared;
  prepared.padded_patch_rows = max_soft_tokens * kPoolSize * kPoolSize;
  const std::uint32_t patch_width = image.width / kPatchSize;
  const std::uint32_t patch_height = image.height / kPatchSize;
  prepared.soft_token_count = patch_width * patch_height / (kPoolSize * kPoolSize);
  prepared.pixels.resize(static_cast<std::size_t>(prepared.padded_patch_rows) * kPatchSize * kPatchSize * 3);
  prepared.positions.resize(static_cast<std::size_t>(prepared.padded_patch_rows) * 2 * sizeof(std::int32_t), 0xff);
  auto* output = prepared.pixels.data();
  for (std::uint32_t py = 0; py < patch_height; ++py) {
    for (std::uint32_t px = 0; px < patch_width; ++px) {
      const std::size_t row = static_cast<std::size_t>(py) * patch_width + px;
      write_u32(prepared.positions.data() + row * 8, px);
      write_u32(prepared.positions.data() + row * 8 + 4, py);
      for (std::uint32_t y = 0; y < kPatchSize; ++y) {
        const auto* source = image.pixels.data() +
            (static_cast<std::size_t>(py * kPatchSize + y) * image.width + px * kPatchSize) * 3;
        output = std::copy_n(source, kPatchSize * 3, output);
      }
    }
  }
  return prepared;
}

}  // namespace

PreparedImage prepare_image_data_url(std::string_view data_url, std::uint32_t max_soft_tokens) {
  if (!vision_engine::is_supported_soft_token_capacity(max_soft_tokens))
    invalid("max_soft_tokens must be one of 70, 140, 280, 560, 1120");
  if (data_url.size() > kImageMaxDataUrlBytes) invalid("image data URL exceeds 8 MiB");
  constexpr std::string_view png_prefix = "data:image/png;base64,";
  constexpr std::string_view jpeg_prefix = "data:image/jpeg;base64,";
  const bool png = data_url.substr(0, png_prefix.size()) == png_prefix;
  const bool jpeg = data_url.substr(0, jpeg_prefix.size()) == jpeg_prefix;
  if (!png && !jpeg) invalid("image must be a base64 data URL with image/png or image/jpeg MIME type");
  const auto encoded = decode_base64(data_url.substr(png ? png_prefix.size() : jpeg_prefix.size()));
  if (png && (encoded.size() < 8 || png_sig_cmp(encoded.data(), 0, 8))) invalid("image MIME type does not match PNG bytes");
  if (jpeg && (encoded.size() < 2 || encoded[0] != 0xff || encoded[1] != 0xd8)) invalid("image MIME type does not match JPEG bytes");
  if (png) {
    for (std::size_t offset = 8; offset + 12 <= encoded.size();) {
      const auto* chunk = encoded.data() + offset;
      const std::size_t size = (static_cast<std::uint32_t>(chunk[0]) << 24) |
          (static_cast<std::uint32_t>(chunk[1]) << 16) |
          (static_cast<std::uint32_t>(chunk[2]) << 8) | chunk[3];
      if (size > encoded.size() - offset - 12) invalid("truncated PNG chunk");
      if (std::memcmp(chunk + 4, "acTL", 4) == 0) invalid("animated PNG images are unsupported");
      offset += size + 12;
    }
  }
  return prepare_rgb_image(png ? decode_png(encoded) : decode_jpeg(encoded), max_soft_tokens);
}

PreparedImage prepare_rgb_image(RgbImage image, std::uint32_t max_soft_tokens) {
  if (!vision_engine::is_supported_soft_token_capacity(max_soft_tokens))
    invalid("max_soft_tokens must be one of 70, 140, 280, 560, 1120");
  validate_size(image.width, image.height);
  if (image.pixels.size() != std::size_t(image.width) * image.height * 3)
    invalid("RGB image byte count differs from dimensions");
  const auto [width, height] = resized_shape(image, max_soft_tokens);
  return patchify(resize(std::move(image), width, height), max_soft_tokens);
}

}  // namespace gewell::gemma4
