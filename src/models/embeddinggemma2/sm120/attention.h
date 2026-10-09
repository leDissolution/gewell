#pragma once
#include <cuda_bf16.h>

namespace gewell::embeddinggemma2 {
// Unmasked bidirectional vision attention with scale 1. Q/K/V/output are
// token-major [rows,12,64]; no device workspace is used. With device row
// offsets `starts[images+1]`, images are concatenated and attend only within
// themselves; `rows` is then the longest image.
void vision_attention(const __nv_bfloat16* query, const __nv_bfloat16* key,
                      const __nv_bfloat16* value, __nv_bfloat16* output, int rows,
                      const int* starts = nullptr, int images = 1);
}  // namespace gewell::embeddinggemma2
