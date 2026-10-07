#include "kv_write.cuh"
#include "kv_storage.cuh"
#include "cuda_memory.cuh"
#include "gewell/compact_global_cache.h"
#include "gewell/models/gemma4/26b_a4b/model.h"
#include <algorithm>
#include <stdexcept>

namespace gewell::gemma4_26b_a4b::sm120 {
namespace {
using B = __nv_bfloat16;
template <bool Global>
__device__ void write_row(const B* key, const B* value, kv_cache::DeviceView cache,
                         unsigned position, unsigned token, unsigned head) {
  constexpr unsigned heads = Global ? 2 : 8, width = Global ? 512 : 256;
  unsigned absolute = position + token;
  key += (std::size_t(token) * heads + head) * width;
  value += (std::size_t(token) * heads + head) * width;
  if constexpr (Global) {
    auto* record = compact_global_cache::paged_row(cache.page_pool, cache.page_offsets,
        cache.page_tokens, cache.layer_offset_elements, head, absolute, cache.format);
    if (cache.format == kv_cache::Format::fp8) {
      kv_storage::store_compact(record,key,value);
    } else {
      for (unsigned d = threadIdx.x; d < 640; d += blockDim.x)
        record[d] = d < 128 ? key[d < 64 ? d : d + 192] : value[d - 128];
    }
  } else {
    auto index = std::size_t(head) * cache.capacity + absolute % cache.capacity;
    auto* k = kv_storage::row(cache.key,index,width,cache.format);
    auto* v = kv_storage::row(cache.value,index,width,cache.format);
    if (cache.format == kv_cache::Format::fp8) {
      kv_storage::store_separate<width>(k,v,key,value);
    } else {
      k[threadIdx.x] = key[threadIdx.x];
      v[threadIdx.x] = value[threadIdx.x];
    }
  }
}
template <bool Global>
__global__ void write_rows(const B* key, const B* value, kv_cache::DeviceView cache,
                          unsigned position, unsigned first) {
  write_row<Global>(key,value,cache,position,first+blockIdx.y,blockIdx.x);
}

constexpr unsigned kBatchSize=32;
struct WriteTile {
  CacheWriteInput input;
  unsigned first, end;
};
struct WriteBatch { WriteTile tiles[kBatchSize]; };
template <bool Global>
__global__ void write_batch_rows(const __grid_constant__ WriteBatch batch) {
  unsigned index=0;
  while (blockIdx.x>=batch.tiles[index].end) ++index;
  const auto& tile=batch.tiles[index];
  const unsigned token=tile.first+blockIdx.x-(index?batch.tiles[index-1].end:0);
  write_row<Global>(tile.input.key,tile.input.value,tile.input.cache,
      tile.input.position,token,blockIdx.y);
}

void validate_write(const CacheWriteInput& input, bool global) {
  if (!input.key || !input.value || input.rows>4096 ||
      std::uint64_t(input.position)+input.rows>kMaxPositions)
    throw std::invalid_argument("26B cache write: invalid input rows or position");
  validate_cache_view(input.cache,global,input.position+input.rows);
}
}  // namespace

void validate_cache_view(const kv_cache::DeviceView& cache, bool global, unsigned end) {
  if (end > kMaxPositions ||
      (cache.format != kv_cache::Format::bf16 && cache.format != kv_cache::Format::fp8))
    throw std::invalid_argument("26B cache: invalid extent or format");
  if (global && end) {
    const auto layer_words = std::size_t(kGlobalKvHeadCount) * 256 *
        kv_cache::row_words(640, cache.format, 2);
    if (!cache.page_pool || !cache.page_offsets || cache.page_tokens != 256 ||
        (std::uint64_t(end) + 255) / 256 > cache.page_count ||
        cache.layer_offset_elements > cache.page_stride_elements ||
        layer_words > cache.page_stride_elements - cache.layer_offset_elements)
      throw std::invalid_argument("26B cache: global pages do not cover prefix");
  } else if (!global && (!cache.key || !cache.value || cache.capacity != 1024))
    throw std::invalid_argument("26B cache: invalid local ring");
}

void write_cache(const B* key, const B* value, kv_cache::DeviceView cache,
                 bool global, unsigned position, unsigned rows, cudaStream_t stream) {
  if (!rows) return;
  validate_write({key,value,cache,position,rows},global);
  if (global) {
    write_rows<true><<<dim3(2,rows),256,0,stream>>>(key,value,cache,position,0);
  } else {
    unsigned retained = std::min(rows,1024u);
    write_rows<false><<<dim3(8,retained),256,0,stream>>>(key,value,cache,position,rows-retained);
  }
  cuda_detail::check_cuda(cudaGetLastError(),"26B cache write launch");
}

void write_cache_batch(const std::vector<CacheWriteInput>& inputs, bool global, cudaStream_t stream) {
  for (const auto& input:inputs) if (input.rows) validate_write(input,global);
  WriteBatch batch{};
  unsigned count=0, rows=0;
  const auto flush=[&] {
    if (!count) return;
    if (global) write_batch_rows<true><<<dim3(rows,2),256,0,stream>>>(batch);
    else write_batch_rows<false><<<dim3(rows,8),256,0,stream>>>(batch);
    cuda_detail::check_cuda(cudaGetLastError(),"26B batched cache write launch");
    count=rows=0;
  };
  for (const auto& input:inputs) {
    if (!input.rows) continue;
    const unsigned retained=global?input.rows:std::min(input.rows,1024u);
    rows+=retained;
    batch.tiles[count++]={input,input.rows-retained,rows};
    if (count==kBatchSize) flush();
  }
  flush();
}
}  // namespace gewell::gemma4_26b_a4b::sm120
