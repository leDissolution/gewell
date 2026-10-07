// Included inside mtp_attention's implementation namespace, after its bounded
// batch descriptors. Decode, verification and frozen assistant queries share
// this E4M3 QK/PV kernel; cache bytes and commit policy are unchanged.

__device__ __forceinline__ unsigned fp8_word(const unsigned char* p) {
  return *reinterpret_cast<const unsigned*>(p);
}

__device__ __forceinline__ void fp8_mma(float (&c)[4],
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

using detail::fp8_scale;
using detail::fp8_quantize;
using detail::fp8_pack_four;

// Padded rows avoid the QK bank aliasing of a power-of-two stride. Swizzle
// whole V words so eight-element vector loaders do not serialize the transpose.
__device__ __forceinline__ unsigned fp8_v_offset(unsigned d, unsigned t) {
  return d * 36 + (t ^ ((d / 32 % 8) * 4));
}

constexpr unsigned fp8_shared_bytes(bool global, bool frozen) {
  const unsigned d = global ? 512 : 256;
  const unsigned queries = global && !frozen ? 32 : 16;
  return (queries + 32) * (d + 16) + d * 36;
}

template<unsigned QueryHeads, bool Global, bool Paged, bool Frozen>
__global__ __launch_bounds__(kThreads, Global && !Frozen ? 1 : 2)
void fp8_partial_attention(const __grid_constant__ AttentionBatch batch,
                            const BF16* k_norm) {
  constexpr unsigned D = Global ? 512 : 256;
  constexpr unsigned Heads = Global ? QueryHeads / 8 : QueryHeads / 2, Gqa = QueryHeads / Heads;
  constexpr unsigned Queries = Global && !Frozen ? 32 : 16;
  constexpr unsigned GroupRows = Queries / Gqa, ColumnWarps = Queries == 32 ? 4 : 8;
  constexpr unsigned QStride = D + 16;
  extern __shared__ __align__(16) unsigned char workspace[];
  auto* q8 = workspace;
  auto* k8 = q8 + Queries * QStride;
  auto* v8 = k8 + 32 * QStride;
  __shared__ __align__(16) unsigned char p8[Queries * 32];
  __shared__ float qs[Queries], ks[32], vs[32], ps[Queries], scores[Queries * 32];
  __shared__ float maximum[Queries], denominator[Queries], rescale[Queries];
  __shared__ unsigned bad_key[32], bad_query[Queries];
  unsigned index = 0;
  while (blockIdx.y >= batch.tiles[index].end_blocks) ++index;
  const auto& tile = batch.tiles[index];
  const auto& input = tile.input;
  const auto block = blockIdx.y - (index ? batch.tiles[index - 1].end_blocks : 0);
  const unsigned group = block % tile.groups, split = block / tile.groups;
  const unsigned first_row = tile.first + group * GroupRows;
  const unsigned valid_rows = min(GroupRows, input.rows - first_row);
  const unsigned last = Frozen ? input.base_position - 1 : input.local_image
      ? input.base_position + input.rows - 1 : input.base_position + first_row + valid_rows - 1;
  const unsigned first_position = Frozen ? last : input.base_position + first_row;
  const unsigned first = !Global && first_position >= 1023 ? first_position - 1023 : 0;
  const unsigned head = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;

  // Each Q/K/V vector has its own scale. In particular, a future value may
  // never change the quantization scale of a visible value.
  for (unsigned q = warp; q < Queries; q += 8) {
    const bool valid = q / Gqa < valid_rows;
    float values[D / 32], amax = 0;
    bool bad = false;
    for (unsigned i = 0; i < D / 32; ++i) {
      const float x = valid ? float(input.query[
          (std::size_t(head * Gqa + q % Gqa) * input.rows + first_row + q / Gqa) * D + lane + i * 32]) : 0;
      bad |= !isfinite(x);
      values[i] = isfinite(x) ? x : 0;
      amax = fmaxf(amax, fabsf(values[i]));
    }
    amax = __shfl_sync(0xffffffffU, warp_max(amax), 0);
    const float scale = fp8_scale(amax);
    const float inverse = 1.0F / scale;
    const auto any_bad = __any_sync(0xffffffffU, bad);
    if (!lane) { qs[q] = scale; maximum[q] = -CUDART_INF_F; denominator[q] = 0; bad_query[q] = any_bad; }
    for (unsigned i = 0; i < D / 32; ++i)
      q8[q * QStride + lane + i * 32] = fp8_quantize(values[i], scale, inverse);
  }
  __syncthreads();
  float numerator[D / (ColumnWarps * 8)][4]{};
  uint4 norms[D / 256];
  if constexpr (Global) {
#pragma unroll
    for (unsigned i = 0; i < D / 256; ++i) norms[i] = load_eight(k_norm + lane * 8 + i * 256);
  }
  const float norm_maximum = Global ? fp8_cache::norm_bound(k_norm) : 0;
  for (unsigned begin = first + split * 32; begin <= last; begin += tile.splits * 32) {
    for (unsigned t = warp; t < 32; t += 8) {
      const unsigned position = begin + t;
      const bool valid = position <= last;
      float keys[D / 32], values[D / 32], kmax = 0, vmax = 0;
      bool bad = false;
      const BF16* record = nullptr;
      if constexpr (Global)
        if (valid && position < input.base_position)
          record = compact_record<Paged>(input.cache, head, position);
      if (valid && position < input.base_position && input.cache.format == kv_cache::Format::fp8) {
        float key_scale, value_scale;
        fp8_cache::Scales scales{};
        const unsigned char *local_k = nullptr, *local_v = nullptr;
        if constexpr (Global) {
          scales = fp8_cache::scales(record, norm_maximum);
          key_scale = scales.key; value_scale = scales.value;
        } else {
          const auto offset = std::size_t(head) * 1024 + position % 1024;
          local_k = reinterpret_cast<const unsigned char*>(kv_storage::row(input.cache.key, offset, D, input.cache.format));
          local_v = reinterpret_cast<const unsigned char*>(kv_storage::row(input.cache.value, offset, D, input.cache.format));
          key_scale = *reinterpret_cast<const float*>(local_k + D);
          value_scale = *reinterpret_cast<const float*>(local_v + D);
        }
        bool invalid = false;
#pragma unroll
        for (unsigned d = lane * 4; d < D; d += 128) {
          uint2 packed;
          if constexpr (Global) packed = fp8_cache::compact_four(record, k_norm, d, scales);
          else packed = {*reinterpret_cast<const unsigned*>(local_k + d),
                         *reinterpret_cast<const unsigned*>(local_v + d)};
          invalid |= !fp8_cache::finite_four(packed.x) || !fp8_cache::finite_four(packed.y);
          *reinterpret_cast<unsigned*>(k8 + t * QStride + d) = packed.x;
#pragma unroll
          for (unsigned j = 0; j < 4; ++j) v8[fp8_v_offset(d + j, t)] = packed.y >> (j * 8);
        }
        const auto any_bad = __any_sync(0xffffffffU, invalid);
        if (!lane) { ks[t] = key_scale; vs[t] = value_scale; bad_key[t] = any_bad; }
        continue;
      }
#pragma unroll
      for (unsigned i = 0; i < D / 256; ++i) {
        const unsigned d = lane * 8 + i * 256;
        uint4 k{}, v{};
        if (valid) {
          const bool staged = !Frozen && position >= input.base_position;
          if (staged) {
            k = load_eight(input.staged_key + (std::size_t(head) * input.rows + position - input.base_position) * D + d);
            v = load_eight(input.staged_value + (std::size_t(position - input.base_position) * Heads + head) * D + d);
          } else if constexpr (Global) {
            v = kv_storage::load_eight(record, 128 + d, 640, input.cache.format, 128);
            if (d < 64 || (d >= 256 && d < 320))
              k = kv_storage::load_eight(record, d < 64 ? d : d - 192, 640, input.cache.format, 128);
          } else {
            const auto offset = std::size_t(head) * 1024 + position % 1024;
            k = kv_storage::load_eight(kv_storage::row(input.cache.key, offset, D, input.cache.format), d, D, input.cache.format);
            v = kv_storage::load_eight(kv_storage::row(input.cache.value, offset, D, input.cache.format), d, D, input.cache.format);
          }
          if constexpr (Global) {
            if (!(d < 64 || (d >= 256 && d < 320))) {
              auto* kp = reinterpret_cast<__nv_bfloat162*>(&k);
              const auto* vp = reinterpret_cast<const __nv_bfloat162*>(&v);
              const auto* np = reinterpret_cast<const __nv_bfloat162*>(&norms[i]);
#pragma unroll
              for (unsigned j = 0; j < 4; ++j) kp[j] = __hmul2(vp[j], np[j]);
            }
          }
        }
        const auto* kp = reinterpret_cast<const BF16*>(&k);
        const auto* vp = reinterpret_cast<const BF16*>(&v);
#pragma unroll
        for (unsigned j = 0; j < 8; ++j) {
          const float key = float(kp[j]), value = float(vp[j]);
          bad |= !isfinite(key) || !isfinite(value);
          keys[i * 8 + j] = isfinite(key) ? key : 0;
          values[i * 8 + j] = isfinite(value) ? value : 0;
          kmax = fmaxf(kmax, fabsf(keys[i * 8 + j]));
          vmax = fmaxf(vmax, fabsf(values[i * 8 + j]));
        }
      }
      const float kscale = fp8_scale(__shfl_sync(0xffffffffU, warp_max(kmax), 0));
      const float vscale = fp8_scale(__shfl_sync(0xffffffffU, warp_max(vmax), 0));
      const float kinverse = 1.0F / kscale, vinverse = 1.0F / vscale;
      const auto any_bad = __any_sync(0xffffffffU, bad);
      if (!lane) { ks[t] = kscale; vs[t] = vscale; bad_key[t] = any_bad; }
#pragma unroll
      for (unsigned i = 0; i < D / 32; i += 4) {
        const unsigned d = lane * 8 + (i / 8) * 256 + i % 8;
        *reinterpret_cast<unsigned*>(k8 + t * QStride + d) = fp8_pack_four(keys + i, kscale, kinverse);
        const auto v = fp8_pack_four(values + i, vscale, vinverse);
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) v8[fp8_v_offset(d + j, t)] = v >> (j * 8);
      }
    }
    __syncthreads();
    const unsigned qr = lane / 4, kc = (lane % 4) * 4;
    if (warp < Queries / 4) {
      const unsigned query_first = (warp / 4) * 16;
      float sum[4]{};
      for (unsigned d = 0; d < D; d += 32) {
        const auto* a = q8 + (query_first + qr) * QStride + d + kc;
        const auto* b = k8 + (warp % 4 * 8 + qr) * QStride + d + kc;
        fp8_mma(sum, fp8_word(a), fp8_word(a + 8 * QStride),
            fp8_word(a + 16), fp8_word(a + 8 * QStride + 16), fp8_word(b), fp8_word(b + 16));
      }
      for (unsigned i = 0; i < 4; ++i) {
        const unsigned q = query_first + qr + (i / 2) * 8, t = warp % 4 * 8 + (lane % 4) * 2 + i % 2;
        scores[q * 32 + t] = sum[i] * qs[q] * ks[t];
      }
    }
    __syncthreads();
    for (unsigned q = warp; q < Queries; q += 8) {
      const unsigned position = Frozen ? last : input.base_position + first_row + q / Gqa;
      const unsigned key_position = begin + lane;
      const bool visible = q / Gqa < valid_rows &&
          (key_position <= position || (input.local_image && key_position <= last)) &&
          (Global || key_position >= position || position - key_position < 1024);
      const float score = visible ? scores[q * 32 + lane] : -CUDART_INF_F;
      const float next = fmaxf(maximum[q], __shfl_sync(0xffffffffU, warp_max(score), 0));
      const float p = visible ? expf(score - next) : 0;
      const float pv = p * vs[lane];
      const float pscale = fp8_scale(__shfl_sync(0xffffffffU, warp_max(pv), 0));
      const float old_scale = maximum[q] == -CUDART_INF_F ? 0 : expf(maximum[q] - next);
      const float sum = __shfl_sync(0xffffffffU, warp_sum(p), 0);
      const auto any_bad = __any_sync(0xffffffffU, visible && bad_key[lane]);
      p8[q * 32 + lane] = fp8_quantize(pv, pscale, 1.0F / pscale);
      if (!lane) {
        maximum[q] = next; denominator[q] = fmaf(denominator[q], old_scale, sum);
        rescale[q] = old_scale; ps[q] = pscale; bad_query[q] |= any_bad;
      }
    }
    __syncthreads();
    const unsigned query_first = warp / ColumnWarps * 16;
    for (unsigned j = 0; j < D / (ColumnWarps * 8); ++j) {
      const auto* a = p8 + (query_first + qr) * 32 + kc;
      const unsigned d = (j * ColumnWarps + warp % ColumnWarps) * 8 + qr;
      float sum[4]{};
      fp8_mma(sum, fp8_word(a), fp8_word(a + 8 * 32), fp8_word(a + 16),
          fp8_word(a + 8 * 32 + 16), fp8_word(v8 + fp8_v_offset(d, kc)),
          fp8_word(v8 + fp8_v_offset(d, kc + 16)));
      for (unsigned i = 0; i < 4; ++i) {
        const unsigned q = query_first + qr + (i / 2) * 8;
        numerator[j][i] = fmaf(numerator[j][i], rescale[q], sum[i] * ps[q]);
      }
    }
    __syncthreads();
  }
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned q = warp / ColumnWarps * 16 + lane / 4 + (i / 2) * 8;
    if (q / Gqa >= valid_rows) continue;
    const unsigned row = first_row + q / Gqa - tile.first;
    const auto result = (std::size_t(row) * QueryHeads + head * Gqa + q % Gqa) * tile.splits + split;
    for (unsigned j = 0; j < D / (ColumnWarps * 8); ++j) {
      const unsigned d = (j * ColumnWarps + warp % ColumnWarps) * 8 + (lane % 4) * 2 + i % 2;
      tile.partial[result * D + d] = bad_query[q] ? CUDART_NAN_F : numerator[j][i];
    }
    if (warp % ColumnWarps == 0 && lane % 4 == 0 && i % 2 == 0) {
      tile.maxima[result] = maximum[q]; tile.denominators[result] = denominator[q];
    }
  }
}

template<unsigned QueryHeads, bool Global, bool Paged, bool Frozen>
void launch_fp8_attention(const AttentionBatch& batch, unsigned count,
                           const BF16* k_norm, cudaStream_t stream) {
  const auto& last = batch.tiles[count - 1];
  constexpr auto shared = fp8_shared_bytes(Global, Frozen);
  if constexpr (Global && !Frozen) {
    static const bool configured = [] {
      const auto status = cudaFuncSetAttribute(fp8_partial_attention<QueryHeads, Global, Paged, Frozen>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
      if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
      return true;
    }();
    (void)configured;
  }
  fp8_partial_attention<QueryHeads, Global, Paged, Frozen>
      <<<dim3(Global ? QueryHeads / 8 : QueryHeads / 2, last.end_blocks), kThreads, shared, stream>>>(batch, k_norm);
  check_launch();
  batched_finalize_attention<Global ? 512 : 256, QueryHeads>
      <<<dim3(QueryHeads, last.end_rows), kThreads, 0, stream>>>(batch);
  check_launch();
}
