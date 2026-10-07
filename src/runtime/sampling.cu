#include "gewell/mtp_sampling.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace gewell::mtp_sampling {
namespace {

constexpr unsigned kThreads = 256;
constexpr std::size_t kAlignment = 256;
constexpr float kNormalizationTolerance = 2.0e-4F;

void check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(std::string("MTP sampling: ") + message);
  }
}

std::size_t align_up(std::size_t bytes) {
  return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

struct DistributionSummary {
  std::uint32_t cutoff;
  float mass;
};

struct ScratchPlan {
  std::size_t scores_in;
  std::size_t scores_out;
  std::size_t tokens_in;
  std::size_t tokens_out;
  std::size_t weights;
  std::size_t cumulative;
  std::size_t summary;
  std::size_t greedy_result;
  std::size_t validation_partials;
  std::size_t temporary;
  std::size_t temporary_bytes;
  std::size_t bytes;
};

ScratchPlan make_plan(std::uint32_t vocabulary_size) {
  require(vocabulary_size > 0 && vocabulary_size <= INT_MAX,
          "vocabulary size must be in 1..INT_MAX");
  const std::size_t row_bytes =
      static_cast<std::size_t>(vocabulary_size) * sizeof(float);
  require(row_bytes <= (std::numeric_limits<std::size_t>::max() - 8 * kAlignment) /
                           8,
          "scratch size overflows");
  std::size_t sort_bytes = 0;
  std::size_t scan_bytes = 0;
  std::size_t argmax_bytes = 0;
  check_cuda(cub::DeviceRadixSort::SortPairsDescending(
                 nullptr, sort_bytes, static_cast<float*>(nullptr),
                 static_cast<float*>(nullptr),
                 static_cast<std::uint32_t*>(nullptr),
                 static_cast<std::uint32_t*>(nullptr), vocabulary_size),
             "query MTP radix-sort scratch");
  check_cuda(cub::DeviceScan::InclusiveSum(
                 nullptr, scan_bytes, static_cast<float*>(nullptr),
                 static_cast<float*>(nullptr), vocabulary_size),
             "query MTP scan scratch");
  check_cuda(cub::DeviceReduce::ArgMax(
                 nullptr, argmax_bytes, static_cast<float*>(nullptr),
                 static_cast<cub::KeyValuePair<int, float>*>(nullptr),
                 static_cast<int>(vocabulary_size)),
             "query MTP argmax scratch");
  ScratchPlan plan{};
  std::size_t offset = 0;
  const auto row = [&offset, row_bytes]() {
    const std::size_t result = offset;
    offset = align_up(offset + row_bytes);
    return result;
  };
  plan.scores_in = row();
  plan.scores_out = row();
  plan.tokens_in = row();
  plan.tokens_out = row();
  plan.weights = row();
  plan.cumulative = row();
  plan.summary = offset;
  plan.greedy_result = align_up(offset + sizeof(DistributionSummary));
  plan.validation_partials = align_up(plan.greedy_result + sizeof(cub::KeyValuePair<int, float>));
  const std::size_t validation_blocks = std::min(32U, (vocabulary_size - 1) / kThreads + 1);
  plan.temporary = align_up(plan.validation_partials +
      (std::size_t(kMaxDraftTokens) + 1) * validation_blocks * sizeof(float));
  plan.temporary_bytes = std::max({sort_bytes, scan_bytes, argmax_bytes});
  require(plan.temporary_bytes <=
              std::numeric_limits<std::size_t>::max() - plan.temporary -
                  kAlignment,
          "temporary scratch size overflows");
  plan.bytes = align_up(plan.temporary + plan.temporary_bytes);
  return plan;
}

template <typename T>
T* at(void* scratch, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<unsigned char*>(scratch) + offset);
}

ScratchPlan check_scratch(std::uint32_t vocabulary_size, void* scratch,
                          std::size_t scratch_size, Status* status) {
  require(scratch != nullptr && status != nullptr,
          "scratch and status must be nonnull");
  require(reinterpret_cast<std::uintptr_t>(scratch) % kAlignment == 0,
          "scratch must be 256-byte aligned");
  const ScratchPlan plan = make_plan(vocabulary_size);
  require(scratch_size >= plan.bytes, "scratch allocation is too small");
  return plan;
}

unsigned blocks(std::uint32_t count) {
  return (count - 1) / kThreads + 1;
}

__device__ void set_error(Status* status, Status error) {
  atomicCAS(reinterpret_cast<unsigned int*>(status),
            static_cast<unsigned int>(Status::success),
            static_cast<unsigned int>(error));
}

__device__ bool valid_uniform(float value) {
  return isfinite(value) && value >= 0.0F && value < 1.0F;
}

__global__ void greedy_probabilities_kernel(
    const cub::KeyValuePair<int, float>* best, std::uint32_t vocabulary_size,
    float* probabilities, Status* status) {
  const unsigned token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= vocabulary_size) return;
  if (!isfinite(best->value)) {
    if (token == 0) set_error(status, Status::invalid_distribution);
    probabilities[token] = 0.0F;
    return;
  }
  probabilities[token] = token == static_cast<unsigned>(best->key) ? 1.0F : 0.0F;
}

__global__ void make_scores_kernel(const __nv_bfloat16* logits,
                                   std::uint32_t vocabulary_size,
                                   float temperature, float* scores,
                                   std::uint32_t* tokens, Status* status,
                                   const std::uint32_t* allowed_tokens) {
  const std::uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= vocabulary_size) {
    return;
  }
  const float logit = __bfloat162float(logits[token]);
  float score = logit / temperature;
  if (!isfinite(logit) || !isfinite(score)) {
    set_error(status, Status::invalid_logits);
    score = 0.0F;
  }
  if (allowed_tokens &&
      !(allowed_tokens[token / 32] & (std::uint32_t{1} << (token % 32)))) {
    score = -CUDART_INF_F;
  }
  // Stable radix sorting otherwise distinguishes negative and positive zero.
  scores[token] = score == 0.0F ? 0.0F : score;
  tokens[token] = token;
}

constexpr unsigned kGreedyBatchEntries = 128;
struct GreedyCandidate {
  float score;
  std::uint32_t token;
};
struct GreedyBetter {
  __device__ GreedyCandidate operator()(const GreedyCandidate& left,
                                        const GreedyCandidate& right) const {
    return left.score > right.score ||
                   (left.score == right.score && left.token < right.token)
               ? left
               : right;
  }
};
struct GreedyBatch {
  GreedyDistributionInput inputs[kGreedyBatchEntries];
};

constexpr unsigned kGreedyTile = 4096;

__global__ void initialize_greedy_batch_kernel(
    const __grid_constant__ GreedyBatch batch) {
  if (threadIdx.x == 0) *batch.inputs[blockIdx.x].best_scratch = UINT_MAX;
}

__device__ GreedyCandidate greedy_candidate(
    const GreedyDistributionInput& input, std::uint32_t token) {
  const float logit = __bfloat162float(input.logits[token]);
  const float score = isfinite(logit) ? logit : 0.0F;
  return {score == 0.0F ? 0.0F : score, token};
}

__global__ void greedy_best_batch_kernel(
    const __grid_constant__ GreedyBatch batch,
    std::uint32_t vocabulary_size) {
  using Reduce = cub::BlockReduce<GreedyCandidate, kThreads>;
  __shared__ typename Reduce::TempStorage reduction;
  const auto& input = batch.inputs[blockIdx.y];
  GreedyCandidate best{-CUDART_INF_F, UINT_MAX};
  bool invalid = false;
  const unsigned first = blockIdx.x * kGreedyTile;
  const unsigned end = min(first + kGreedyTile, vocabulary_size);
  for (std::uint32_t token = first + threadIdx.x; token < end;
       token += blockDim.x) {
    invalid |= !isfinite(__bfloat162float(input.logits[token]));
    auto candidate = greedy_candidate(input, token);
    if (input.allowed_tokens &&
        !(input.allowed_tokens[token / 32] &
          (std::uint32_t{1} << (token % 32))))
      candidate.score = -CUDART_INF_F;
    best = GreedyBetter{}(best, candidate);
  }
  best = Reduce(reduction).Reduce(best, GreedyBetter{});
  const bool invalid_row = __syncthreads_or(invalid);
  if (threadIdx.x != 0) return;
  if (invalid_row) set_error(input.status, Status::invalid_logits);
  if (!isfinite(best.score)) return;
  // The winner only improves under the total (score, lower-token-ID) order.
  // Reuse the existing one-word scratch; no per-tile allocation is needed.
  unsigned previous = atomicCAS(input.best_scratch, UINT_MAX, UINT_MAX);
  while (previous == UINT_MAX ||
         GreedyBetter{}(best, greedy_candidate(input, previous)).token == best.token) {
    if (previous == best.token) break;
    const unsigned observed = atomicCAS(input.best_scratch, previous, best.token);
    if (observed == previous) break;
    previous = observed;
  }
}

__global__ void finalize_greedy_batch_kernel(
    const __grid_constant__ GreedyBatch batch) {
  if (threadIdx.x != 0) return;
  const auto& input = batch.inputs[blockIdx.x];
  const unsigned best = *input.best_scratch;
  if (best == UINT_MAX) set_error(input.status, Status::invalid_distribution);
  *input.selected_id = *input.status == Status::success ? best : 0;
}

__global__ void greedy_probabilities_batch_kernel(
    const __grid_constant__ GreedyBatch batch,
    std::uint32_t vocabulary_size) {
  const auto& input = batch.inputs[blockIdx.y];
  const std::uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t best = *input.best_scratch;
  if (input.output_probs && token < vocabulary_size)
    input.output_probs[token] = token == best ? 1.0F : 0.0F;
}

constexpr unsigned kGreedyVerificationBatchEntries = 128;
struct GreedyVerificationBatch {
  GreedyVerificationInput inputs[kGreedyVerificationBatchEntries];
};

__global__ void verify_greedy_sequences_kernel(
    const __grid_constant__ GreedyVerificationBatch batch,
    std::uint32_t vocabulary_size) {
  const auto& input = batch.inputs[blockIdx.x];
  *input.result = {0, 0, input.draft_count};
  if (*input.status != Status::success) return;
  if (!valid_uniform(*input.final_uniform)) {
    set_error(input.status, Status::invalid_uniform);
    return;
  }
  for (std::uint32_t index = 0; index < input.draft_count; ++index) {
    const std::uint32_t draft = input.draft_ids[index];
    if (draft >= vocabulary_size) {
      set_error(input.status, Status::invalid_draft_token);
      return;
    }
    if (!valid_uniform(input.accept_uniforms[index])) {
      set_error(input.status, Status::invalid_uniform);
      return;
    }
    const std::uint32_t target = input.target_ids[index];
    if (target >= vocabulary_size) {
      set_error(input.status, Status::invalid_distribution);
      return;
    }
    if (draft != target) {
      input.output_ids[index] = target;
      input.result->accepted_drafts = index;
      input.result->output_count = index + 1;
      input.result->rejected_index = index;
      return;
    }
    input.output_ids[index] = draft;
  }
  const std::uint32_t bonus = input.target_ids[input.draft_count];
  if (bonus >= vocabulary_size) {
    set_error(input.status, Status::invalid_distribution);
    return;
  }
  input.output_ids[input.draft_count] = bonus;
  *input.result = {input.draft_count, input.draft_count + 1,
                   input.draft_count};
}

__global__ void make_weights_kernel(const float* scores, std::uint32_t count,
                                    float* weights) {
  const std::uint32_t rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank < count) {
    // All-masked rows have no finite maximum and must retain zero mass,
    // rather than manufacturing NaNs from (-inf)-(-inf).
    weights[rank] = isfinite(scores[rank])
                        ? expf(scores[rank] - scores[0]) : 0.0F;
  }
}

__global__ void choose_nucleus_kernel(const float* cumulative,
                                      std::uint32_t retained, float top_p,
                                      DistributionSummary* summary,
                                      Status* status) {
  if (*status != Status::success) {
    return;
  }
  const float total = cumulative[retained - 1];
  if (!(total > 0.0F) || !isfinite(total)) {
    set_error(status, Status::invalid_distribution);
    return;
  }
  const float threshold = top_p * total;
  std::uint32_t low = 0;
  std::uint32_t high = retained;
  while (low < high) {
    const std::uint32_t middle = low + (high - low) / 2;
    if (cumulative[middle] >= threshold) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  const std::uint32_t cutoff = low < retained ? low : retained - 1;
  summary->cutoff = cutoff;
  summary->mass = cumulative[cutoff];
}

__global__ void scatter_probabilities_kernel(
    const std::uint32_t* sorted_tokens, const float* weights,
    std::uint32_t vocabulary_size, const DistributionSummary* summary,
    float* probabilities, const Status* status) {
  const std::uint32_t rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= vocabulary_size || *status != Status::success) {
    return;
  }
  probabilities[sorted_tokens[rank]] =
      rank <= summary->cutoff ? weights[rank] / summary->mass : 0.0F;
}

__global__ void probability_partials_kernel(const float* probabilities,
                                            std::uint32_t vocabulary_size,
                                            float* partials, Status* status) {
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  const float* row = probabilities +
                     static_cast<std::size_t>(blockIdx.y) * vocabulary_size;
  float sum = 0.0F;
  for (std::uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
       token < vocabulary_size; token += blockDim.x * gridDim.x) {
    const float probability = row[token];
    if (!isfinite(probability) || probability < 0.0F ||
        probability > 1.0F + kNormalizationTolerance) {
      set_error(status, Status::invalid_distribution);
    } else {
      sum += probability;
    }
  }
  const float total = Reduce(storage).Sum(sum);
  if (threadIdx.x == 0)
    partials[blockIdx.y * gridDim.x + blockIdx.x] = total;
}

__global__ void validate_probability_totals_kernel(const float* partials,
                                                  unsigned blocks_per_row,
                                                  Status* status) {
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  const float value = threadIdx.x < blocks_per_row
      ? partials[blockIdx.x * blocks_per_row + threadIdx.x] : 0.0F;
  const float total = Reduce(storage).Sum(value);
  if (threadIdx.x == 0 &&
      (!isfinite(total) || fabsf(total - 1.0F) > kNormalizationTolerance)) {
    set_error(status, Status::invalid_distribution);
  }
}

void validate_probabilities(const float* probabilities, unsigned rows,
                            unsigned vocabulary, void* scratch,
                            const ScratchPlan& plan, Status* status,
                            cudaStream_t stream) {
  const unsigned parts = std::min(32U, blocks(vocabulary));
  auto* partials = at<float>(scratch, plan.validation_partials);
  probability_partials_kernel<<<dim3(parts, rows), kThreads, 0, stream>>>(
      probabilities, vocabulary, partials, status);
  validate_probability_totals_kernel<<<rows, kThreads, 0, stream>>>(
      partials, parts, status);
  check_cuda(cudaGetLastError(), "validate MTP probability rows");
}

__device__ std::uint32_t draw_cdf(const float* weights, const float* cumulative,
                                std::uint32_t vocabulary_size, float uniform,
                                Status* status, Status zero_mass_error) {
  if (!valid_uniform(uniform)) {
    set_error(status, Status::invalid_uniform);
    return UINT_MAX;
  }
  const float total = cumulative[vocabulary_size - 1];
  if (!(total > 0.0F) || !isfinite(total)) {
    set_error(status, zero_mass_error);
    return UINT_MAX;
  }
  // Rounding u*total up to total must not select a trailing zero-mass token.
  const float threshold = fminf(uniform * total, nextafterf(total, 0.0F));
  std::uint32_t low = 0;
  std::uint32_t high = vocabulary_size;
  while (low < high) {
    const std::uint32_t middle = low + (high - low) / 2;
    if (cumulative[middle] > threshold) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  if (low == vocabulary_size) {
    set_error(status, Status::invalid_distribution);
    return UINT_MAX;
  }
  if (weights[low] > 0.0F) return low;
  // Parallel prefix reductions can round upward across a zero-weight bucket.
  // Keep the normal binary-search path, but never emit outside the support:
  // advance to the next positive bucket, or the last positive bucket when
  // rounding introduced mass in a trailing zero plateau.
  for (std::uint32_t token = low + 1; token < vocabulary_size; ++token)
    if (weights[token] > 0.0F) return token;
  while (low > 0)
    if (weights[--low] > 0.0F) return low;
  set_error(status, zero_mass_error);
  return UINT_MAX;
}

__global__ void sample_cdf_kernel(const float* weights, const float* cumulative,
                                  std::uint32_t vocabulary_size,
                                  const float* uniform,
                                  std::uint32_t* output_token, Status* status) {
  *output_token = UINT_MAX;
  if (*status == Status::success) {
    *output_token = draw_cdf(weights, cumulative, vocabulary_size, *uniform, status,
                             Status::invalid_distribution);
  }
}

struct ProbabilityCandidate {
  float probability;
  std::uint32_t token;
};

__device__ bool better_candidate(const ProbabilityCandidate& left,
                                 const ProbabilityCandidate& right) {
  return left.probability > right.probability ||
         (left.probability == right.probability && left.token < right.token);
}

struct SelectBetterCandidate {
  __device__ ProbabilityCandidate operator()(
      const ProbabilityCandidate& left,
      const ProbabilityCandidate& right) const {
    return better_candidate(left, right) ? left : right;
  }
};

__device__ float probability_at(const float* row, unsigned index) {
  return row[index];
}
__device__ float probability_at(const TokenProbability* row, unsigned index) {
  return row[index].probability;
}
__device__ unsigned token_at(const float*, unsigned index) { return index; }
__device__ unsigned token_at(const TokenProbability* row, unsigned index) {
  return row[index].token;
}
__device__ float lookup_probability(const float* row, unsigned size,
                                    unsigned token) {
  return token < size ? row[token] : CUDART_NAN_F;
}
__device__ float lookup_probability(const TokenProbability* row, unsigned size,
                                    unsigned token) {
  unsigned low = 0, high = size;
  while (low < high) {
    const unsigned middle = low + (high - low) / 2;
    if (row[middle].token < token) low = middle + 1;
    else high = middle;
  }
  return low < size && row[low].token == token ? row[low].probability : 0.0F;
}

__global__ void gather_capture_rows_kernel(const std::uint16_t* probes,
    unsigned rows, unsigned width, const Result* result, const Status* status,
    std::uint16_t* output) {
  const unsigned count = result->output_count;
  const bool valid = *status == Status::success && count && count <= rows;
  for (unsigned col = threadIdx.x; col < width; col += blockDim.x)
    output[std::size_t(blockIdx.x) * width + col] = valid
        ? probes[(std::size_t(blockIdx.x) * rows + count - 1) * width + col] : 0;
}

template <typename Entry>
__global__ void capture_scores_kernel(const Entry* target, const Entry* draft,
    const std::uint32_t* ids, unsigned row_size, const std::uint32_t* greedy_ids,
    const Status* status, MtpCaptureScores* output) {
  const unsigned step = blockIdx.x, lane = threadIdx.x;
  if (*status != Status::success) {
    if (!lane) output[step] = {};
    return;
  }
  if (greedy_ids) {
    if (!lane) output[step] = {1.0F, ids[step] == greedy_ids[step] ? 1.0F : 0.0F, 0.0F, 1.0F};
    return;
  }
  const auto* q = draft + std::size_t(step) * row_size;
  const auto* p = target + std::size_t(step) * row_size;
  __shared__ float entropy[kThreads], maximum[kThreads];
  float e = 0, m = 0;
  for (unsigned col = lane; col < row_size; col += kThreads) {
    const float value = probability_at(q, col);
    if (value > 0) e -= value * logf(value);
    m = fmaxf(m, value);
  }
  entropy[lane] = e; maximum[lane] = m;
  __syncthreads();
  for (unsigned stride = kThreads / 2; stride; stride /= 2) {
    if (lane < stride) {
      entropy[lane] += entropy[lane + stride];
      maximum[lane] = fmaxf(maximum[lane], maximum[lane + stride]);
    }
    __syncthreads();
  }
  if (!lane) output[step] = {lookup_probability(q, row_size, ids[step]),
      lookup_probability(p, row_size, ids[step]), entropy[0], maximum[0]};
}

template <typename Entry>
__global__ void summarize_logprobs_kernel(
    const Entry* probabilities, const std::uint32_t* selected_ids,
    std::uint32_t vocabulary_size, std::uint32_t top_logprobs,
    TokenLogprobs* output) {
  using Reduce = cub::BlockReduce<ProbabilityCandidate, kThreads>;
  __shared__ typename Reduce::TempStorage reduction;
  __shared__ ProbabilityCandidate winner;

  const std::uint32_t row_index = blockIdx.x;
  const Entry* row = probabilities +
                     static_cast<std::size_t>(row_index) * vocabulary_size;
  TokenLogprobs& result = output[row_index];
  if (threadIdx.x == 0) {
    const std::uint32_t selected = selected_ids[row_index];
    result.logprob = logf(lookup_probability(row, vocabulary_size, selected));
    result.count = 0;
    for (std::uint32_t rank = 0; rank < kMaxTopLogprobs; ++rank) {
      result.top[rank] = {};
    }
  }

  ProbabilityCandidate local[kMaxTopLogprobs];
  for (std::uint32_t rank = 0; rank < top_logprobs; ++rank) {
    local[rank] = {-1.0F, UINT_MAX};
  }
  for (std::uint32_t token = threadIdx.x; token < vocabulary_size;
       token += blockDim.x) {
    const ProbabilityCandidate candidate{probability_at(row, token),
                                          token_at(row, token)};
    if (!(candidate.probability > 0.0F) || top_logprobs == 0 ||
        !better_candidate(candidate, local[top_logprobs - 1])) {
      continue;
    }
    std::uint32_t rank = top_logprobs - 1;
    while (rank != 0 && better_candidate(candidate, local[rank - 1])) {
      local[rank] = local[rank - 1];
      --rank;
    }
    local[rank] = candidate;
  }

  std::uint32_t local_rank = 0;
  for (std::uint32_t rank = 0; rank < top_logprobs; ++rank) {
    const ProbabilityCandidate candidate = local[local_rank];
    const ProbabilityCandidate aggregate =
        Reduce(reduction).Reduce(candidate, SelectBetterCandidate{});
    if (threadIdx.x == 0) {
      winner = aggregate;
    }
    __syncthreads();
    if (!(winner.probability > 0.0F)) {
      break;
    }
    if (threadIdx.x == 0) {
      result.top[rank] = {winner.token, logf(winner.probability)};
      result.count = rank + 1;
    }
    if (candidate.token == winner.token) {
      ++local_rank;
    }
  }
}

__global__ void find_rejection_kernel(
    const float* target_probs, const float* draft_probs,
    const std::uint32_t* draft_ids, std::uint32_t draft_count,
    std::uint32_t vocabulary_size, const float* accept_uniforms,
    const float* final_uniform, std::uint32_t* output_ids, Result* result,
    Status* status) {
  *result = {0, 0, draft_count};
  if (*status != Status::success) {
    return;
  }
  if (!valid_uniform(*final_uniform)) {
    set_error(status, Status::invalid_uniform);
    return;
  }
  for (std::uint32_t index = 0; index < draft_count; ++index) {
    const std::uint32_t draft = draft_ids[index];
    if (draft >= vocabulary_size) {
      set_error(status, Status::invalid_draft_token);
      return;
    }
    const float uniform = accept_uniforms[index];
    if (!valid_uniform(uniform)) {
      set_error(status, Status::invalid_uniform);
      return;
    }
    const std::size_t offset =
        static_cast<std::size_t>(index) * vocabulary_size + draft;
    const float q = draft_probs[offset];
    const float p = target_probs[offset];
    if (!(q > 0.0F)) {
      set_error(status, Status::zero_draft_probability);
      return;
    }
    if (!(p >= q || uniform * q < p)) {
      result->rejected_index = index;
      break;
    }
    output_ids[index] = draft;
    ++result->accepted_drafts;
  }
  result->output_count = result->accepted_drafts + 1;
}

__global__ void make_final_weights_kernel(
    const float* target_probs, const float* draft_probs,
    std::uint32_t draft_count, std::uint32_t vocabulary_size,
    const Result* result, float* weights, const Status* status) {
  const std::uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= vocabulary_size) {
    return;
  }
  if (*status != Status::success) {
    weights[token] = 0.0F;
    return;
  }
  const std::uint32_t row = result->accepted_drafts;
  const std::size_t offset =
      static_cast<std::size_t>(row) * vocabulary_size + token;
  weights[token] = row == draft_count
                       ? target_probs[offset]
                       : fmaxf(target_probs[offset] - draft_probs[offset], 0.0F);
}

__global__ void sample_final_kernel(const float* weights, const float* cumulative,
                                    std::uint32_t vocabulary_size,
                                    std::uint32_t draft_count,
                                    const float* final_uniform,
                                    std::uint32_t* output_ids, Result* result,
                                    Status* status) {
  if (*status != Status::success) {
    result->output_count = 0;
    return;
  }
  const std::uint32_t token = draw_cdf(
      weights, cumulative, vocabulary_size, *final_uniform, status,
      result->accepted_drafts == draft_count ? Status::invalid_distribution
                                             : Status::zero_residual_mass);
  if (*status == Status::success) {
    output_ids[result->accepted_drafts] = token;
  } else {
    result->output_count = 0;
  }
}

void scan(const float* input, float* output, std::uint32_t elements,
          void* scratch, const ScratchPlan& plan, cudaStream_t stream) {
  std::size_t temporary_bytes = plan.temporary_bytes;
  check_cuda(cub::DeviceScan::InclusiveSum(
                 at<void>(scratch, plan.temporary), temporary_bytes, input,
                 output, elements, stream),
             "MTP probability scan");
}

struct SelectionCandidate {
  float score;
  std::uint32_t token;
};

// Each tile keeps its exact top-k. A discarded candidate already has k better
// candidates in its own tile, so it cannot enter the global top-k. Stable sorts
// and contiguous tile order retain lower-token-ID ties through every merge.
template <int Items, bool Final>
__device__ __forceinline__ void compact_candidates_body(
    const __nv_bfloat16* logits, const SelectionCandidate* input,
    unsigned count, float temperature, const std::uint32_t* allowed_tokens,
    unsigned retained, float top_p, SelectionCandidate* candidates,
    TokenProbability* output, Status* status, unsigned tile) {
  using Sort = cub::BlockRadixSort<float, kThreads, Items, std::uint32_t>;
  using TokenSort = cub::BlockRadixSort<std::uint32_t, kThreads, Items, float>;
  using Scan = cub::BlockScan<float, kThreads>;
  __shared__ union {
    typename Sort::TempStorage sort;
    typename TokenSort::TempStorage token_sort;
    typename Scan::TempStorage scan;
  } storage;
  __shared__ float maximum, mass, total;
  __shared__ unsigned cutoff;
  float scores[Items];
  std::uint32_t tokens[Items];
  for (unsigned item = 0; item < Items; ++item) {
    const unsigned index = tile * kThreads * Items + threadIdx.x * Items + item;
    float score = -CUDART_INF_F;
    unsigned token = UINT_MAX;
    if (index < count) {
      if (input) {
        score = input[index].score;
        token = input[index].token;
      } else {
        token = index;
        const float logit = __bfloat162float(logits[index]);
        score = logit / temperature;
        if (!isfinite(logit) || !isfinite(score)) {
          set_error(status, Status::invalid_logits);
          score = 0.0F;
        }
        if (allowed_tokens &&
            !(allowed_tokens[token / 32] & (std::uint32_t{1} << (token % 32))))
          score = -CUDART_INF_F;
      }
    }
    scores[item] = score == 0.0F ? 0.0F : score;
    tokens[item] = token;
  }
  Sort(storage.sort).SortDescending(scores, tokens);
  if constexpr (!Final) {
    for (unsigned item = 0; item < Items; ++item) {
      const unsigned rank = threadIdx.x * Items + item;
      if (rank < retained)
        candidates[std::size_t(tile) * retained + rank] =
            {scores[item], tokens[item]};
    }
  } else {
    if (threadIdx.x == 0) {
      maximum = scores[0];
      cutoff = retained - 1;
    }
    __syncthreads();
    for (unsigned item = 0; item < Items; ++item) {
      const unsigned rank = threadIdx.x * Items + item;
      scores[item] = rank < retained && isfinite(scores[item])
                         ? expf(scores[item] - maximum) : 0.0F;
      if (rank >= retained) tokens[item] = UINT_MAX;
    }
    float cumulative[Items];
    Scan(storage.scan).InclusiveSum(scores, cumulative);
    for (unsigned item = 0; item < Items; ++item)
      if (threadIdx.x * Items + item == retained - 1) total = cumulative[item];
    __syncthreads();
    if (threadIdx.x == 0 && (!(total > 0.0F) || !isfinite(total)))
      set_error(status, Status::invalid_distribution);
    for (unsigned item = 0; item < Items; ++item) {
      const unsigned rank = threadIdx.x * Items + item;
      if (rank < retained && cumulative[item] >= top_p * total)
        atomicMin(&cutoff, rank);
    }
    __syncthreads();
    for (unsigned item = 0; item < Items; ++item)
      if (threadIdx.x * Items + item == cutoff) mass = cumulative[item];
    __syncthreads();
    for (unsigned item = 0; item < Items; ++item)
      scores[item] = threadIdx.x * Items + item <= cutoff && mass > 0.0F
                         ? scores[item] / mass : 0.0F;
    TokenSort(storage.token_sort).Sort(tokens, scores);
    for (unsigned item = 0; item < Items; ++item) {
      const unsigned rank = threadIdx.x * Items + item;
      if (rank < retained) output[rank] = {tokens[item], scores[item]};
    }
  }
}

template <int Items, bool Final>
__global__ void compact_candidates_kernel(
    const __nv_bfloat16* logits, const SelectionCandidate* input,
    unsigned count, float temperature, const std::uint32_t* allowed_tokens,
    unsigned retained, float top_p, SelectionCandidate* candidates,
    TokenProbability* output, Status* status) {
  compact_candidates_body<Items, Final>(logits, input, count, temperature,
      allowed_tokens, retained, top_p, candidates, output, status, blockIdx.x);
}

constexpr unsigned kCompactBatchEntries = 32;
struct CompactCandidateRow {
  const __nv_bfloat16* logits;
  const SelectionCandidate* input;
  SelectionCandidate* next;
  TokenProbability* output;
  const std::uint32_t* allowed;
  Status* status;
  float temperature, top_p;
};
struct CompactCandidateBatch { CompactCandidateRow rows[kCompactBatchEntries]; };

template <int Items, bool Final>
__global__ void compact_candidates_batch_kernel(
    const __grid_constant__ CompactCandidateBatch batch,
    unsigned count, unsigned retained) {
  const auto& row = batch.rows[blockIdx.y];
  compact_candidates_body<Items, Final>(row.logits, row.input, count,
      row.temperature, row.allowed, retained, row.top_p, row.next,
      row.output, row.status, blockIdx.x);
}

using CompactReduce = cub::BlockReduce<float, kThreads>;
using CompactScan = cub::BlockScan<float, kThreads>;
constexpr unsigned kCompactItems = kMaxCompactTopK / kThreads;

__device__ void validate_compact_row(
    const TokenProbability* row, unsigned size, unsigned vocabulary,
    Status* status, CompactReduce::TempStorage& storage) {
  float sum = 0.0F;
  for (unsigned index = threadIdx.x; index < size; index += kThreads) {
    const auto entry = row[index];
    if (!isfinite(entry.probability) || entry.probability < 0.0F ||
        entry.probability > 1.0F + kNormalizationTolerance ||
        entry.token >= vocabulary ||
        (index && row[index - 1].token >= entry.token))
      set_error(status, Status::invalid_distribution);
    sum += entry.probability;
  }
  sum = CompactReduce(storage).Sum(sum);
  if (threadIdx.x == 0 &&
      (!isfinite(sum) || fabsf(sum - 1.0F) > kNormalizationTolerance))
    set_error(status, Status::invalid_distribution);
  __syncthreads();
}

__global__ void validate_compact_rows_kernel(
    const TokenProbability* target, const TokenProbability* draft,
    unsigned depth, unsigned size, unsigned vocabulary, Status* status) {
  __shared__ CompactReduce::TempStorage storage;
  const std::size_t offset = std::size_t(blockIdx.x) * size;
  validate_compact_row(target + offset, size, vocabulary, status, storage);
  if (blockIdx.x < depth)
    validate_compact_row(draft + offset, size, vocabulary, status, storage);
}

__device__ void draw_compact(
    float (&weights)[kCompactItems], const TokenProbability* row, unsigned size,
    float uniform, std::uint32_t* output, Status* status, Status zero_mass,
    CompactScan::TempStorage& storage, unsigned& selected) {
  __shared__ float total;
  __shared__ unsigned last_positive;
  bool positive[kCompactItems];
  if (threadIdx.x == 0) last_positive = 0;
  __syncthreads();
  unsigned last = 0;
  for (unsigned item = 0; item < kCompactItems; ++item) {
    const unsigned index = threadIdx.x * kCompactItems + item;
    positive[item] = index < size && weights[item] > 0.0F;
    if (positive[item]) last = index;
  }
  atomicMax(&last_positive, last);
  CompactScan(storage).InclusiveSum(weights, weights);
  __syncthreads();
  // Use the final positive CDF bucket. Different parallel prefix reductions
  // can round a trailing zero bucket above it; never sample zero-mass entries.
  for (unsigned item = 0; item < kCompactItems; ++item)
    if (threadIdx.x * kCompactItems + item == last_positive) total = weights[item];
  __syncthreads();
  if (threadIdx.x == 0) {
    selected = UINT_MAX;
    *output = UINT_MAX;
    if (!valid_uniform(uniform)) set_error(status, Status::invalid_uniform);
    else if (!(total > 0.0F) || !isfinite(total)) set_error(status, zero_mass);
  }
  __syncthreads();
  if (*status != Status::success) return;
  const float threshold = fminf(uniform * total, nextafterf(total, 0.0F));
  for (unsigned item = 0; item < kCompactItems; ++item) {
    const unsigned index = threadIdx.x * kCompactItems + item;
    if (positive[item] && weights[item] > threshold) {
      atomicMin(&selected, index);
      break;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (selected < size) *output = row[selected].token;
    else set_error(status, Status::invalid_distribution);
  }
  __syncthreads();
}

__global__ void sample_compact_kernel(
    const TokenProbability* row, unsigned size, unsigned vocabulary,
    const float* uniform, std::uint32_t* output, Status* status) {
  __shared__ union {
    CompactReduce::TempStorage reduce;
    CompactScan::TempStorage scan;
  } storage;
  __shared__ unsigned selected;
  validate_compact_row(row, size, vocabulary, status, storage.reduce);
  float weights[kCompactItems];
  for (unsigned item = 0; item < kCompactItems; ++item) {
    const unsigned index = threadIdx.x * kCompactItems + item;
    weights[item] = index < size ? row[index].probability : 0.0F;
  }
  draw_compact(weights, row, size, *uniform, output, status,
               Status::invalid_distribution, storage.scan, selected);
}

__global__ void verify_compact_kernel(
    const TokenProbability* target, const TokenProbability* draft,
    const std::uint32_t* draft_ids, unsigned depth, unsigned size,
    unsigned vocabulary, const float* uniforms, const float* final_uniform,
    std::uint32_t* output, Result* result, Status* status) {
  __shared__ CompactScan::TempStorage storage;
  __shared__ unsigned selected;
  if (threadIdx.x == 0) {
    *result = {0, 0, depth};
    for (unsigned index = 0; index < depth && *status == Status::success; ++index) {
      const unsigned token = draft_ids[index];
      if (token >= vocabulary) { set_error(status, Status::invalid_draft_token); break; }
      if (!valid_uniform(uniforms[index])) { set_error(status, Status::invalid_uniform); break; }
      const auto offset = std::size_t(index) * size;
      const float p = lookup_probability(target + offset, size, token);
      const float q = lookup_probability(draft + offset, size, token);
      if (!(q > 0.0F)) { set_error(status, Status::zero_draft_probability); break; }
      if (!(p >= q || uniforms[index] * q < p)) {
        result->rejected_index = index;
        break;
      }
      output[index] = token;
      ++result->accepted_drafts;
    }
  }
  __syncthreads();
  if (*status != Status::success) return;
  const unsigned row = result->accepted_drafts;
  const auto* p = target + std::size_t(row) * size;
  const bool bonus = row == depth;
  float weights[kCompactItems];
  for (unsigned item = 0; item < kCompactItems; ++item) {
    const unsigned index = threadIdx.x * kCompactItems + item;
    weights[item] = 0.0F;
    if (index < size)
      weights[item] = bonus ? p[index].probability
          : fmaxf(p[index].probability - lookup_probability(
                draft + std::size_t(row) * size, size, p[index].token), 0.0F);
  }
  draw_compact(weights, p, size, *final_uniform, output + row, status,
               bonus ? Status::invalid_distribution : Status::zero_residual_mass,
               storage, selected);
  if (threadIdx.x == 0 && *status == Status::success) result->output_count = row + 1;
}

void check_compact_size(unsigned size, unsigned vocabulary) {
  require(vocabulary > 0 && vocabulary <= INT_MAX && size > 0 &&
              size <= kMaxCompactTopK && size <= vocabulary,
          "compact row must have 1..min(256, vocabulary) entries");
}

}  // namespace

const char* status_message(Status status) {
  switch (status) {
    case Status::success:
      return "success";
    case Status::invalid_logits:
      return "non-finite logits or temperature-scaled scores";
    case Status::invalid_distribution:
      return "invalid or unnormalized probability row";
    case Status::invalid_uniform:
      return "uniform draw is outside [0,1)";
    case Status::invalid_draft_token:
      return "draft token is outside the vocabulary";
    case Status::zero_draft_probability:
      return "draft token has zero proposal probability";
    case Status::zero_residual_mass:
      return "rejection has no positive residual mass";
  }
  return "unknown MTP sampling status";
}

std::size_t scratch_bytes(std::uint32_t vocabulary_size) {
  return make_plan(vocabulary_size).bytes;
}

void clear_status(Status* status, cudaStream_t stream) {
  require(status != nullptr, "status must be nonnull");
  check_cuda(cudaMemsetAsync(status, 0, sizeof(Status), stream),
             "clear MTP sampling status");
}

void build_distribution(const __nv_bfloat16* logits,
                        std::uint32_t vocabulary_size, float temperature,
                        float top_p, std::uint32_t top_k, float* output_probs,
                        void* scratch, std::size_t scratch_size, Status* status,
                        cudaStream_t stream,
                        const std::uint32_t* allowed_tokens) {
  require(logits != nullptr && output_probs != nullptr,
          "logits and probabilities must be nonnull");
  require(std::isfinite(temperature) && temperature >= 0.0F,
          "temperature must be finite and nonnegative");
  require(std::isfinite(top_p) && top_p >= 0.0F && top_p <= 1.0F,
          "top-p must be in [0,1]");
  const ScratchPlan plan =
      check_scratch(vocabulary_size, scratch, scratch_size, status);
  const bool greedy = temperature == 0.0F || top_p == 0.0F || top_k == 1;
  const std::uint32_t retained =
      greedy ? 1 : (top_k == 0 ? vocabulary_size
                               : std::min(top_k, vocabulary_size));
  auto* scores_in = at<float>(scratch, plan.scores_in);
  auto* scores_out = at<float>(scratch, plan.scores_out);
  auto* tokens_in = at<std::uint32_t>(scratch, plan.tokens_in);
  auto* tokens_out = at<std::uint32_t>(scratch, plan.tokens_out);
  auto* weights = at<float>(scratch, plan.weights);
  auto* cumulative = at<float>(scratch, plan.cumulative);
  auto* summary = at<DistributionSummary>(scratch, plan.summary);
  make_scores_kernel<<<blocks(vocabulary_size), kThreads, 0, stream>>>(
      logits, vocabulary_size, greedy ? 1.0F : temperature, scores_in,
      tokens_in, status, allowed_tokens);
  check_cuda(cudaGetLastError(), "make MTP sampling scores");
  std::size_t temporary_bytes = plan.temporary_bytes;
  if (greedy) {
    auto* best = at<cub::KeyValuePair<int, float>>(scratch, plan.greedy_result);
    check_cuda(cub::DeviceReduce::ArgMax(
                   at<void>(scratch, plan.temporary), temporary_bytes,
                   scores_in, best, static_cast<int>(vocabulary_size), stream),
               "MTP greedy argmax");
    greedy_probabilities_kernel<<<blocks(vocabulary_size), kThreads, 0, stream>>>(
        best, vocabulary_size, output_probs, status);
    check_cuda(cudaGetLastError(), "make MTP greedy probability row");
    return;
  }
  check_cuda(cub::DeviceRadixSort::SortPairsDescending(
                 at<void>(scratch, plan.temporary), temporary_bytes, scores_in,
                 scores_out, tokens_in, tokens_out, vocabulary_size, 0, 32,
                 stream),
             "MTP sampling radix sort");
  make_weights_kernel<<<blocks(retained), kThreads, 0, stream>>>(
      scores_out, retained, weights);
  check_cuda(cudaGetLastError(), "make MTP sampling weights");
  scan(weights, cumulative, retained, scratch, plan, stream);
  choose_nucleus_kernel<<<1, 1, 0, stream>>>(cumulative, retained, top_p, summary,
                                           status);
  scatter_probabilities_kernel<<<blocks(vocabulary_size), kThreads, 0, stream>>>(
      tokens_out, weights, vocabulary_size, summary, output_probs, status);
  check_cuda(cudaGetLastError(), "normalize MTP probability row");
}

void build_greedy_distributions(
    const std::vector<GreedyDistributionInput>& inputs,
    std::uint32_t vocabulary_size, cudaStream_t stream) {
  require(!inputs.empty(), "greedy distribution batch must be nonempty");
  require(vocabulary_size > 0 && vocabulary_size <= INT_MAX,
          "vocabulary size must be in 1..INT_MAX");
  for (const auto& input : inputs) {
    require(input.logits && input.status &&
                input.best_scratch && input.selected_id,
            "greedy distribution pointers must be nonnull");
    require(input.best_scratch != input.selected_id,
            "greedy best scratch must not alias selected ID");
  }
  GreedyBatch batch{};
  for (std::size_t first = 0; first < inputs.size();
       first += kGreedyBatchEntries) {
    const auto count = static_cast<unsigned>(std::min<std::size_t>(
        kGreedyBatchEntries, inputs.size() - first));
    std::copy_n(inputs.begin() + first, count, batch.inputs);
    initialize_greedy_batch_kernel<<<count, 1, 0, stream>>>(batch);
    greedy_best_batch_kernel<<<
        dim3((vocabulary_size + kGreedyTile - 1) / kGreedyTile, count),
        kThreads, 0, stream>>>(batch, vocabulary_size);
    finalize_greedy_batch_kernel<<<count, 1, 0, stream>>>(batch);
    check_cuda(cudaGetLastError(), "find batched MTP greedy winners");
    if (std::any_of(batch.inputs, batch.inputs + count,
                    [](const auto& input) { return input.output_probs; })) {
      greedy_probabilities_batch_kernel<<<
          dim3(blocks(vocabulary_size), count), kThreads, 0, stream>>>(
          batch, vocabulary_size);
      check_cuda(cudaGetLastError(), "make batched MTP greedy probabilities");
    }
  }
}

void verify_greedy_sequences(
    const std::vector<GreedyVerificationInput>& inputs,
    std::uint32_t vocabulary_size, cudaStream_t stream) {
  require(!inputs.empty(), "greedy verification batch must be nonempty");
  require(vocabulary_size > 0 && vocabulary_size <= INT_MAX,
          "vocabulary size must be in 1..INT_MAX");
  for (const auto& input : inputs) {
    require(input.target_ids && input.final_uniform && input.output_ids &&
                input.result && input.status,
            "greedy verification pointers must be nonnull");
    require(input.draft_count <= kMaxDraftTokens,
            "draft count exceeds 1279");
    require(input.draft_count == 0 ||
                (input.draft_ids && input.accept_uniforms),
            "nonempty greedy verification needs draft inputs");
  }
  GreedyVerificationBatch batch{};
  for (std::size_t first = 0; first < inputs.size();
       first += kGreedyVerificationBatchEntries) {
    const auto count = static_cast<unsigned>(std::min<std::size_t>(
        kGreedyVerificationBatchEntries, inputs.size() - first));
    std::copy_n(inputs.begin() + first, count, batch.inputs);
    verify_greedy_sequences_kernel<<<count, 1, 0, stream>>>(batch,
                                                            vocabulary_size);
    check_cuda(cudaGetLastError(), "verify batched MTP greedy sequences");
  }
}

void sample_distribution(const float* probs, std::uint32_t vocabulary_size,
                         const float* uniform, std::uint32_t* output_token,
                         void* scratch, std::size_t scratch_size, Status* status,
                         cudaStream_t stream) {
  require(probs != nullptr && uniform != nullptr && output_token != nullptr,
          "sampling inputs and output must be nonnull");
  const ScratchPlan plan =
      check_scratch(vocabulary_size, scratch, scratch_size, status);
  validate_probabilities(probs, 1, vocabulary_size, scratch, plan, status, stream);
  auto* cumulative = at<float>(scratch, plan.cumulative);
  scan(probs, cumulative, vocabulary_size, scratch, plan, stream);
  sample_cdf_kernel<<<1, 1, 0, stream>>>(probs, cumulative, vocabulary_size, uniform,
                                       output_token, status);
  check_cuda(cudaGetLastError(), "sample MTP probability row");
}

void gather_capture_rows(const std::uint16_t* probes, std::uint32_t layers,
    std::uint32_t rows, std::uint32_t width, const Result* result,
    const Status* status, std::uint16_t* output, cudaStream_t stream) {
  require(probes && result && status && output && layers && rows && width,
          "invalid target probe capture shape or pointers");
  gather_capture_rows_kernel<<<layers, kThreads, 0, stream>>>(probes, rows, width, result, status, output);
  check_cuda(cudaGetLastError(), "gather selected target probe rows");
}

void capture_scores(const void* target, const void* draft,
    const std::uint32_t* ids, std::uint32_t depth, std::uint32_t vocabulary_size,
    std::uint32_t compact_size, const std::uint32_t* greedy_ids,
    const Status* status, MtpCaptureScores* output, cudaStream_t stream) {
  require(ids && status && output && (greedy_ids || (target && draft)), "invalid capture pointers");
  require(depth && depth <= kMaxDraftTokens && vocabulary_size && compact_size <= vocabulary_size,
          "invalid capture shape");
  if (compact_size && !greedy_ids)
    capture_scores_kernel<<<depth, kThreads, 0, stream>>>(
        static_cast<const TokenProbability*>(target), static_cast<const TokenProbability*>(draft),
        ids, compact_size, nullptr, status, output);
  else
    capture_scores_kernel<<<depth, kThreads, 0, stream>>>(
        static_cast<const float*>(target), static_cast<const float*>(draft),
        ids, vocabulary_size, greedy_ids, status, output);
  check_cuda(cudaGetLastError(), "capture MTP probability scores");
}

void summarize_logprobs(const float* probs,
                        const std::uint32_t* selected_ids,
                        std::uint32_t row_count,
                        std::uint32_t vocabulary_size,
                        std::uint32_t top_logprobs, TokenLogprobs* output,
                        cudaStream_t stream) {
  require(probs != nullptr && selected_ids != nullptr && output != nullptr,
          "logprob inputs and output must be nonnull");
  require(row_count > 0 && row_count <= INT_MAX,
          "logprob row count must be in 1..INT_MAX");
  require(vocabulary_size > 0 && vocabulary_size <= INT_MAX,
          "logprob vocabulary size must be in 1..INT_MAX");
  require(top_logprobs <= kMaxTopLogprobs,
          "top-logprobs must be in 0..20");
  require(static_cast<std::size_t>(row_count) <=
              static_cast<std::size_t>(
                  std::numeric_limits<std::ptrdiff_t>::max()) /
                  sizeof(float) / vocabulary_size,
          "logprob probability row size overflows");
  summarize_logprobs_kernel<<<row_count, kThreads, 0, stream>>>(
      probs, selected_ids, vocabulary_size, top_logprobs, output);
  check_cuda(cudaGetLastError(), "summarize MTP log probabilities");
}

void verify_sequence(const float* target_probs, const float* draft_probs,
                     const std::uint32_t* draft_ids,
                     std::uint32_t draft_count,
                     std::uint32_t vocabulary_size,
                     const float* accept_uniforms, const float* final_uniform,
                     std::uint32_t* output_ids, Result* result, void* scratch,
                     std::size_t scratch_size, Status* status,
                     cudaStream_t stream) {
  require(draft_count <= kMaxDraftTokens, "draft count exceeds 1279");
  require(target_probs != nullptr && final_uniform != nullptr &&
              output_ids != nullptr && result != nullptr,
          "verification target inputs and outputs must be nonnull");
  require(draft_count == 0 ||
              (draft_probs != nullptr && draft_ids != nullptr &&
               accept_uniforms != nullptr),
          "nonempty verification needs draft inputs");
  const ScratchPlan plan =
      check_scratch(vocabulary_size, scratch, scratch_size, status);
  validate_probabilities(target_probs, draft_count + 1, vocabulary_size,
                         scratch, plan, status, stream);
  if (draft_count != 0) {
    validate_probabilities(draft_probs, draft_count, vocabulary_size,
                           scratch, plan, status, stream);
  }
  find_rejection_kernel<<<1, 1, 0, stream>>>(
      target_probs, draft_probs, draft_ids, draft_count, vocabulary_size,
      accept_uniforms, final_uniform, output_ids, result, status);
  auto* weights = at<float>(scratch, plan.weights);
  auto* cumulative = at<float>(scratch, plan.cumulative);
  make_final_weights_kernel<<<blocks(vocabulary_size), kThreads, 0, stream>>>(
      target_probs, draft_probs, draft_count, vocabulary_size, result, weights,
      status);
  check_cuda(cudaGetLastError(), "verify MTP proposals and form residual");
  scan(weights, cumulative, vocabulary_size, scratch, plan, stream);
  sample_final_kernel<<<1, 1, 0, stream>>>(
      weights, cumulative, vocabulary_size, draft_count, final_uniform, output_ids,
      result, status);
  check_cuda(cudaGetLastError(), "sample MTP correction or bonus");
}

std::uint32_t compact_row_size(std::uint32_t vocabulary_size, float temperature,
                               float top_p, std::uint32_t top_k) {
  if (temperature == 0.0F || top_p == 0.0F || top_k <= 1) return 0;
  const auto retained = std::min(vocabulary_size, top_k);
  return retained <= kMaxCompactTopK ? retained : 0;
}

void build_compact_distribution(
    const __nv_bfloat16* logits, std::uint32_t vocabulary_size,
    float temperature, float top_p, std::uint32_t row_size,
    TokenProbability* output, void* scratch, std::size_t scratch_size,
    Status* status, cudaStream_t stream, const std::uint32_t* allowed_tokens) {
  check_compact_size(row_size, vocabulary_size);
  require(logits && output && std::isfinite(temperature) && temperature > 0.0F &&
              std::isfinite(top_p) && top_p > 0.0F && top_p <= 1.0F,
          "invalid compact distribution inputs/settings");
  const auto plan = check_scratch(vocabulary_size, scratch, scratch_size, status);
  // Each pair of adjacent full-vocabulary scratch rows fits V candidates.
  auto* next = at<SelectionCandidate>(scratch, plan.scores_in);
  auto* other = at<SelectionCandidate>(scratch, plan.tokens_in);
  const SelectionCandidate* input = nullptr;
  unsigned count = vocabulary_size;
  if (count > 4096) {
    const unsigned tiles = (count + 1023) / 1024;
    compact_candidates_kernel<4, false><<<tiles, kThreads, 0, stream>>>(
        logits, nullptr, count, temperature, allowed_tokens, row_size, top_p,
        next, nullptr, status);
    input = next;
    std::swap(next, other);
    count = tiles * row_size;
  }
  while (count > 4096) {
    const unsigned tiles = (count + 4095) / 4096;
    compact_candidates_kernel<16, false><<<tiles, kThreads, 0, stream>>>(
        logits, input, count, temperature, allowed_tokens, row_size, top_p,
        next, nullptr, status);
    input = next;
    std::swap(next, other);
    count = tiles * row_size;
  }
  if (count <= 1024)
    compact_candidates_kernel<4, true><<<1, kThreads, 0, stream>>>(
        logits, input, count, temperature, allowed_tokens, row_size, top_p,
        nullptr, output, status);
  else
    compact_candidates_kernel<16, true><<<1, kThreads, 0, stream>>>(
        logits, input, count, temperature, allowed_tokens, row_size, top_p,
        nullptr, output, status);
  check_cuda(cudaGetLastError(), "select and normalize compact MTP top-k");
}

void build_compact_distributions(
    const std::vector<CompactDistributionInput>& inputs,
    std::uint32_t vocabulary_size, cudaStream_t stream) {
  require(!inputs.empty(), "compact distribution batch must be nonempty");
  const auto plan = make_plan(vocabulary_size);
  for (const auto& row : inputs) {
    check_compact_size(row.row_size, vocabulary_size);
    require(row.logits && row.output && row.status && row.scratch &&
                reinterpret_cast<std::uintptr_t>(row.scratch) % kAlignment == 0 &&
                row.scratch_size >= plan.bytes &&
                std::isfinite(row.temperature) && row.temperature > 0 &&
                std::isfinite(row.top_p) && row.top_p > 0 && row.top_p <= 1,
            "invalid compact batch row inputs/settings/scratch");
  }
  for (std::size_t first = 0; first < inputs.size();) {
    const unsigned retained = inputs[first].row_size;
    unsigned rows = 1;
    while (rows < kCompactBatchEntries && first + rows < inputs.size() &&
           inputs[first + rows].row_size == retained) ++rows;
    if (rows == 1) {
      const auto& row = inputs[first++];
      build_compact_distribution(row.logits, vocabulary_size, row.temperature,
          row.top_p, retained, row.output, row.scratch, row.scratch_size,
          row.status, stream, row.allowed_tokens);
      continue;
    }
    CompactCandidateBatch batch{};
    SelectionCandidate* other[kCompactBatchEntries]{};
    for (unsigned i = 0; i < rows; ++i) {
      const auto& row = inputs[first + i];
      batch.rows[i] = {row.logits, nullptr,
          at<SelectionCandidate>(row.scratch, plan.scores_in), row.output,
          row.allowed_tokens, row.status, row.temperature, row.top_p};
      other[i] = at<SelectionCandidate>(row.scratch, plan.tokens_in);
    }
    const auto advance = [&] {
      for (unsigned i = 0; i < rows; ++i) {
        batch.rows[i].input = batch.rows[i].next;
        std::swap(batch.rows[i].next, other[i]);
      }
    };
    unsigned count = vocabulary_size;
    if (count > 4096) {
      const unsigned tiles = (count + 1023) / 1024;
      compact_candidates_batch_kernel<4, false><<<dim3(tiles, rows), kThreads, 0, stream>>>(
          batch, count, retained);
      advance();
      count = tiles * retained;
    }
    while (count > 4096) {
      const unsigned tiles = (count + 4095) / 4096;
      compact_candidates_batch_kernel<16, false><<<dim3(tiles, rows), kThreads, 0, stream>>>(
          batch, count, retained);
      advance();
      count = tiles * retained;
    }
    if (count <= 1024)
      compact_candidates_batch_kernel<4, true><<<dim3(1, rows), kThreads, 0, stream>>>(
          batch, count, retained);
    else
      compact_candidates_batch_kernel<16, true><<<dim3(1, rows), kThreads, 0, stream>>>(
          batch, count, retained);
    check_cuda(cudaGetLastError(), "select and normalize batched compact MTP top-k");
    first += rows;
  }
}

void sample_compact_distribution(
    const TokenProbability* probs, std::uint32_t row_size,
    std::uint32_t vocabulary_size, const float* uniform,
    std::uint32_t* output_token, Status* status, cudaStream_t stream) {
  check_compact_size(row_size, vocabulary_size);
  require(probs && uniform && output_token && status,
          "compact sample inputs and output must be nonnull");
  sample_compact_kernel<<<1, kThreads, 0, stream>>>(
      probs, row_size, vocabulary_size, uniform, output_token, status);
  check_cuda(cudaGetLastError(), "sample compact MTP distribution");
}

void verify_compact_sequence(
    const TokenProbability* target, const TokenProbability* draft,
    const std::uint32_t* draft_ids, std::uint32_t draft_count,
    std::uint32_t row_size, std::uint32_t vocabulary_size,
    const float* accept_uniforms, const float* final_uniform,
    std::uint32_t* output_ids, Result* result, Status* status,
    cudaStream_t stream) {
  check_compact_size(row_size, vocabulary_size);
  require(draft_count <= kMaxDraftTokens && target && final_uniform &&
              output_ids && result && status &&
              (draft_count == 0 || (draft && draft_ids && accept_uniforms)),
          "invalid compact verification inputs");
  validate_compact_rows_kernel<<<draft_count + 1, kThreads, 0, stream>>>(
      target, draft, draft_count, row_size, vocabulary_size, status);
  verify_compact_kernel<<<1, kThreads, 0, stream>>>(
      target, draft, draft_ids, draft_count, row_size, vocabulary_size,
      accept_uniforms, final_uniform, output_ids, result, status);
  check_cuda(cudaGetLastError(), "verify compact MTP sequence");
}

void summarize_compact_logprobs(
    const TokenProbability* probs, const std::uint32_t* selected_ids,
    std::uint32_t row_count, std::uint32_t row_size,
    std::uint32_t top_logprobs, TokenLogprobs* output, cudaStream_t stream) {
  check_compact_size(row_size, kMaxCompactTopK);
  require(probs && selected_ids && output && row_count > 0 && row_count <= INT_MAX &&
              top_logprobs <= kMaxTopLogprobs,
          "invalid compact logprob arguments");
  summarize_logprobs_kernel<<<row_count, kThreads, 0, stream>>>(
      probs, selected_ids, row_size, top_logprobs, output);
  check_cuda(cudaGetLastError(), "summarize compact MTP log probabilities");
}

}  // namespace gewell::mtp_sampling
