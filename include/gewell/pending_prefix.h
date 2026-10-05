#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gewell/prefix_index.h"

namespace gewell::pending_prefix {

using Prompt = std::shared_ptr<const std::vector<std::uint32_t>>;

// Exact common history, rounded back to feature begin if a token mismatch or
// truncated prompt falls inside an image. Spans may extend beyond the token
// vectors when callers compare a shorter prefix of a complete prompt.
[[nodiscard]] std::uint32_t common_tokens(
    const std::vector<std::uint32_t>& a_tokens,
    const std::vector<prefix_index::ImageSpan>& a_images,
    const std::vector<std::uint32_t>& b_tokens,
    const std::vector<prefix_index::ImageSpan>& b_images);

struct Dependent {
  std::size_t request_id{};
  Prompt prompt;
  std::uint32_t fork_tokens{};
};

struct JoinResult {
  bool joined{};
  // Common history at a safe image boundary, including when dispatch has
  // already passed this boundary.
  std::uint32_t common_tokens{};
};

struct Step {
  std::uint32_t begin{};
  std::uint32_t end{};
};

// Host-side lifetime and boundary bookkeeping for one shared prefill path.
// GPU execution/KV ownership belongs to the caller's work record, never to a
// representative request. Prompt storage survives cancellation until this
// work is destroyed after its last dispatched step has synchronized.
class Work {
 public:
  Work(Prompt prompt, std::size_t request_id,
       std::size_t max_dependents = 4096,
       std::uint32_t initial_completed = 0,
       const std::vector<prefix_index::ImageSpan>& images = {});

  // A join can shorten the next boundary, but cannot split an already
  // dispatched chunk or rewind completed work. False also covers a full or
  // failed work record. Invalid prompts and duplicate request IDs are errors.
  [[nodiscard]] JoinResult join(
      std::size_t request_id, Prompt prompt,
      const std::vector<prefix_index::ImageSpan>& images = {});
  bool cancel(std::size_t request_id);
  bool detach(std::size_t request_id);
  // Commit a completed checkpoint restored after dependents registered. Only
  // fresh, idle work may restore, and it cannot skip a known fork boundary.
  // This changes the completed cursor without dispatching or removing waiters.
  void restore_prefix(std::uint32_t completed);

  // The caller first branches/detaches dependents at a ready boundary.
  // complete_step must follow synchronization, including a failed GPU step.
  // Text stops before the next image. Image features dispatch as one complete
  // span even when their row count exceeds max_rows.
  [[nodiscard]] Step next_step(std::uint32_t max_rows) const;
  [[nodiscard]] Step begin_step(std::uint32_t max_rows);
  void complete_step(bool success);

  [[nodiscard]] const std::vector<Dependent>& dependents() const {
    return dependents_;
  }
  [[nodiscard]] const Prompt& prompt() const { return prompt_; }
  [[nodiscard]] const std::vector<prefix_index::ImageSpan>& images() const {
    return images_;
  }
  [[nodiscard]] std::uint32_t completed() const { return completed_; }
  [[nodiscard]] std::optional<std::uint32_t> dispatched_end() const {
    return dispatched_end_;
  }
  [[nodiscard]] bool inflight() const { return dispatched_end_.has_value(); }
  [[nodiscard]] bool failed() const { return failed_; }
  // Empty work has no future boundary; return its completed cursor.
  [[nodiscard]] std::uint32_t next_boundary() const;
  [[nodiscard]] bool at_boundary() const;
  [[nodiscard]] bool releasable() const {
    return dependents_.empty() && !inflight();
  }

 private:
  Prompt prompt_;
  std::vector<prefix_index::ImageSpan> images_;
  std::size_t max_dependents_{};
  std::vector<Dependent> dependents_;
  std::uint32_t completed_{};
  std::optional<std::uint32_t> dispatched_end_;
  bool failed_{};
};

}  // namespace gewell::pending_prefix
