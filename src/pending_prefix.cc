#include "gewell/pending_prefix.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gewell::pending_prefix {
namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(std::string("pending prefix: ") + message);
}

void validate_prompt(const Prompt& prompt) {
  require(prompt && !prompt->empty(), "prompt must be nonempty");
  require(prompt->size() <= std::numeric_limits<std::uint32_t>::max(),
          "prompt exceeds cursor capacity");
}

void validate_images(const std::vector<prefix_index::ImageSpan>& images) {
  std::uint32_t previous_end = 0;
  for (const auto& image : images) {
    require(image.begin < image.end && image.begin >= previous_end,
            "images must be nonempty, ordered, and nonoverlapping");
    previous_end = image.end;
  }
}

std::uint32_t safe_boundary(
    std::uint32_t position,
    const std::vector<prefix_index::ImageSpan>& images) {
  for (const auto& image : images) {
    if (image.begin >= position) break;
    if (position < image.end) return image.begin;
  }
  return position;
}

}  // namespace

std::uint32_t common_tokens(
    const std::vector<std::uint32_t>& a_tokens,
    const std::vector<prefix_index::ImageSpan>& a_images,
    const std::vector<std::uint32_t>& b_tokens,
    const std::vector<prefix_index::ImageSpan>& b_images) {
  require(a_tokens.size() <= std::numeric_limits<std::uint32_t>::max() &&
              b_tokens.size() <= std::numeric_limits<std::uint32_t>::max(),
          "prompt exceeds cursor capacity");
  validate_images(a_images);
  validate_images(b_images);
  const auto limit = std::min(a_tokens.size(), b_tokens.size());
  std::size_t a_index = 0, b_index = 0;
  std::uint32_t common = 0;
  while (common < limit && a_tokens[common] == b_tokens[common]) {
    const auto* a_image = a_index < a_images.size() &&
                                 a_images[a_index].begin == common
                             ? &a_images[a_index]
                             : nullptr;
    const auto* b_image = b_index < b_images.size() &&
                                 b_images[b_index].begin == common
                             ? &b_images[b_index]
                             : nullptr;
    if (static_cast<bool>(a_image) != static_cast<bool>(b_image) ||
        (a_image && (a_image->end != b_image->end ||
                     a_image->digest != b_image->digest))) {
      break;
    }
    if (a_image) ++a_index;
    if (b_image) ++b_index;
    ++common;
  }
  return safe_boundary(safe_boundary(common, a_images), b_images);
}

Work::Work(Prompt prompt, std::size_t request_id, std::size_t max_dependents,
           std::uint32_t initial_completed,
           const std::vector<prefix_index::ImageSpan>& images)
    : prompt_(std::move(prompt)), images_(images), max_dependents_(max_dependents),
      completed_(initial_completed) {
  validate_prompt(prompt_);
  validate_images(images_);
  require(images_.empty() || images_.back().end <= prompt_->size(),
          "image span exceeds prompt");
  require(max_dependents_ > 0, "dependent limit must be positive");
  require(completed_ <= prompt_->size(), "initial cursor exceeds prompt");
  require(safe_boundary(completed_, images_) == completed_,
          "initial cursor splits an image");
  dependents_.push_back({request_id, prompt_, static_cast<std::uint32_t>(prompt_->size())});
}

JoinResult Work::join(std::size_t request_id, Prompt prompt,
                      const std::vector<prefix_index::ImageSpan>& images) {
  validate_prompt(prompt);
  validate_images(images);
  require(images.empty() || images.back().end <= prompt->size(),
          "image span exceeds prompt");
  require(std::none_of(dependents_.begin(), dependents_.end(),
                      [request_id](const Dependent& d) { return d.request_id == request_id; }),
          "request already depends on this work");
  const auto common = common_tokens(*prompt_, images_, *prompt, images);
  if (failed_ || dependents_.size() == max_dependents_ || common == 0 ||
      common < dispatched_end_.value_or(completed_)) {
    return {false, common};
  }
  dependents_.push_back({request_id, std::move(prompt), common});
  return {true, common};
}

bool Work::cancel(std::size_t request_id) { return detach(request_id); }

bool Work::detach(std::size_t request_id) {
  const auto it = std::find_if(dependents_.begin(), dependents_.end(),
      [request_id](const Dependent& d) { return d.request_id == request_id; });
  if (it == dependents_.end()) return false;
  dependents_.erase(it);
  return true;
}

void Work::restore_prefix(std::uint32_t completed) {
  require(!failed_ && !inflight() && completed_ == 0,
          "only fresh idle work can restore a prefix");
  require(completed <= next_boundary(), "restored prefix crosses a known boundary");
  require(safe_boundary(completed, images_) == completed,
          "restored prefix splits an image");
  completed_ = completed;
}

std::uint32_t Work::next_boundary() const {
  if (dependents_.empty()) return completed_;
  return std::min_element(dependents_.begin(), dependents_.end(),
      [](const Dependent& a, const Dependent& b) {
        return a.fork_tokens < b.fork_tokens;
      })->fork_tokens;
}

bool Work::at_boundary() const {
  return !failed_ && !inflight() && !dependents_.empty() && completed_ == next_boundary();
}

Step Work::next_step(std::uint32_t max_rows) const {
  require(!failed_, "failed work cannot dispatch");
  require(!inflight(), "step already in flight");
  require(!dependents_.empty(), "no dependents to process");
  require(max_rows > 0, "step size must be positive");
  const auto boundary = next_boundary();
  require(completed_ < boundary, "ready boundary must be detached before dispatch");
  auto end = completed_ + std::min(max_rows, boundary - completed_);
  for (const auto& image : images_) {
    if (image.end <= completed_) continue;
    end = completed_ == image.begin ? image.end : std::min(end, image.begin);
    break;
  }
  require(end <= boundary, "step crosses a known boundary");
  return {completed_, end};
}

Step Work::begin_step(std::uint32_t max_rows) {
  const auto step = next_step(max_rows);
  dispatched_end_ = step.end;
  return step;
}

void Work::complete_step(bool success) {
  require(inflight(), "no dispatched step to complete");
  if (success) completed_ = *dispatched_end_;
  else failed_ = true;
  dispatched_end_.reset();
}

}  // namespace gewell::pending_prefix
