#pragma once

#include "gewell/tokenizer.h"
#include "json.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gewell::constraint {

// Reject unsupported constraints instead of passing ignored keywords through
// to the grammar engine. Annotations such as Pydantic titles/defaults are kept.
// Objects declaring properties generate only those keys; dictionary objects
// retain their additionalProperties value schema.
[[nodiscard]] nlohmann::json normalize_schema(const nlohmann::json& schema);

class Compiled {
 public:
  [[nodiscard]] std::size_t memory_bytes() const;
  [[nodiscard]] std::size_t mask_words() const;

 private:
  struct Impl;
  explicit Compiled(std::shared_ptr<const Impl> impl);
  std::shared_ptr<const Impl> impl_;
  friend class Compiler;
  friend class State;
};

// Thread-safe shared compiler with a bounded cache. Compilation runs before
// scheduler admission. JSON object mode uses compile({{"type", "object"}}).
class Compiler {
 public:
  Compiler(const text::Tokenizer& tokenizer, std::uint32_t maximum_speculative_depth);
  ~Compiler();
  [[nodiscard]] std::shared_ptr<const Compiled> compile(const nlohmann::json& schema);
  // Constrain function selection, JSON arguments, call count, and handoff.
  // Non-strict functions use generic JSON objects; strict functions use their
  // parameter schemas. Validate all declarations, then allow only selected names.
  // An optional answer grammar applies only to text output.
  [[nodiscard]] std::shared_ptr<const Compiled> compile_tools(
      const nlohmann::json& tools, const std::vector<std::string>& selected,
      bool required, bool parallel,
      std::shared_ptr<const Compiled> answer = {});

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class State {
 public:
  State(std::shared_ptr<const Compiled> compiled, bool allow_thinking = false,
        bool initial_reasoning = false);
  ~State();
  // Each set bit permits a token; bit zero is the lowest token in each word.
  [[nodiscard]] std::size_t mask_words() const;
  void fill_mask(std::uint32_t* mask);
  // Invalid tokens return false without changing the committed state.
  [[nodiscard]] bool accept(std::uint32_t token);
  [[nodiscard]] bool completed() const;
  [[nodiscard]] bool terminated() const;
  // Fill depth+1 verifier rows while temporarily walking the draft prefix,
  // then restore state. Rows after an invalid/terminal draft are unreachable
  // and filled with all ones; the verifier must never commit from those rows.
  void block_masks(const std::uint32_t* draft_ids, std::uint32_t depth,
                   std::uint32_t* masks);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::constraint
