#include "gewell/runtime/mtp_stats.h"
#include "gewell/runtime/scheduler.h"
#include "gewell/console.h"
#include "mtp_file.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <tuple>

namespace gewell::runtime {
struct MtpStats::Impl {
  struct Group {
    std::uint64_t cycles{}, failed{}, emitted{};
    std::vector<std::uint64_t> accepted;
    double wall{}, gpu{}, draft{}, verify{}, select{};
  };
  struct Window {
    std::uint64_t index{}, total_cycles{}, cycles{}, first_batch{}, last_batch{};
    std::size_t output_begin{}, output_end{};
    // Actual depth, actual ready batch, and ordinary versus verifier execution.
    std::map<std::tuple<std::uint32_t, std::uint32_t, bool>, Group> groups;
  };
  mtp_file::File file;
  const std::string path;
  const std::uint32_t window_cycles;
  std::map<std::uint64_t, Window> windows;
  std::uint64_t next_sequence{};

  explicit Impl(const BatchLimits& limits)
      : path(limits.mtp_stats_path), window_cycles(limits.mtp_stats_window) {
    if (path.empty() || !window_cycles) throw std::runtime_error("MTP statistics require a path and positive window size");
    file = mtp_file::open(path);
    bool configured = false;
    const auto existing = mtp_file::scan(path, [&](const nlohmann::json& row) {
      const auto& event = row.at("event");
      if (event == "mtp_stats_config") {
        if (row.at("version") != 1) throw std::runtime_error("unsupported statistics version");
        configured = true;
      } else {
        if (!configured || (event != "mtp_request" && event != "mtp_window" && event != "mtp_request_end"))
          throw std::runtime_error("expected statistics configuration or request record");
        row.at("sequence");
      }
    });
    next_sequence = existing.next_sequence;
    mtp_file::resume(file.get(), path, existing);
    const auto unix_seconds = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (!write({{"event", "mtp_stats_config"}, {"version", 1}, {"started_unix_seconds", unix_seconds},
                {"window_cycles", window_cycles}, {"max_batch", limits.capacity},
                {"mtp_depth", limits.mtp_depth}, {"mtp_min_depth", limits.mtp_min_depth},
                {"decode_width", limits.decode_width}, {"timing_scope", "shared_decode_batch"}}))
      throw std::runtime_error("MTP statistics: cannot write " + path);
  }

  bool write(const nlohmann::json& record) {
    if (!file) return false;
    const auto line = record.dump() + '\n';
    if (std::fwrite(line.data(), 1, line.size(), file.get()) == line.size() && std::fflush(file.get()) == 0)
      return true;
    const std::string error = std::strerror(errno);
    file.reset();
    // Statistics failure must not abort an otherwise healthy inference job.
    console::event("mtp_stats_error", {{"path", path}, {"error", error}},
        "MTP statistics disabled after a write failure: " + path + ": " + error, true);
    return false;
  }

  Window& start(const BatchRequest& request) {
    auto [it, inserted] = windows.try_emplace(request.accepted_order);
    if (inserted) {
      write({{"event", "mtp_request"}, {"id", request.id}, {"sequence", request.accepted_order},
             {"prompt_tokens", request.prompt->size()}, {"max_new_tokens", request.max_new_tokens},
             {"cached_tokens", request.cached_tokens}, {"shared_tokens", request.shared_tokens},
             {"temperature", request.sampling.temperature}, {"top_p", request.sampling.top_p},
             {"top_k", request.sampling.top_k}, {"seed", request.seed},
             {"ordinary_decode", request.ordinary_decode}});
    }
    return it->second;
  }

  void flush(const BatchRequest& request, Window& window) {
    if (!window.cycles) return;
    auto groups = nlohmann::json::array();
    for (const auto& [shape, group] : window.groups) {
      const auto& [depth, batch, speculative] = shape;
      groups.push_back({{"depth", depth}, {"batch", batch}, {"mtp", speculative},
          {"cycles", group.cycles}, {"failed_cycles", group.failed},
          {"accepted", group.accepted}, {"emitted_tokens", group.emitted},
          {"wall_seconds", group.wall}, {"gpu_seconds", group.gpu},
          {"draft_gpu_seconds", group.draft}, {"verify_gpu_seconds", group.verify},
          {"select_gpu_seconds", group.select}});
    }
    write({{"event", "mtp_window"}, {"id", request.id}, {"sequence", request.accepted_order},
           {"window", window.index}, {"cycles", window.cycles},
           {"first_batch", window.first_batch}, {"last_batch", window.last_batch},
           {"output_begin", window.output_begin}, {"output_end", window.output_end}, {"groups", groups}});
    ++window.index;
    window.cycles = 0;
    window.groups.clear();
  }
};

MtpStats::MtpStats(const BatchLimits& limits) : impl_(std::make_unique<Impl>(limits)) {}
MtpStats::~MtpStats() = default;
bool MtpStats::enabled() const { return bool(impl_->file); }
std::uint64_t MtpStats::next_sequence() const { return impl_->next_sequence; }

void MtpStats::observe(const BatchRequest& request, const MtpStatsSample& sample) {
  if (!enabled()) return;
  auto& window = impl_->start(request);
  if (!window.cycles) {
    window.first_batch = sample.batch_id;
    window.output_begin = sample.output_begin;
  }
  window.last_batch = sample.batch_id;
  window.output_end = sample.output_begin + sample.emitted;
  ++window.total_cycles;
  ++window.cycles;
  auto& group = window.groups[{sample.depth, sample.batch_size, sample.speculative}];
  if (group.accepted.empty()) group.accepted.resize(sample.depth + 1);
  ++group.cycles;
  if (sample.failed) ++group.failed;
  else ++group.accepted.at(sample.accepted);
  group.emitted += sample.emitted;
  group.wall += sample.wall_seconds;
  group.gpu += sample.gpu_seconds;
  group.draft += sample.draft_seconds;
  group.verify += sample.verify_seconds;
  group.select += sample.select_seconds;
  if (window.cycles == impl_->window_cycles) impl_->flush(request, window);
}

void MtpStats::finish(const BatchRequest& request, std::string_view status) {
  if (request.operation != Operation::generate || !request.prompt) return;
  if (enabled()) {
    auto& window = impl_->start(request);
    impl_->flush(request, window);
    impl_->write({{"event", "mtp_request_end"}, {"id", request.id}, {"sequence", request.accepted_order},
        {"status", status}, {"windows", window.index}, {"decode_cycles", window.total_cycles},
        {"output_tokens", request.outputs.size()}, {"mtp_cycles", request.mtp_cycles},
        {"mtp_proposed", request.mtp_proposed}, {"mtp_accepted", request.mtp_accepted},
        {"accepted", request.mtp_accepted_histogram}});
  }
  impl_->windows.erase(request.accepted_order);
}
}  // namespace gewell::runtime
