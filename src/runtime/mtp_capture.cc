#include "gewell/runtime/mtp_capture.h"
#include "gewell/runtime/scheduler.h"
#include "gewell/console.h"
#include "mtp_file.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace gewell::runtime {
namespace {
std::uint64_t mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}
}

struct MtpCapture::Impl {
  mtp_file::File index, data;
  MtpCaptureSettings settings;
  std::uint64_t attempts{}, samples{}, offset{}, next_sequence{};
  const std::uint32_t target_width, assistant_width;

  Impl(const BatchLimits& limits, const MtpCaptureGeometry& geometry)
      : settings(limits.mtp_capture), target_width(geometry.target_width), assistant_width(geometry.assistant_width) {
    if (!target_width || !assistant_width || !geometry.layer_count)
      throw std::runtime_error("MTP capture requires the loaded model geometry");
    if (settings.layers.empty()) settings.layers = geometry.default_layers;
    if (settings.path.empty() || !settings.every || !settings.max_samples || !limits.mtp_depth)
      throw std::runtime_error("MTP capture requires a directory, positive sampling limits, and MTP enabled");
    if (settings.layers.empty() || !settings.layers.front() || settings.layers.back() > geometry.layer_count ||
        !std::is_sorted(settings.layers.begin(), settings.layers.end()) ||
        std::adjacent_find(settings.layers.begin(), settings.layers.end()) != settings.layers.end())
      throw std::runtime_error("--mtp-capture-layers must be increasing completed-layer counts in 1.." +
                               std::to_string(geometry.layer_count));
    std::filesystem::create_directory(settings.path);
    const auto root = std::filesystem::path(settings.path);
    const auto index_path = (root / "samples.jsonl").string();
    const auto data_path = (root / "hidden.bf16").string();
    index = mtp_file::open(index_path);
    const auto now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    nlohmann::json config = {{"event", "mtp_capture_config"}, {"version", 1}, {"started_unix_seconds", now},
          {"every", settings.every}, {"max_samples", settings.max_samples},
          {"selection", "splitmix64(sequence ^ splitmix64(source_cycle)) % every == 0; label next round"},
          {"target_layers", settings.layers}, {"target_probe_kind", "post_layer_residual_before_next_norm"},
          {"dtype", "bfloat16_le"}, {"hidden_file", "hidden.bf16"},
          {"layout", "target_hidden[target_width], assistant_hidden[depth,assistant_width], target_probes[layers,target_width]"},
          {"max_batch", limits.capacity}, {"mtp_depth", limits.mtp_depth},
          {"mtp_min_depth", limits.mtp_min_depth}, {"decode_width", limits.decode_width}};
    bool configured = false;
    const auto existing = mtp_file::scan(index_path, [&](const nlohmann::json& row) {
      if (row.at("event") == "mtp_capture_config") {
        for (const auto* key : {"version", "mtp_depth", "target_layers", "target_probe_kind", "dtype", "hidden_file", "layout"})
          if (row.at(key) != config.at(key))
            throw std::runtime_error(std::string("capture configuration differs: ") + key);
        configured = true;
        return;
      }
      if (!configured || row.at("event") != "mtp_capture")
        throw std::runtime_error("expected capture configuration or sample");
      const auto target = row.at("target_width").get<std::uint32_t>();
      const auto assistant = row.at("assistant_width").get<std::uint32_t>();
      const auto depth = row.at("depth").get<std::uint32_t>();
      const auto bytes = 2 * ((settings.layers.size() + 1) * target + std::uint64_t(depth) * assistant);
      if (!target || !assistant || !depth || depth > limits.mtp_depth ||
          target != target_width || assistant != assistant_width || row.at("target_layers") != settings.layers ||
          row.at("sample") != samples || row.at("byte_offset") != offset || row.at("byte_length") != bytes)
        throw std::runtime_error("invalid capture shape, sample number, or byte range");
      row.at("sequence");
      offset += bytes;
      ++samples;
    });
    next_sequence = existing.next_sequence;
    data = mtp_file::open(data_path, samples ? "r+b" : "a+b");
    if (mtp_file::size(data.get(), data_path) < offset)
      mtp_file::fail(data_path, "missing indexed payload; existing capture was left unchanged");
    mtp_file::resume(index.get(), index_path, existing);
    mtp_file::trim(data.get(), data_path, offset);
    config["sample_start"] = samples;
    if (!write_index(config))
      throw std::runtime_error("MTP capture: cannot write configuration in " + settings.path);
  }

  void disable(const std::string& error) {
    index.reset(); data.reset();
    console::event("mtp_capture_error", {{"path", settings.path}, {"error", error}},
        "MTP capture disabled: " + settings.path + ": " + error, true);
  }
  bool write_index(const nlohmann::json& value) {
    const auto line = value.dump() + '\n';
    if (std::fwrite(line.data(), 1, line.size(), index.get()) == line.size() &&
        std::fflush(index.get()) == 0) return true;
    disable(std::strerror(errno));
    return false;
  }
};

MtpCapture::MtpCapture(const BatchLimits& limits, const MtpCaptureGeometry& geometry)
    : impl_(std::make_unique<Impl>(limits, geometry)) {}
MtpCapture::~MtpCapture() = default;
const std::vector<std::uint32_t>& MtpCapture::layers() const { return impl_->settings.layers; }
bool MtpCapture::enabled() const { return impl_->index && impl_->attempts < impl_->settings.max_samples; }
std::uint64_t MtpCapture::next_sequence() const { return impl_->next_sequence; }
bool MtpCapture::select(std::uint64_t sequence, std::uint64_t cycle) {
  if (!enabled() || mix(sequence ^ mix(cycle)) % impl_->settings.every) return false;
  if (++impl_->attempts == impl_->settings.max_samples)
    console::event("mtp_capture_limit", {{"path", impl_->settings.path}, {"attempts", impl_->attempts}},
        "MTP capture sample limit reached: " + impl_->settings.path);
  return true;
}

void MtpCapture::record(const BatchRequest& request, const BatchMtpInput& input,
    const MtpOutcome& result, std::uint64_t batch_id, std::uint32_t batch_size, std::uint32_t emitted) {
  auto& s = *impl_;
  if (!s.index || !input.capture || !result.error.empty()) return;
  const auto& features = *input.capture;
  if (features.target_width != s.target_width || features.assistant_width != s.assistant_width ||
      features.target_hidden.size() != features.target_width ||
      features.assistant_hidden.size() != std::size_t(input.depth) * features.assistant_width ||
      features.draft_tokens.size() != input.depth || features.scores.size() != input.depth ||
      features.probes.width != features.target_width || features.probes.layers != s.settings.layers ||
      features.probes.hidden.size() != features.probes.layers.size() * features.target_width ||
      features.probes.position != input.position || features.probes.pending_token != input.pending_token) {
    s.disable("backend returned an invalid feature shape");
    return;
  }
  auto scores = nlohmann::json::array();
  for (const auto& score : features.scores)
    scores.push_back({score.draft_probability, score.target_probability,
                      score.draft_entropy, score.draft_max_probability});
  const auto bytes = 2 * (features.target_hidden.size() + features.assistant_hidden.size() + features.probes.hidden.size());
  nlohmann::json record = {
    {"event", "mtp_capture"}, {"sample", s.samples}, {"id", request.id}, {"sequence", request.accepted_order},
    {"cycle", request.mtp_cycles}, {"batch_id", batch_id}, {"batch_size", batch_size},
    {"position", input.position}, {"output_begin", request.outputs.size()},
    {"prompt_tokens", request.prompt->size()}, {"max_new_tokens", request.max_new_tokens},
    {"pending_token", input.pending_token}, {"depth", input.depth},
    {"accepted", result.verification.accepted_drafts}, {"rejected_index", result.verification.rejected_index},
    {"selected_tokens", result.tokens}, {"emitted", emitted}, {"eos_truncated", emitted < result.tokens.size()},
    {"honor_eos", request.honor_eos}, {"constrained", bool(request.constraint)},
    {"temperature", input.temperature}, {"top_p", input.top_p}, {"top_k", input.top_k}, {"seed", request.seed},
    {"previous_depth", request.mtp_previous_depth}, {"previous_accepted", request.mtp_previous_accepted},
    {"prior_cycles", request.mtp_cycles}, {"prior_proposed", request.mtp_proposed}, {"prior_accepted", request.mtp_accepted},
    {"target_width", features.target_width}, {"assistant_width", features.assistant_width},
    {"target_layers", features.probes.layers}, {"probe_source_cycle", request.mtp_cycles - 1},
    {"byte_offset", s.offset}, {"byte_length", bytes}, {"draft_tokens", features.draft_tokens},
    {"scores", scores}, {"score_columns", {"q", "p", "q_entropy_nats", "q_max"}}};
  // Publish the index only after the corresponding payload is flushed. A
  // partial last line or unindexed binary tail can be discarded after a crash.
  for (const auto* values : {&features.target_hidden, &features.assistant_hidden, &features.probes.hidden}) {
    if (std::fwrite(values->data(), sizeof(std::uint16_t), values->size(), s.data.get()) != values->size()) {
      s.disable(std::strerror(errno)); return;
    }
  }
  if (std::fflush(s.data.get()) != 0) { s.disable(std::strerror(errno)); return; }
  if (s.write_index(record)) { s.offset += bytes; ++s.samples; }
}
}  // namespace gewell::runtime
