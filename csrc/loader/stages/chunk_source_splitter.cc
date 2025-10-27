#include "loader/stages/chunk_source_splitter.h"

#include <algorithm>
#include <numeric>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "loader/data_loader_metrics.h"

namespace lczero {
namespace training {

ChunkSourceSplitter::ChunkSourceSplitter(
    const ChunkSourceSplitterConfig& config,
    const StageRegistry& existing_stages)
    : SingleInputStage<ChunkSourceSplitterConfig, InputType>(config,
                                                             existing_stages) {
  if (config.output().empty()) {
    throw std::runtime_error("ChunkSourceSplitter requires at least 1 output.");
  }

  // Validate parallel arrays have same size.
  if (config.output_size() != config.weight_size()) {
    throw std::runtime_error(absl::StrCat(
        "ChunkSourceSplitter output and weight arrays must have same size: ",
        config.output_size(), " vs ", config.weight_size()));
  }

  // Create output queues from parallel arrays.
  outputs_.reserve(config.output_size());
  for (size_t i = 0; i < static_cast<size_t>(config.output_size()); ++i) {
    const auto& queue_cfg = config.output(static_cast<int>(i));
    const uint64_t weight = i < static_cast<size_t>(config.weight_size())
                                ? config.weight(static_cast<int>(i))
                                : 1;

    if (absl::c_any_of(outputs_, [&](const auto& existing_out) {
          return existing_out->name == queue_cfg.name();
        })) {
      throw std::runtime_error(std::string(absl::StrCat(
          "Duplicate output name in ChunkSourceSplitter: ", queue_cfg.name())));
    }

    auto* out = outputs_
                    .emplace_back(std::make_unique<Output>(
                        queue_cfg.name(), weight, queue_cfg.queue_capacity(),
                        ToOverflowBehavior(queue_cfg.overflow_behavior())))
                    .get();
    LOG(INFO) << "ChunkSourceSplitter configured output '" << out->name
              << "' weight=" << out->weight
              << " capacity=" << queue_cfg.queue_capacity();
  }

  // Precompute cumulative weights for fast assignment.
  cumulative_.resize(outputs_.size());
  std::transform_inclusive_scan(
      outputs_.begin(), outputs_.end(), cumulative_.begin(),
      std::plus<uint64_t>{},
      [](const std::unique_ptr<Output>& out) { return out->weight; });

  // Validate total weight is positive.
  if (cumulative_.back() == 0) {
    throw std::runtime_error(
        "ChunkSourceSplitter requires at least one output with positive "
        "weight.");
  }
}

ChunkSourceSplitter::~ChunkSourceSplitter() { Stop(); }

void ChunkSourceSplitter::Start() {
  LOG(INFO) << "Starting ChunkSourceSplitter worker.";
  worker_ = std::jthread([this]() { Worker(); });
}

void ChunkSourceSplitter::Stop() {
  bool expected = false;
  if (!stop_requested_.compare_exchange_strong(expected, true)) {
    return;
  }
  LOG(INFO) << "Stopping ChunkSourceSplitter.";
  input_queue()->Close();
  for (auto& out : outputs_) out->queue.Close();
  if (worker_.joinable()) worker_.join();
  LOG(INFO) << "ChunkSourceSplitter stopped.";
}

QueueBase* ChunkSourceSplitter::GetOutput(std::string_view name) {
  auto iter = absl::c_find_if(
      outputs_, [&](const auto& out) { return out->name == name; });
  if (iter == outputs_.end()) {
    throw std::runtime_error(
        absl::StrCat("Unknown output '", name, "' for ChunkSourceSplitter."));
  }
  return &(*iter)->queue;
}

StageMetricProto ChunkSourceSplitter::FlushMetrics() {
  StageMetricProto metric;
  metric.set_stage_type("chunk_source_splitter");
  for (auto& out : outputs_) {
    *metric.add_queue_metrics() = MetricsFromQueue(out->name, out->queue);
  }
  return metric;
}

void ChunkSourceSplitter::Worker() {
  // Create producers for each output in this thread.
  std::vector<Queue<OutputType>::Producer> producers;
  producers.reserve(outputs_.size());
  for (auto& out : outputs_) {
    producers.emplace_back(out->queue.CreateProducer());
  }

  try {
    while (true) {
      InputType item = input_queue()->Get();
      if (item.message_type ==
          FilePathProvider::MessageType::kInitialScanComplete) {
        // Broadcast to all outputs.
        for (auto& prod : producers) {
          prod.Put(
              OutputType{.source = nullptr, .message_type = item.message_type});
        }
        continue;
      }

      // Share ownership of the ChunkSource with any produced views.
      std::shared_ptr<ChunkSource> shared_source(std::move(item.source));

      auto per_output_indices = BuildAssignments(shared_source);
      // Emit only non-empty views, preserving original message type (kFile).
      for (size_t i = 0; i < outputs_.size(); ++i) {
        if (per_output_indices[i].empty()) continue;
        auto view = std::make_unique<ChunkSourceView>(
            shared_source, std::move(per_output_indices[i]));
        producers[i].Put(
            OutputType{.source = std::move(view),
                       .message_type = FilePathProvider::MessageType::kFile});
      }
    }
  } catch (const QueueClosedException&) {
    // Input queue closed — producers will close queues automatically when
    // destroyed if this thread holds the last producer.
    LOG(INFO) << "ChunkSourceSplitter worker exiting: input closed.";
  }
}

std::vector<std::vector<uint32_t>> ChunkSourceSplitter::BuildAssignments(
    const std::shared_ptr<ChunkSource>& source) {
  const std::string sort_key = source->GetChunkSortKey();
  const size_t n = source->GetChunkCount();

  // Prepare result containers with a rough reservation.
  std::vector<std::vector<uint32_t>> indices(outputs_.size());

  for (size_t i = 0; i < n; ++i) {
    const uint64_t h =
        static_cast<uint64_t>(absl::Hash<std::pair<std::string, size_t>>{}(
            std::make_pair(sort_key, i)));
    const uint64_t r = h % cumulative_.back();
    // Find the output where cumulative[j-1] <= r < cumulative[j].
    const auto it = std::upper_bound(cumulative_.begin(), cumulative_.end(), r);
    const size_t idx = it - cumulative_.begin();
    indices[idx].push_back(static_cast<uint32_t>(i));
  }

  return indices;
}

}  // namespace training
}  // namespace lczero
