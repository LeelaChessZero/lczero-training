#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "loader/chunk_source/chunk_source.h"
#include "loader/chunk_source/chunk_source_view.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/stage.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

// Splits an incoming ChunkSource into several ChunkSourceViews based on a
// deterministic hash of (sort_key, index). Emits to multiple named outputs.
class ChunkSourceSplitter
    : public SingleInputStage<ChunkSourceSplitterConfig, ChunkSourceWithPhase> {
 public:
  using InputType = ChunkSourceWithPhase;
  using OutputType = ChunkSourceWithPhase;

  explicit ChunkSourceSplitter(const ChunkSourceSplitterConfig& config,
                               const StageRegistry& existing_stages);
  ~ChunkSourceSplitter();

  void Start() override;
  void Stop() override;

  StageMetricProto FlushMetrics() override;
  QueueBase* GetOutput(std::string_view name = "") override;

 private:
  struct Output {
    std::string name;
    uint64_t weight;
    Queue<OutputType> queue;
    Output(std::string_view name, uint64_t weight, size_t capacity,
           OverflowBehavior overflow)
        : name(name), weight(weight), queue(capacity, overflow) {}
  };

  void Worker();

  // Builds per-output indices given a source; uses absl::Hash on
  // (sort_key, index) and weights to assign indices.
  std::vector<std::vector<uint32_t>> BuildAssignments(
      const std::shared_ptr<ChunkSource>& source);

  std::vector<std::unique_ptr<Output>> outputs_;
  std::vector<uint64_t> cumulative_;
  std::jthread worker_;
  std::atomic<bool> stop_requested_{false};
};

}  // namespace training
}  // namespace lczero
