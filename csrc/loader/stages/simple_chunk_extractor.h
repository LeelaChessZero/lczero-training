#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#include "absl/random/random.h"
#include "loader/chunk_source/chunk_source.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/stage.h"
#include "loader/stages/training_chunk.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

// Single-threaded stage that shuffles chunks within each source.
class SimpleChunkExtractor
    : public SingleInputStage<SimpleChunkExtractorConfig, ChunkSourceWithPhase>,
      public SingleOutputStage<TrainingChunk> {
 public:
  explicit SimpleChunkExtractor(const SimpleChunkExtractorConfig& config);
  ~SimpleChunkExtractor();

  void Start() override;
  void Stop() override;
  StageMetricProto FlushMetrics() override;

 private:
  void Worker();
  void ProcessSource(Queue<TrainingChunk>::Producer& producer,
                     std::unique_ptr<ChunkSource> source);
  std::optional<TrainingChunk> LoadChunk(ChunkSource& source,
                                         const std::string& sort_key,
                                         size_t index);

  std::jthread worker_thread_;
  std::atomic<bool> stop_requested_{false};
  std::atomic<uint64_t> chunks_processed_{0};
  std::atomic<uint64_t> chunks_dropped_{0};
  std::atomic<uint64_t> sources_processed_{0};
  absl::BitGen bitgen_;
};

}  // namespace training
}  // namespace lczero
