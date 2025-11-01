#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>

#include "absl/random/random.h"
#include "loader/chunk_source/chunk_source.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/stage.h"
#include "loader/stages/training_chunk.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

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
  void Worker(std::stop_token stop_token);
  void ProcessSource(Queue<TrainingChunk>::Producer& producer,
                     std::unique_ptr<ChunkSource> source,
                     std::stop_token stop_token);
  std::optional<TrainingChunk> LoadChunk(ChunkSource& source,
                                         const std::string& sort_key,
                                         size_t index);

  std::atomic<uint64_t> chunks_processed_{0};
  std::atomic<uint64_t> chunks_dropped_{0};
  std::atomic<uint64_t> sources_processed_{0};
  absl::BitGen bitgen_;
  ThreadPool thread_pool_{1};
};

}  // namespace training
}  // namespace lczero
