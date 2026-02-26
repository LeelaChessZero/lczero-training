// ABOUTME: Stage that unpacks chunks into FrameType frames.
// ABOUTME: Converts stream of std::string chunks to FrameType stream.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <stop_token>
#include <vector>

#include "absl/random/random.h"
#include "absl/types/optional.h"
#include "loader/data_loader_metrics.h"
#include "loader/frame_type.h"
#include "loader/stages/stage.h"
#include "loader/stages/training_chunk.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

// Worker pool that unpacks chunks into frames.
// Takes parsed TrainingChunk objects as input and outputs individual
// FrameType frames.
class ChunkUnpacker
    : public SingleInputStage<ChunkUnpackerConfig, TrainingChunk> {
 public:
  using InputType = TrainingChunk;

  explicit ChunkUnpacker(const ChunkUnpackerConfig& config);
  ~ChunkUnpacker();

  void Start() override;
  void Stop() override;
  QueueBase* GetOutput(std::string_view name) override;
  StageMetricProto FlushMetrics() override;

  Queue<FrameType>* output_queue() { return &primary_output_queue_; }

 private:
  struct ThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  void Worker(std::stop_token stop_token, ThreadContext* context);

  const ChunkUnpackerConfig config_;
  const uint32_t run_seed_;
  Queue<FrameType> primary_output_queue_;
  std::optional<Queue<CacheRequest>> prefetch_output_queue_;
  // thread_contexts_ must be declared before thread_pool_ to ensure
  // thread_pool_ is destroyed first (stopping threads before contexts).
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  ThreadPool thread_pool_;
};

std::vector<uint32_t> PickSampledPositions(int32_t n, double p,
                                           int32_t iteration,
                                           absl::BitGen& gen);

}  // namespace training
}  // namespace lczero
