// ABOUTME: Stage that unpacks chunks into V6TrainingData frames.
// ABOUTME: Converts stream of std::string chunks to V6TrainingData stream.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/random/random.h"
#include "absl/types/optional.h"
#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "loader/data_loader_metrics.h"
#include "loader/stages/stage.h"
#include "loader/stages/training_chunk.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

using FrameType = V6TrainingData;

// Worker pool that unpacks chunks into frames.
// Takes parsed TrainingChunk objects as input and outputs individual
// V6TrainingData frames.
class ChunkUnpacker
    : public SingleInputStage<ChunkUnpackerConfig, TrainingChunk> {
 public:
  using InputType = TrainingChunk;
  using OutputType = FrameType;

  ChunkUnpacker(const ChunkUnpackerConfig& config,
                const StageRegistry& existing_stages);
  ~ChunkUnpacker();

  Queue<OutputType>* output();
  void Start() override;
  void Stop() override;
  StageMetricProto FlushMetrics() override;

  QueueBase* GetOutput(std::string_view name = "") override;

 private:
  struct ThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  void Worker(ThreadContext* context);

  const absl::optional<float> position_sampling_rate_;
  const absl::optional<uint32_t> position_count_;
  const uint32_t run_seed_;
  Queue<OutputType> output_queue_;
  // thread_contexts_ must be declared before thread_pool_ to ensure
  // thread_pool_ is destroyed first (stopping threads before contexts).
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  ThreadPool thread_pool_;
  std::atomic<bool> stop_requested_{false};
};

std::vector<uint32_t> PickSampledPositions(int32_t n, double p,
                                           int32_t iteration,
                                           absl::BitGen& gen);

std::vector<uint32_t> PickFixedPositionCount(uint32_t n, uint32_t k,
                                             absl::BitGen& gen,
                                             uint32_t iteration);

}  // namespace training
}  // namespace lczero
