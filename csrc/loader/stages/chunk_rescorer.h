// ABOUTME: Stage that rescales training chunks using Syzygy tablebases.
// ABOUTME: Adjusts frame metadata by invoking the classic LCZero rescorer.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "libs/lc0/src/syzygy/syzygy.h"
#include "libs/lc0/src/trainingdata/rescorer.h"
#include "loader/stages/stage.h"
#include "loader/stages/training_chunk.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/metrics/load_metric.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

// Stage that takes TrainingChunk objects, applies tablebase-based rescoring and
// forwards the updated chunks downstream.
class ChunkRescorer
    : public SingleInputStage<ChunkRescorerConfig, TrainingChunk> {
 public:
  using InputType = TrainingChunk;
  using OutputType = TrainingChunk;
  using RescoreFn = std::function<std::vector<V6TrainingData>(
      std::vector<V6TrainingData>, SyzygyTablebase*, float, float, float, int)>;

  ChunkRescorer(const ChunkRescorerConfig& config,
                const StageList& existing_stages,
                RescoreFn rescore_fn = RescoreTrainingData);
  ~ChunkRescorer() override;

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
  void InitializeTablebase();

  SyzygyTablebase tablebase_;
  bool tablebase_initialized_ = false;
  std::string syzygy_paths_;
  float dist_temp_;
  float dist_offset_;
  float dtz_boost_;
  int new_input_format_;

  Queue<OutputType> output_queue_;
  ThreadPool thread_pool_;
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  std::atomic<bool> stop_requested_{false};
  RescoreFn rescore_fn_;
};

}  // namespace training
}  // namespace lczero
