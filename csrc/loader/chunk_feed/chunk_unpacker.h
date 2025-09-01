// ABOUTME: Stage that unpacks chunks into V6TrainingData frames.
// ABOUTME: Converts stream of std::string chunks to V6TrainingData stream.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

using FrameType = V6TrainingData;

// Worker pool that unpacks chunks into frames.
// Takes std::string chunks containing packed V6TrainingData as input and
// outputs individual V6TrainingData frames.
class ChunkUnpacker {
 public:
  using InputType = std::string;
  using OutputType = FrameType;

  ChunkUnpacker(Queue<InputType>* input_queue,
                const ChunkUnpackerConfig& config);
  ~ChunkUnpacker();

  Queue<OutputType>* output();
  ChunkUnpackerMetricsProto FlushMetrics();

 private:
  struct ThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  void Worker(ThreadContext* context);

  Queue<InputType>* input_queue_;
  Queue<OutputType> output_queue_;
  // thread_contexts_ must be declared before thread_pool_ to ensure
  // thread_pool_ is destroyed first (stopping threads before contexts).
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  ThreadPool thread_pool_;
};

}  // namespace training
}  // namespace lczero