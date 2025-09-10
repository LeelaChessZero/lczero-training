// ABOUTME: Stage that provides shuffled frames using reservoir sampling.
// ABOUTME: Takes V6TrainingData frames and outputs them in randomized order.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/random/random.h"
#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

using FrameType = V6TrainingData;

// Worker that implements reservoir sampling for training frames.
// Takes V6TrainingData frames as input and outputs them in shuffled order
// using reservoir sampling algorithm.
class ShufflingFrameSampler {
 public:
  using InputType = FrameType;
  using OutputType = FrameType;

  ShufflingFrameSampler(Queue<InputType>* input_queue,
                        const ShufflingFrameSamplerConfig& config);
  ~ShufflingFrameSampler();

  Queue<OutputType>* output();
  void Start();
  ShufflingFrameSamplerMetricsProto FlushMetrics();

 private:
  struct ThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  void Worker(ThreadContext* context);
  void MainSamplingLoop(absl::FixedArray<FrameType>& reservoir,
                        Queue<OutputType>::Producer& producer,
                        ThreadContext* context);

  Queue<InputType>* input_queue_;
  Queue<OutputType> output_queue_;
  size_t reservoir_size_per_thread_;
  absl::BitGen gen_;
  // thread_contexts_ must be declared before thread_pool_ to ensure
  // thread_pool_ is destroyed first (stopping threads before contexts).
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  ThreadPool thread_pool_;
};

}  // namespace training
}  // namespace lczero