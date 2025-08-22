// ABOUTME: Stage that converts V6TrainingData frames into tensor batches.
// ABOUTME: Produces TensorTuple with tensors for training pipeline.
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"
#include "utils/tensor.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

using FrameType = V6TrainingData;

// Worker pool that converts V6TrainingData frames into tensor batches.
// Takes individual V6TrainingData frames as input and outputs TensorTuple
// containing batched tensors in the format required for training.
class TensorGenerator {
 public:
  using InputType = FrameType;
  using OutputType = TensorTuple;

  TensorGenerator(Queue<InputType>* input_queue,
                  const TensorGeneratorConfig& config);

  Queue<OutputType>* output();
  TensorGeneratorMetricsProto FlushMetrics();

 private:
  struct ThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  void Worker(ThreadContext* context);
  void ConvertFramesToTensors(const std::vector<FrameType>& frames,
                              TensorTuple& tensors);
  void ProcessPlanes(const std::vector<FrameType>& frames,
                     TypedTensor<float>& planes_tensor);

  Queue<InputType>* input_queue_;
  Queue<OutputType> output_queue_;
  size_t batch_size_;
  // thread_contexts_ must be declared before thread_pool_ to ensure
  // thread_pool_ is destroyed first (stopping threads before contexts).
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  ThreadPool thread_pool_;
};

}  // namespace training
}  // namespace lczero