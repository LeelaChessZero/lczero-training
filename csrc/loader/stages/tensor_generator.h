// ABOUTME: Stage that converts FrameType frames into tensor batches.
// ABOUTME: Produces TrainingTensors with tensors for training pipeline.
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <stop_token>
#include <vector>

#include "loader/data_loader.h"
#include "loader/data_loader_metrics.h"
#include "loader/frame_type.h"
#include "loader/stages/stage.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"
#include "utils/tensor.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

// Worker pool that converts FrameType frames into tensor batches.
// Takes individual FrameType frames as input and outputs TensorTuple
// containing batched tensors in the format required for training.
class TensorGenerator
    : public SingleInputStage<TensorGeneratorConfig, FrameType>,
      public SingleOutputStage<TensorTuple> {
 public:
  using InputType = FrameType;
  using OutputType = TensorTuple;

  explicit TensorGenerator(const TensorGeneratorConfig& config);
  ~TensorGenerator();

  void Start() override;
  void Stop() override;
  StageMetricProto FlushMetrics() override;

 private:
  struct ThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  void Worker(std::stop_token stop_token, ThreadContext* context);
  TensorTuple ConvertFramesToTensors(const std::vector<FrameType>& frames);
  void ProcessPlanes(const std::vector<FrameType>& frames,
                     TypedTensor<float>& planes_tensor);

  size_t batch_size_;
  // thread_contexts_ must be declared before thread_pool_ to ensure
  // thread_pool_ is destroyed first (stopping threads before contexts).
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  ThreadPool thread_pool_;
};

}  // namespace training
}  // namespace lczero
