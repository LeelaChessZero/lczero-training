// ABOUTME: Stage that provides shuffled frames using reservoir sampling.
// ABOUTME: Takes FrameType frames and outputs them in randomized order.
#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <stop_token>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/random/random.h"
#include "loader/data_loader_metrics.h"
#include "loader/frame_type.h"
#include "loader/stages/stage.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

// Worker that implements reservoir sampling for training frames.
// Takes FrameType frames as input and outputs them in shuffled order
// using reservoir sampling algorithm.
class ShufflingFrameSampler
    : public SingleInputStage<ShufflingFrameSamplerConfig, FrameType>,
      public SingleOutputStage<FrameType> {
 public:
  using InputType = FrameType;
  using OutputType = FrameType;

  explicit ShufflingFrameSampler(const ShufflingFrameSamplerConfig& config);
  ~ShufflingFrameSampler();

  void Start() override;
  void Stop() override;
  StageMetricProto FlushMetrics() override;

 private:
  struct ThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  void Worker(std::stop_token stop_token, ThreadContext* context);
  void MainSamplingLoop(std::stop_token stop_token,
                        absl::FixedArray<FrameType>& reservoir,
                        Queue<OutputType>::Producer& producer,
                        ThreadContext* context);

  size_t reservoir_size_per_thread_;
  absl::BitGen gen_;
  // thread_contexts_ must be declared before thread_pool_ to ensure
  // thread_pool_ is destroyed first (stopping threads before contexts).
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  ThreadPool thread_pool_;
};

}  // namespace training
}  // namespace lczero
