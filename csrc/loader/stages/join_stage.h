// ABOUTME: Stage that joins multiple input queues into a single output.
// ABOUTME: Spawns one thread per input to read and forward items.
#pragma once

#include <atomic>
#include <memory>
#include <stop_token>
#include <vector>

#include "loader/data_loader_metrics.h"
#include "loader/frame_type.h"
#include "loader/stages/stage.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

// Template stage that joins multiple input queues into a single output.
// Spawns one thread per input to consume and forward items.
template <typename T>
class JoinStage : public SingleOutputStage<T> {
 public:
  using OutputType = T;

  explicit JoinStage(const JoinPositionsConfig& config);
  ~JoinStage();

  void Start() override;
  void Stop() override;
  StageMetricProto FlushMetrics() override;
  void SetInputs(absl::Span<QueueBase* const> inputs) override;

 private:
  struct ThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  void Worker(std::stop_token stop_token, Queue<T>* input_queue,
              ThreadContext* context);

  std::vector<Queue<T>*> input_queues_;
  // thread_contexts_ must be declared before thread_pool_ to ensure
  // thread_pool_ is destroyed first (stopping threads before contexts).
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  std::unique_ptr<ThreadPool> thread_pool_;
};

using JoinPositions = JoinStage<FrameType>;

}  // namespace training
}  // namespace lczero
