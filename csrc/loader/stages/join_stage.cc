#include "loader/stages/join_stage.h"

#include <absl/log/log.h>

#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/metrics/load_metric.h"

namespace lczero {
namespace training {

template <typename T>
JoinStage<T>::JoinStage(const JoinPositionsConfig& config)
    : SingleOutputStage<T>(config.output()) {}

template <typename T>
JoinStage<T>::~JoinStage() {
  Stop();
}

template <typename T>
void JoinStage<T>::SetInputs(absl::Span<QueueBase* const> inputs) {
  input_queues_.clear();
  for (QueueBase* base_queue : inputs) {
    auto* typed_queue = dynamic_cast<Queue<T>*>(base_queue);
    if (!typed_queue) throw std::runtime_error("Input queue type mismatch");
    input_queues_.push_back(typed_queue);
  }
}

template <typename T>
void JoinStage<T>::Start() {
  thread_contexts_.clear();
  threads_.clear();
  for (size_t i = 0; i < input_queues_.size(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
  }
  for (size_t i = 0; i < input_queues_.size(); ++i) {
    threads_.emplace_back(
        [this, i]() { Worker(input_queues_[i], thread_contexts_[i].get()); });
  }
}

template <typename T>
void JoinStage<T>::Worker(Queue<T>* input_queue, ThreadContext* context) {
  auto producer = this->output_queue()->CreateProducer();
  try {
    while (true) {
      auto item = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue->Get();
      }();
      producer.Put(std::move(item));
    }
  } catch (const QueueClosedException&) {
  }
}

template <typename T>
void JoinStage<T>::Stop() {
  for (auto& input_queue : input_queues_) input_queue->Close();
  for (auto& thread : threads_) {
    if (thread.joinable()) thread.join();
  }
  this->output_queue()->Close();
}

template <typename T>
StageMetricProto JoinStage<T>::FlushMetrics() {
  StageMetricProto metrics;
  LoadMetricProto aggregated_load;
  aggregated_load.set_name("load");
  for (const auto& context : thread_contexts_) {
    UpdateFrom(aggregated_load, context->load_metric_updater.FlushMetrics());
  }
  *metrics.add_load_metrics() = std::move(aggregated_load);
  *metrics.add_queue_metrics() =
      MetricsFromQueue("output", *this->output_queue());

  return metrics;
}

// Explicit template instantiation for FrameType.
template class JoinStage<FrameType>;

}  // namespace training
}  // namespace lczero
