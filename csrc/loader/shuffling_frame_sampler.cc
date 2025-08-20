#include "loader/shuffling_frame_sampler.h"

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/random/uniform_int_distribution.h"
#include "loader/data_loader_metrics.h"
#include "proto/training_config.pb.h"
#include "proto/training_metrics.pb.h"

namespace lczero {
namespace training {

ShufflingFrameSampler::ShufflingFrameSampler(
    Queue<InputType>* input_queue, const ShufflingFrameSamplerConfig& config)
    : input_queue_(input_queue),
      output_queue_(config.output_queue_size()),
      reservoir_size_per_thread_(config.reservoir_size_per_thread()),
      thread_pool_(config.num_worker_threads(), ThreadPoolOptions{}) {
  LOG(INFO) << "Starting ShufflingFrameSampler with "
            << config.num_worker_threads() << " threads, reservoir size "
            << config.reservoir_size_per_thread();

  // Initialize thread contexts and start worker threads.
  thread_contexts_.reserve(config.num_worker_threads());
  for (size_t i = 0; i < config.num_worker_threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
    thread_pool_.Enqueue([this, i]() { Worker(thread_contexts_[i].get()); });
  }
}

Queue<ShufflingFrameSampler::OutputType>* ShufflingFrameSampler::output() {
  return &output_queue_;
}

void ShufflingFrameSampler::Worker(ThreadContext* context) {
  // Create producer early so that if input queue closes during reservoir
  // prefilling, the producer will be destroyed and close the output queue.
  auto producer = output_queue_.CreateProducer();
  absl::FixedArray<FrameType> reservoir(reservoir_size_per_thread_);

  try {
    // Phase 1: Prefill the reservoir
    LOG(INFO) << "ShufflingFrameSampler worker prefilling reservoir";
    absl::c_generate(reservoir, [this, context]() {
      LoadMetricPauser pauser(context->load_metric_updater);
      return input_queue_->Get();
    });

    // Phase 2: Main sampling loop
    MainSamplingLoop(reservoir, producer, context);
  } catch (const QueueClosedException&) {
    // Input queue is closed.
  }
}

void ShufflingFrameSampler::MainSamplingLoop(
    absl::FixedArray<FrameType>& reservoir,
    Queue<OutputType>::Producer& producer, ThreadContext* context) {
  absl::uniform_int_distribution<size_t> dist(0, reservoir.size() - 1);

  while (true) {
    const size_t random_index = dist(gen_);
    {
      LoadMetricPauser pauser(context->load_metric_updater);
      producer.Put(std::move(reservoir[random_index]));
    }
    {
      LoadMetricPauser pauser(context->load_metric_updater);
      reservoir[random_index] = input_queue_->Get();
    }
  }
}

ShufflingFrameSamplerMetricsProto ShufflingFrameSampler::FlushMetrics() {
  ShufflingFrameSamplerMetricsProto result;
  for (const auto& context : thread_contexts_) {
    UpdateFrom(*result.mutable_load(),
               context->load_metric_updater.FlushMetrics());
  }
  *result.mutable_queue() = MetricsFromQueue(output_queue_);
  return result;
}

}  // namespace training
}  // namespace lczero