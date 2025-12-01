#include "loader/stages/shuffling_frame_sampler.h"

#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/random/uniform_int_distribution.h"
#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"

namespace lczero {
namespace training {

ShufflingFrameSampler::ShufflingFrameSampler(
    const ShufflingFrameSamplerConfig& config)
    : SingleInputStage<ShufflingFrameSamplerConfig, InputType>(config),
      SingleOutputStage<OutputType>(config.output()),
      reservoir_size_per_thread_(config.reservoir_size_per_thread()),
      thread_pool_(config.threads(), ThreadPoolOptions{}) {
  LOG(INFO) << "Initializing ShufflingFrameSampler with " << config.threads()
            << " threads, reservoir size "
            << config.reservoir_size_per_thread();

  // Initialize thread contexts but don't start worker threads yet.
  thread_contexts_.reserve(config.threads());
  for (size_t i = 0; i < config.threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
  }
}

ShufflingFrameSampler::~ShufflingFrameSampler() { Stop(); }

void ShufflingFrameSampler::Start() {
  LOG(INFO) << "Starting ShufflingFrameSampler worker threads.";
  for (size_t i = 0; i < thread_contexts_.size(); ++i) {
    thread_pool_.Enqueue([this, i](std::stop_token stop_token) {
      Worker(stop_token, thread_contexts_[i].get());
    });
  }
}

void ShufflingFrameSampler::Stop() {
  if (thread_pool_.stop_token().stop_requested()) return;

  LOG(INFO) << "Stopping ShufflingFrameSampler.";
  thread_pool_.Shutdown();
  output_queue()->Close();
  LOG(INFO) << "ShufflingFrameSampler stopped.";
}

void ShufflingFrameSampler::Worker(std::stop_token stop_token,
                                   ThreadContext* context) {
  // Create producer early so that if input queue closes during reservoir
  // prefilling, the producer will be destroyed and close the output queue.
  auto producer = output_queue()->CreateProducer();
  absl::FixedArray<FrameType> reservoir(reservoir_size_per_thread_);

  try {
    // Phase 1: Prefill the reservoir
    LOG(INFO) << "ShufflingFrameSampler worker prefilling reservoir";
    absl::c_generate(reservoir, [this, context, stop_token]() {
      LoadMetricPauser pauser(context->load_metric_updater);
      return input_queue()->Get(stop_token);
    });

    // Phase 2: Main sampling loop
    MainSamplingLoop(stop_token, reservoir, producer, context);
  } catch (const QueueClosedException&) {
    LOG(INFO) << "ShufflingFrameSampler worker stopping, queue closed.";
  } catch (const QueueRequestCancelled&) {
    LOG(INFO) << "ShufflingFrameSampler worker stopping, request cancelled.";
  }
}

void ShufflingFrameSampler::MainSamplingLoop(
    std::stop_token stop_token, absl::FixedArray<FrameType>& reservoir,
    Queue<OutputType>::Producer& producer, ThreadContext* context) {
  absl::uniform_int_distribution<size_t> dist(0, reservoir.size() - 1);

  while (true) {
    const size_t random_index = dist(gen_);
    {
      LoadMetricPauser pauser(context->load_metric_updater);
      producer.Put(std::move(reservoir[random_index]), stop_token);
    }
    {
      LoadMetricPauser pauser(context->load_metric_updater);
      reservoir[random_index] = input_queue()->Get(stop_token);
    }
  }
}

StageMetricProto ShufflingFrameSampler::FlushMetrics() {
  StageMetricProto stage_metric;
  LoadMetricProto aggregated_load;
  aggregated_load.set_name("load");
  for (const auto& context : thread_contexts_) {
    UpdateFrom(aggregated_load, context->load_metric_updater.FlushMetrics());
  }
  *stage_metric.add_load_metrics() = std::move(aggregated_load);
  *stage_metric.add_queue_metrics() =
      MetricsFromQueue("output", *output_queue());
  return stage_metric;
}

}  // namespace training
}  // namespace lczero
