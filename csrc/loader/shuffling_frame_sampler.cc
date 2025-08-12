#include "loader/shuffling_frame_sampler.h"

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/random/uniform_int_distribution.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {

ShufflingFrameSampler::ShufflingFrameSampler(
    Queue<InputType>* input_queue, const ShufflingFrameSamplerConfig& config)
    : input_queue_(input_queue),
      output_queue_(config.output_queue_size()),
      thread_pool_(config.num_worker_threads(), ThreadPoolOptions{}),
      reservoir_size_per_thread_(config.reservoir_size_per_thread()) {
  LOG(INFO) << "Starting ShufflingFrameSampler with "
            << config.num_worker_threads() << " threads, reservoir size "
            << config.reservoir_size_per_thread();
  // Start the worker threads.
  for (size_t i = 0; i < config.num_worker_threads(); ++i) {
    thread_pool_.Enqueue([this]() { Worker(); });
  }
}

Queue<ShufflingFrameSampler::OutputType>* ShufflingFrameSampler::output() {
  return &output_queue_;
}

void ShufflingFrameSampler::Worker() {
  // Create producer early so that if input queue closes during reservoir
  // prefilling, the producer will be destroyed and close the output queue.
  auto producer = output_queue_.CreateProducer();
  absl::FixedArray<FrameType> reservoir(reservoir_size_per_thread_);

  try {
    // Phase 1: Prefill the reservoir
    LOG(INFO) << "ShufflingFrameSampler worker prefilling reservoir";
    absl::c_generate(reservoir, [this]() { return input_queue_->Get(); });

    // Phase 2: Main sampling loop
    MainSamplingLoop(reservoir, producer);
  } catch (const QueueClosedException&) {
    // Input queue is closed.
  }
}

void ShufflingFrameSampler::MainSamplingLoop(
    absl::FixedArray<FrameType>& reservoir,
    Queue<OutputType>::Producer& producer) {
  absl::uniform_int_distribution<size_t> dist(0, reservoir.size() - 1);

  while (true) {
    const size_t random_index = dist(gen_);
    producer.Put(std::move(reservoir[random_index]));
    reservoir[random_index] = input_queue_->Get();
  }
}

}  // namespace training
}  // namespace lczero