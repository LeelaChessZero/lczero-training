#include "loader/chunk_feed/chunk_unpacker.h"

#include <cstring>

#include "absl/log/log.h"
#include "loader/data_loader_metrics.h"
#include "proto/training_config.pb.h"
#include "proto/training_metrics.pb.h"

namespace lczero {
namespace training {

ChunkUnpacker::ChunkUnpacker(Queue<InputType>* input_queue,
                             const ChunkUnpackerConfig& config)
    : input_queue_(input_queue),
      output_queue_(config.queue_capacity()),
      thread_pool_(config.threads(), ThreadPoolOptions{}) {
  LOG(INFO) << "Starting ChunkUnpacker with " << config.threads()
            << " worker threads";

  // Initialize thread contexts and start worker threads.
  thread_contexts_.reserve(config.threads());
  for (size_t i = 0; i < config.threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
    thread_pool_.Enqueue([this, i]() { Worker(thread_contexts_[i].get()); });
  }
}

Queue<ChunkUnpacker::OutputType>* ChunkUnpacker::output() {
  return &output_queue_;
}

void ChunkUnpacker::Worker(ThreadContext* context) {
  // Create a local producer for this worker thread.
  auto producer = output_queue_.CreateProducer();

  try {
    while (true) {
      auto chunk = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue_->Get();
      }();

      // Check if chunk size is valid for V6TrainingData frames.
      if (chunk.size() % sizeof(V6TrainingData) != 0) {
        LOG(WARNING) << "Chunk size " << chunk.size()
                     << " is not a multiple of V6TrainingData size "
                     << sizeof(V6TrainingData) << ", skipping chunk";
        continue;
      }

      size_t num_frames = chunk.size() / sizeof(V6TrainingData);
      const char* data = chunk.data();

      // Unpack each frame from the chunk.
      for (size_t i = 0; i < num_frames; ++i) {
        V6TrainingData frame;
        std::memcpy(&frame, data + i * sizeof(V6TrainingData),
                    sizeof(V6TrainingData));
        LoadMetricPauser pauser(context->load_metric_updater);
        producer.Put(std::move(frame));
      }
    }
  } catch (const QueueClosedException&) {
    // Input queue is closed, the local producer will be destroyed when this
    // function exits which may close the output queue if this is the last
    // producer.
  }
}

ChunkUnpackerMetricsProto ChunkUnpacker::FlushMetrics() {
  ChunkUnpackerMetricsProto result;
  for (const auto& context : thread_contexts_) {
    UpdateFrom(*result.mutable_load(),
               context->load_metric_updater.FlushMetrics());
  }
  *result.mutable_queue() = MetricsFromQueue(output_queue_);
  return result;
}

}  // namespace training
}  // namespace lczero