#include "loader/stages/chunk_unpacker.h"

#include <cstring>

#include "absl/log/log.h"
#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"

namespace lczero {
namespace training {

ChunkUnpacker::ChunkUnpacker(const ChunkUnpackerConfig& config,
                             const StageList& existing_stages)
    : SingleInputStage<ChunkUnpackerConfig, InputType>(config, existing_stages),
      output_queue_(config.queue_capacity()),
      thread_pool_(config.threads(), ThreadPoolOptions{}) {
  LOG(INFO) << "Initializing ChunkUnpacker with " << config.threads()
            << " worker threads";

  // Initialize thread contexts but don't start worker threads yet.
  thread_contexts_.reserve(config.threads());
  for (size_t i = 0; i < config.threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
  }
}

ChunkUnpacker::~ChunkUnpacker() { Stop(); }

Queue<ChunkUnpacker::OutputType>* ChunkUnpacker::output() {
  return &output_queue_;
}

QueueBase* ChunkUnpacker::GetOutput(std::string_view name) {
  (void)name;
  return &output_queue_;
}

void ChunkUnpacker::Start() {
  LOG(INFO) << "Starting ChunkUnpacker worker threads.";
  for (size_t i = 0; i < thread_contexts_.size(); ++i) {
    thread_pool_.Enqueue([this, i]() { Worker(thread_contexts_[i].get()); });
  }
}

void ChunkUnpacker::Stop() {
  bool expected = false;
  if (!stop_requested_.compare_exchange_strong(expected, true)) {
    return;
  }

  LOG(INFO) << "Stopping ChunkUnpacker.";
  input_queue()->Close();
  thread_pool_.WaitAll();
  output_queue_.Close();
  LOG(INFO) << "ChunkUnpacker stopped.";
}

void ChunkUnpacker::Worker(ThreadContext* context) {
  // Create a local producer for this worker thread.
  auto producer = output_queue_.CreateProducer();

  try {
    while (true) {
      auto chunk = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue()->Get();
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
    LOG(INFO) << "ChunkUnpacker worker stopping, input queue closed.";
    // Input queue is closed, the local producer will be destroyed when this
    // function exits which may close the output queue if this is the last
    // producer.
  }
}

StageMetricProto ChunkUnpacker::FlushMetrics() {
  StageMetricProto stage_metric;
  auto* metrics = stage_metric.mutable_chunk_unpacker();
  for (const auto& context : thread_contexts_) {
    UpdateFrom(*metrics->mutable_load(),
               context->load_metric_updater.FlushMetrics());
  }
  *stage_metric.add_output_queue_metrics() =
      MetricsFromQueue("output", output_queue_);
  return stage_metric;
}

}  // namespace training
}  // namespace lczero
