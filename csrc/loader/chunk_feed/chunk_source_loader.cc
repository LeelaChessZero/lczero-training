#include "loader/chunk_feed/chunk_source_loader.h"

#include <filesystem>

#include "absl/log/log.h"
#include "loader/chunk_feed/rawfile_chunk_source.h"
#include "loader/chunk_feed/tar_chunk_source.h"
#include "loader/data_loader_metrics.h"
#include "proto/training_config.pb.h"

namespace lczero {
namespace training {

std::unique_ptr<ChunkSource> CreateChunkSourceFromFile(
    const std::filesystem::path& filepath) {
  auto extension = filepath.extension();
  if (extension == ".gz") {
    return std::make_unique<RawFileChunkSource>(filepath);
  }
  if (extension == ".tar") {
    return std::make_unique<TarChunkSource>(filepath);
  }
  return nullptr;
}

ChunkSourceLoader::ChunkSourceLoader(Queue<InputType>* input_queue,
                                     const ChunkSourceLoaderConfig& config)
    : input_queue_(input_queue),
      output_queue_(config.output_queue_size()),
      thread_pool_(config.worker_threads(), ThreadPoolOptions{}) {
  LOG(INFO) << "Starting ChunkSourceLoader with " << config.worker_threads()
            << " worker threads";

  // Initialize thread contexts and start worker threads.
  thread_contexts_.reserve(config.worker_threads());
  for (size_t i = 0; i < config.worker_threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
    thread_pool_.Enqueue([this, i]() { Worker(thread_contexts_[i].get()); });
  }
}

Queue<ChunkSourceLoader::OutputType>* ChunkSourceLoader::output() {
  return &output_queue_;
}

void ChunkSourceLoader::Worker(ThreadContext* context) {
  // Create a local producer for this worker thread
  auto producer = output_queue_.CreateProducer();

  try {
    while (true) {
      auto file = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue_->Get();
      }();

      // Create ChunkSource from the file.
      auto source = CreateChunkSourceFromFile(file.filepath);
      if (source) {
        // Output the ChunkSource with its phase.
        ChunkSourceWithPhase output{.source = std::move(source),
                                    .message_type = file.message_type};
        LoadMetricPauser pauser(context->load_metric_updater);
        producer.Put(std::move(output));
      }
    }
  } catch (const QueueClosedException&) {
    // Input queue is closed, the local producer will be destroyed when this
    // function exits which may close the output queue if this is the last
    // producer
  }
}

ChunkSourceLoaderMetricsProto ChunkSourceLoader::FlushMetrics() {
  ChunkSourceLoaderMetricsProto result;
  for (const auto& context : thread_contexts_) {
    lczero::training::UpdateFrom(*result.mutable_load(),
                                 context->load_metric_updater.FlushMetrics());
  }
  // Get queue metrics.
  *result.mutable_queue() = MetricsFromQueue(output_queue_);

  return result;
}

}  // namespace training
}  // namespace lczero