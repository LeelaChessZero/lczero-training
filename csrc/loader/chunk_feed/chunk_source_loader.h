#pragma once

#include <filesystem>
#include <memory>

#include "loader/chunk_feed/chunk_source.h"
#include "loader/chunk_feed/file_path_provider.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/metrics/load_metric.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

// Creates a ChunkSource based on file extension. Returns RawFileChunkSource for
// .gz files, TarChunkSource for .tar files, or nullptr for unsupported types.
std::unique_ptr<ChunkSource> CreateChunkSourceFromFile(
    const std::filesystem::path& filepath);

struct ChunkSourceWithPhase {
  std::unique_ptr<ChunkSource> source;
  FilePathProvider::MessageType message_type;
};

// Worker pool that converts FilePathProvider output to ChunkSource objects.
// Takes FilePathProvider::File as input and outputs ChunkSourceWithPhase.
class ChunkSourceLoader {
 public:
  using InputType = FilePathProvider::File;
  using OutputType = ChunkSourceWithPhase;

  ChunkSourceLoader(Queue<InputType>* input_queue,
                    const ChunkSourceLoaderConfig& config);
  ~ChunkSourceLoader();

  Queue<OutputType>* output();

  ChunkSourceLoaderMetricsProto FlushMetrics();

 private:
  struct ThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  void Worker(ThreadContext* context);

  Queue<InputType>* input_queue_;
  Queue<OutputType> output_queue_;
  ThreadPool thread_pool_;
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  std::atomic<uint64_t> skipped_files_count_{0};
};

}  // namespace training
}  // namespace lczero