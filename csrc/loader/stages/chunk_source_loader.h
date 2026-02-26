#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <stop_token>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "loader/chunk_source/chunk_source.h"
#include "loader/stages/file_path_provider.h"
#include "loader/stages/stage.h"
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
    const std::filesystem::path& filepath,
    ChunkSourceLoaderConfig::FrameFormat frame_format);

struct ChunkSourceWithPhase {
  std::unique_ptr<ChunkSource> source;
  FilePathProvider::MessageType message_type;
};

// Worker pool that converts FilePathProvider output to ChunkSource objects.
// Takes FilePathProvider::File as input and outputs ChunkSourceWithPhase.
class ChunkSourceLoader
    : public SingleInputStage<ChunkSourceLoaderConfig, FilePathProvider::File>,
      public SingleOutputStage<ChunkSourceWithPhase> {
 public:
  using InputType = FilePathProvider::File;
  using OutputType = ChunkSourceWithPhase;

  explicit ChunkSourceLoader(const ChunkSourceLoaderConfig& config);
  ~ChunkSourceLoader();

  void Start() override;
  void Stop() override;

  StageMetricProto FlushMetrics() override;

 private:
  struct ThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  void Worker(std::stop_token stop_token, ThreadContext* context);
  ThreadPool thread_pool_;
  std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
  std::atomic<uint64_t> skipped_files_count_{0};
  absl::Mutex last_chunk_key_mutex_;
  std::string last_chunk_key_;
  ChunkSourceLoaderConfig::FrameFormat frame_format_;

  // Synchronization for sentinel barrier.
  absl::Mutex phase_mutex_;
  int pre_sentinel_work_count_ ABSL_GUARDED_BY(phase_mutex_) = 0;
  bool sentinel_received_ ABSL_GUARDED_BY(phase_mutex_) = false;
};

}  // namespace training
}  // namespace lczero
