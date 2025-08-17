#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "loader/chunk_feed/chunk_source.h"
#include "loader/chunk_feed/chunk_source_loader.h"
#include "loader/data_loader_metrics.h"
#include "proto/training_config.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/metrics/load_metric.h"
#include "utils/queue.h"
#include "utils/stream_shuffler.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

class ShufflingChunkPool {
 public:
  ShufflingChunkPool(Queue<ChunkSourceWithPhase>* input_queue,
                     const ShufflingChunkPoolConfig& config);
  ~ShufflingChunkPool();

  Queue<std::string>* output();
  void Close();

  ShufflingChunkPoolMetricsProto FlushMetrics();

 private:
  struct ChunkSourceItem {
    size_t start_chunk_index;
    std::unique_ptr<ChunkSource> source;
  };

  struct IndexingThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  struct ChunkLoadingThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  std::vector<std::unique_ptr<ChunkSource>> InitializeChunkSources(
      size_t num_startup_indexing_threads);
  void ProcessInputFiles(
      std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources);
  void IndexingWorker(IndexingThreadContext* context);
  void OutputWorker(ChunkLoadingThreadContext* context);
  void AddNewChunkSource(std::unique_ptr<ChunkSource> source)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(chunk_sources_mutex_);
  std::string GetNextChunkData() ABSL_LOCKS_EXCLUDED(chunk_sources_mutex_);

  const size_t chunk_pool_size_;
  ThreadPool indexing_pool_;
  ThreadPool chunk_loading_pool_;
  Queue<ChunkSourceWithPhase>* input_queue_;
  Queue<std::string> output_queue_;

  absl::Mutex chunk_sources_mutex_;
  std::deque<ChunkSourceItem> chunk_sources_
      ABSL_GUARDED_BY(chunk_sources_mutex_);
  StreamShuffler stream_shuffler_ ABSL_GUARDED_BY(chunk_sources_mutex_);
  std::jthread initialization_thread_;
  std::vector<std::unique_ptr<IndexingThreadContext>> indexing_thread_contexts_;
  std::vector<std::unique_ptr<ChunkLoadingThreadContext>>
      chunk_loading_thread_contexts_;
};

}  // namespace training
}  // namespace lczero