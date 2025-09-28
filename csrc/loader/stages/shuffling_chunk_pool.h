#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "loader/chunk_source/chunk_source.h"
#include "loader/data_loader_metrics.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/stage.h"
#include "loader/stages/training_chunk.h"
#include "proto/data_loader_config.pb.h"
#include "proto/stage_control.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/metrics/load_metric.h"
#include "utils/queue.h"
#include "utils/stream_shuffler.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

class ShufflingChunkPool
    : public SingleInputStage<ShufflingChunkPoolConfig, ChunkSourceWithPhase> {
 public:
  ShufflingChunkPool(const ShufflingChunkPoolConfig& config,
                     const StageList& existing_stages);
  ~ShufflingChunkPool();

  Queue<TrainingChunk>* output();
  void Start() override;
  void Stop() override;

  StageMetricProto FlushMetrics() override;

  QueueBase* GetOutput(std::string_view name = "") override;
  std::optional<StageControlResponse> Control(
      const StageControlRequest& request) override;

  // Anchor management methods for tracking chunks since a specific point.
  std::pair<std::string, int> ResetAnchor();
  int ChunksSinceAnchor();
  std::string CurrentAnchor();
  void SetAnchor(std::string_view anchor);

 private:
  struct ChunkSourceItem {
    size_t start_chunk_index;
    std::unique_ptr<ChunkSource> source;
    absl::flat_hash_set<size_t> dropped_chunks;
    uint32_t reshuffle_count = 0;
  };

  struct IndexingThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  struct ChunkLoadingThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  std::vector<std::unique_ptr<ChunkSource>> InitializeChunkSources(
      size_t startup_indexing_threads);
  void ProcessInputFiles(
      std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources);
  void IndexingWorker(IndexingThreadContext* context);
  void OutputWorker(ChunkLoadingThreadContext* context);
  void AddNewChunkSource(std::unique_ptr<ChunkSource> source)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(chunk_sources_mutex_);
  std::optional<TrainingChunk> GetNextChunkData()
      ABSL_LOCKS_EXCLUDED(chunk_sources_mutex_);

  const size_t chunk_pool_size_;
  const ShufflingChunkPoolConfig config_;
  ThreadPool indexing_pool_;
  ThreadPool chunk_loading_pool_;
  Queue<TrainingChunk> output_queue_;

  std::atomic<int64_t> dropped_chunks_metric_{0};

  absl::Mutex chunk_sources_mutex_;
  std::deque<ChunkSourceItem> chunk_sources_
      ABSL_GUARDED_BY(chunk_sources_mutex_);
  StreamShuffler stream_shuffler_ ABSL_GUARDED_BY(chunk_sources_mutex_);
  std::jthread initialization_thread_;
  std::vector<std::unique_ptr<IndexingThreadContext>> indexing_thread_contexts_;
  std::vector<std::unique_ptr<ChunkLoadingThreadContext>>
      chunk_loading_thread_contexts_;

  // Anchor-related members for tracking chunks since a specific point.
  absl::Mutex anchor_mutex_;
  std::string anchor_ ABSL_GUARDED_BY(anchor_mutex_);
  std::atomic<int> chunks_since_anchor_{0};
  std::atomic<bool> stop_requested_{false};
};

}  // namespace training
}  // namespace lczero
