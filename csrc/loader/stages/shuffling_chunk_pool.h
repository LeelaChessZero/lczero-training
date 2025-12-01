#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/random.h"
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

class ShufflingChunkPool : public Stage {
 public:
  explicit ShufflingChunkPool(const ShufflingChunkPoolConfig& config);
  ~ShufflingChunkPool();

  void Start() override;
  void Stop() override;
  void SetInputs(absl::Span<QueueBase* const> inputs) override;
  QueueBase* GetOutput(std::string_view name) override;

  StageMetricProto FlushMetrics() override;

  std::optional<StageControlResponse> Control(
      const StageControlRequest& request) override;

  // Anchor management methods for tracking chunks since a specific point.
  std::pair<std::string, int> ResetAnchor();
  int ChunksSinceAnchor();
  std::string CurrentAnchor();
  void SetAnchor(std::string_view anchor);

  Queue<ChunkSourceWithPhase>* input_queue() { return primary_input_queue_; }
  Queue<TrainingChunk>* output_queue() { return &primary_output_queue_; }

 private:
  struct CacheNode {
    FrameType frame;
    std::unique_ptr<CacheNode> next;
  };

  struct ChunkSourceItem {
    size_t start_chunk_index;
    std::unique_ptr<ChunkSource> source;
    absl::flat_hash_set<size_t> dropped_chunks;
    // Per-chunk counters and cached weights.
    std::vector<uint16_t> use_counts;
    std::vector<float> weight;
    std::vector<std::unique_ptr<CacheNode>> cache;
  };

  struct SourceIngestionThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  struct ChunkLoadingThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  struct CachingThreadContext {
    LoadMetricUpdater load_metric_updater;
  };

  std::vector<std::unique_ptr<ChunkSource>> InitializeChunkSources();
  void ProcessInputFiles(
      std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources);
  void SourceIngestionWorker(std::stop_token stop_token,
                             SourceIngestionThreadContext* context);
  void OutputWorker(std::stop_token stop_token,
                    ChunkLoadingThreadContext* context);
  void CachingWorker(std::stop_token stop_token, CachingThreadContext* context);
  void AddNewChunkSource(std::unique_ptr<ChunkSource> source)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(chunk_sources_mutex_);
  std::optional<std::variant<TrainingChunk, FrameType>> GetNextChunkData()
      ABSL_LOCKS_EXCLUDED(chunk_sources_mutex_);

  enum class ChunkStatus { kOk, kRetry, kEnd };
  struct ChunkData;

  ChunkStatus GetChunkInfo(ChunkData& out_chunk_data)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(chunk_sources_mutex_);
  bool LoadChunkData(ChunkData& chunk_data)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(chunk_sources_mutex_);
  bool HanseAccept(ChunkData& chunk_data)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(chunk_sources_mutex_);
  float ComputeChunkWeight(absl::Span<const FrameType> frames);
  double ComputeHanseProbability(float weight);

  Queue<ChunkSourceWithPhase>* primary_input_queue_ = nullptr;
  Queue<CacheRequest>* cache_request_queue_ = nullptr;
  std::string primary_output_name_;
  Queue<TrainingChunk> primary_output_queue_;
  std::optional<std::string> cachehit_output_name_;
  std::optional<Queue<FrameType>> cachehit_output_queue_;

  const size_t chunk_pool_size_;
  const ShufflingChunkPoolConfig config_;
  // stop_source_ must be declared before ThreadPools that reference it.
  std::stop_source stop_source_;
  ThreadPool source_ingestion_pool_;
  ThreadPool chunk_loading_pool_;
  ThreadPool caching_pool_;

  std::atomic<int64_t> dropped_chunks_metric_{0};

  absl::Mutex chunk_sources_mutex_;
  std::deque<ChunkSourceItem> chunk_sources_
      ABSL_GUARDED_BY(chunk_sources_mutex_);
  StreamShuffler stream_shuffler_ ABSL_GUARDED_BY(chunk_sources_mutex_);
  float max_weight_ ABSL_GUARDED_BY(chunk_sources_mutex_) = 0.0f;
  std::jthread initialization_thread_;
  std::vector<std::unique_ptr<SourceIngestionThreadContext>>
      source_ingestion_thread_contexts_;
  std::vector<std::unique_ptr<ChunkLoadingThreadContext>>
      chunk_loading_thread_contexts_;
  std::vector<std::unique_ptr<CachingThreadContext>> caching_thread_contexts_;

  // Anchor-related members for tracking chunks since a specific point.
  absl::Mutex anchor_mutex_;
  std::string anchor_ ABSL_GUARDED_BY(anchor_mutex_);
  std::atomic<int> chunks_since_anchor_{0};

  // Thread-local RNG for Hanse sampling.
  static thread_local absl::BitGen bitgen_;

  // Metrics counters.
  std::atomic<uint64_t> hanse_cache_hits_{0};
  std::atomic<uint64_t> hanse_cache_misses_{0};
  std::atomic<uint64_t> hanse_rejected_{0};
  std::atomic<uint64_t> reshuffles_{0};
  std::atomic<uint64_t> cache_hits_{0};
  std::atomic<uint64_t> cache_misses_{0};
  std::atomic<uint64_t> mismatched_use_counts_{0};
  std::atomic<uint64_t> newly_cached_{0};
  std::atomic<uint64_t> dropped_cache_positions_{0};
  std::atomic<uint64_t> chunk_source_not_found_{0};
  std::atomic<uint64_t> cached_positions_{0};

  StatisticsProtoDouble chunk_weight_stats_
      ABSL_GUARDED_BY(chunk_sources_mutex_);
};

}  // namespace training
}  // namespace lczero
