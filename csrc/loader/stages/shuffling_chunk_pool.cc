#include "loader/stages/shuffling_chunk_pool.h"

#include <absl/algorithm/container.h>
#include <absl/base/thread_annotations.h>
#include <absl/log/log.h>
#include <absl/synchronization/mutex.h>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <thread>

#include "loader/chunk_source/chunk_source.h"
#include "loader/data_loader_metrics.h"
#include "loader/stages/chunk_source_loader.h"
#include "proto/data_loader_config.pb.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

ShufflingChunkPool::ShufflingChunkPool(Queue<ChunkSourceWithPhase>* input_queue,
                                       const ShufflingChunkPoolConfig& config)
    : chunk_pool_size_(config.chunk_pool_size()),
      config_(config),
      indexing_pool_(config.indexing_threads(), ThreadPoolOptions{}),
      chunk_loading_pool_(config.chunk_loading_threads(), ThreadPoolOptions{}),
      input_queue_(input_queue),
      output_queue_(config.queue_capacity()) {
  LOG(INFO) << "Initializing ShufflingChunkPool with pool size "
            << config.chunk_pool_size();
}

ShufflingChunkPool::~ShufflingChunkPool() {
  LOG(INFO) << "ShufflingChunkPool shutting down.";
  Close();
  if (initialization_thread_.joinable()) {
    LOG(INFO) << "Waiting for initialization thread to join...";
    initialization_thread_.join();
  }
  LOG(INFO) << "Waiting for indexing pool to finish...";
  indexing_pool_.WaitAll();
  LOG(INFO) << "Waiting for chunk loading pool to finish...";
  chunk_loading_pool_.WaitAll();
  LOG(INFO) << "ShufflingChunkPool shutdown complete.";
}

Queue<std::string>* ShufflingChunkPool::output() { return &output_queue_; }

void ShufflingChunkPool::Start() {
  LOG(INFO) << "Starting ShufflingChunkPool initialization thread.";
  initialization_thread_ = std::jthread([this]() {
    try {
      LOG(INFO) << "Starting ShufflingChunkPool with pool size "
                << config_.chunk_pool_size();
      std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources =
          InitializeChunkSources(config_.startup_indexing_threads());
      ProcessInputFiles(std::move(uninitialized_sources));

      // Start input processing worker that continuously processes new files.
      for (size_t i = 0; i < indexing_pool_.num_threads(); ++i) {
        auto* context =
            indexing_thread_contexts_
                .emplace_back(std::make_unique<IndexingThreadContext>())
                .get();
        indexing_pool_.Enqueue([this, context]() { IndexingWorker(context); });
      }

      // Start output workers after everything is fully initialized.
      LOG(INFO) << "ShufflingChunkPool initialization done, starting workers";
      for (size_t i = 0; i < chunk_loading_pool_.num_threads(); ++i) {
        auto* context =
            chunk_loading_thread_contexts_
                .emplace_back(std::make_unique<ChunkLoadingThreadContext>())
                .get();
        chunk_loading_pool_.Enqueue(
            [this, context]() { OutputWorker(context); });
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "ShufflingChunkPool initialization failed: " << e.what();
      output_queue_.Close();
    }
  });
}

std::vector<std::unique_ptr<ChunkSource>>
ShufflingChunkPool::InitializeChunkSources(size_t startup_indexing_threads) {
  std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources;

  // Read from input queue until kInitialScanComplete.
  while (true) {
    auto chunk_source_with_phase = input_queue_->Get();

    if (chunk_source_with_phase.message_type ==
        FilePathProvider::MessageType::kInitialScanComplete) {
      break;
    }

    if (chunk_source_with_phase.message_type ==
        FilePathProvider::MessageType::kFile) {
      // Add ChunkSource to uninitialized sources.
      uninitialized_sources.push_back(
          std::move(chunk_source_with_phase.source));
    }
  }

  // Sort in descending order (newest first).
  std::sort(uninitialized_sources.begin(), uninitialized_sources.end(),
            [](const auto& a, const auto& b) {
              return a->GetChunkSortKey() > b->GetChunkSortKey();
            });
  std::atomic<size_t> total_chunks = 0;
  size_t sources_to_keep = 0;

  ThreadPool indexing_pool(startup_indexing_threads);

  // Index sources ≈sequentially until we have enough chunks. It's fine to
  // overshoot a bit due to multiple threads.
  std::atomic<bool> anchor_encountered{false};
  std::string current_anchor;
  {
    absl::MutexLock lock(&anchor_mutex_);
    current_anchor = anchor_;
  }

  for (auto& source : uninitialized_sources) {
    indexing_pool.WaitForAvailableThread();
    if (output_queue_.IsClosed()) {
      LOG(INFO) << "Output queue closed, stopping indexing.";
      break;
    }
    if (total_chunks >= chunk_pool_size_) break;

    indexing_pool.Enqueue(
        [&source, &total_chunks, &anchor_encountered, &current_anchor, this]() {
          source->Index();
          size_t chunk_count = source->GetChunkCount();
          total_chunks += chunk_count;

          // Handle anchor counting during initial load
          if (current_anchor.empty()) {
            // No anchor set - count all chunks
            chunks_since_anchor_ += chunk_count;
          } else {
            // Anchor is set - special logic for backward processing
            std::string chunk_key = source->GetChunkSortKey();
            if (chunk_key == current_anchor) {
              // Reset counter when we encounter the anchor
              anchor_encountered = true;
              chunks_since_anchor_ = 0;
            } else if (!anchor_encountered) {
              // Only count chunks for sources newer than the anchor
              chunks_since_anchor_ += chunk_count;
            }
          }

          LOG_EVERY_N_SEC(INFO, 4) << "Loaded so far: " << total_chunks.load();
        });
    ++sources_to_keep;
  }
  indexing_pool.WaitAll();

  if (total_chunks < chunk_pool_size_ && !output_queue_.IsClosed()) {
    throw std::runtime_error(
        absl::StrCat("Not enough chunks to initialize ShufflingChunkPool: ",
                     total_chunks.load(), " < ", chunk_pool_size_));
  }

  // Trim the vector to only keep the sources we need.
  uninitialized_sources.resize(sources_to_keep);
  return uninitialized_sources;
}

void ShufflingChunkPool::ProcessInputFiles(
    std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources) {
  // Initialize chunk sources from the initial scan.
  {
    absl::MutexLock lock(&chunk_sources_mutex_);
    size_t start_chunk_index = 0;
    // Newest sources first, so we add in reverse order.
    std::for_each(
        uninitialized_sources.rbegin(), uninitialized_sources.rend(),
        [this, &start_chunk_index](auto& source) {
          chunk_sources_.push_back({.start_chunk_index = start_chunk_index,
                                    .source = std::move(source)});
          start_chunk_index += chunk_sources_.back().source->GetChunkCount();
        });

    // Initialize stream shuffler with the initial bounds.
    if (!chunk_sources_.empty()) {
      size_t total_chunks = chunk_sources_.back().start_chunk_index +
                            chunk_sources_.back().source->GetChunkCount();
      // Set bounds to provide the last chunk_pool_size_ chunks.
      size_t lower_bound =
          total_chunks > chunk_pool_size_ ? total_chunks - chunk_pool_size_ : 0;
      stream_shuffler_.SetLowerBound(lower_bound);
      stream_shuffler_.SetUpperBound(total_chunks);
    }
  }
}

void ShufflingChunkPool::IndexingWorker(IndexingThreadContext* context) {
  try {
    while (true) {
      auto chunk_source_with_phase = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue_->Get();
      }();

      if (chunk_source_with_phase.message_type ==
          FilePathProvider::MessageType::kFile) {
        // Index the new chunk source.
        auto source = std::move(chunk_source_with_phase.source);
        source->Index();

        // Handle anchor counting for new chunk sources
        std::string chunk_key = source->GetChunkSortKey();
        size_t chunk_count = source->GetChunkCount();
        {
          absl::MutexLock anchor_lock(&anchor_mutex_);
          if (!anchor_.empty() && chunk_key == anchor_) {
            // Reset counter when we encounter the anchor
            chunks_since_anchor_ = 0;
          } else {
            // Increment counter by the number of chunks in this source
            chunks_since_anchor_ += chunk_count;
          }
        }

        absl::MutexLock lock(&chunk_sources_mutex_);
        AddNewChunkSource(std::move(source));
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "Input queue closed, stopping input worker.";
  }
}

void ShufflingChunkPool::OutputWorker(ChunkLoadingThreadContext* context) {
  // Create a local producer for this worker
  auto producer = output_queue_.CreateProducer();

  try {
    while (true) {
      auto chunk_data = GetNextChunkData();
      if (!chunk_data) continue;
      LoadMetricPauser pauser(context->load_metric_updater);
      producer.Put(std::move(*chunk_data));
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "ShufflingChunkPool output worker stopping, queue closed.";
    // Output queue was closed, stop this worker
  } catch (const std::exception& e) {
    LOG(FATAL) << "Output worker encountered an error: " << e.what();
  }
}

std::optional<std::string> ShufflingChunkPool::GetNextChunkData() {
  absl::MutexLock lock(&chunk_sources_mutex_);
  std::optional<size_t> chunk_index = stream_shuffler_.GetNextItem();

  // If shuffler is exhausted, reset it to current window.
  if (!chunk_index && !chunk_sources_.empty()) {
    size_t total_chunks = chunk_sources_.back().start_chunk_index +
                          chunk_sources_.back().source->GetChunkCount();
    size_t lower_bound = total_chunks > chunk_pool_size_
                             ? total_chunks - chunk_pool_size_
                             : chunk_sources_.front().start_chunk_index;
    stream_shuffler_.Reset(lower_bound, total_chunks);
    chunk_index = stream_shuffler_.GetNextItem();
  }
  // If no chunk index after reset, it means no chunk sources are
  // available.
  assert(chunk_index && "No chunk sources available after initialization");

  // Find which source contains this chunk index using binary search.
  auto it =
      absl::c_lower_bound(chunk_sources_, *chunk_index,
                          [](const auto& source_item, size_t chunk_idx) {
                            return source_item.start_chunk_index +
                                       source_item.source->GetChunkCount() <=
                                   chunk_idx;
                          });

  assert(it != chunk_sources_.end() && *chunk_index >= it->start_chunk_index &&
         "Chunk index should be within available chunk sources");

  size_t local_index = *chunk_index - it->start_chunk_index;
  return it->source->GetChunkData(local_index);
}

void ShufflingChunkPool::AddNewChunkSource(std::unique_ptr<ChunkSource> source)
    ABSL_EXCLUSIVE_LOCKS_REQUIRED(chunk_sources_mutex_) {
  // Add new chunk source to the end of the deque.
  size_t old_upper_bound = 0;
  if (!chunk_sources_.empty()) {
    const auto& last_source = chunk_sources_.back();
    old_upper_bound =
        last_source.start_chunk_index + last_source.source->GetChunkCount();
  }

  chunk_sources_.push_back(
      {.start_chunk_index = old_upper_bound, .source = std::move(source)});

  // Calculate current window bounds.
  size_t new_upper_bound = chunk_sources_.back().start_chunk_index +
                           chunk_sources_.back().source->GetChunkCount();

  // Remove old chunks if window exceeds chunk_pool_size_.
  while (!chunk_sources_.empty() && chunk_sources_.size() > 1) {
    size_t window_start = chunk_sources_.front().start_chunk_index +
                          chunk_sources_.front().source->GetChunkCount();
    size_t window_size = new_upper_bound - window_start;

    if (window_size < chunk_pool_size_) break;

    // Remove the oldest chunk source (front of deque).
    chunk_sources_.pop_front();
  }

  // Update stream shuffler bounds with the sliding window.
  size_t window_start = chunk_sources_.front().start_chunk_index;
  size_t new_lower_bound = new_upper_bound > chunk_pool_size_
                               ? new_upper_bound - chunk_pool_size_
                               : window_start;
  stream_shuffler_.SetUpperBound(new_upper_bound);
  stream_shuffler_.SetLowerBound(new_lower_bound);
}

ShufflingChunkPoolMetricsProto ShufflingChunkPool::FlushMetrics() {
  ShufflingChunkPoolMetricsProto result;

  // Aggregate indexing load metrics from all indexing threads.
  for (const auto& context : indexing_thread_contexts_) {
    UpdateFrom(*result.mutable_indexing_load(),
               context->load_metric_updater.FlushMetrics());
  }

  // Aggregate chunk loading load metrics from all chunk loading threads.
  for (const auto& context : chunk_loading_thread_contexts_) {
    UpdateFrom(*result.mutable_chunk_loading_load(),
               context->load_metric_updater.FlushMetrics());
  }

  // Get queue metrics.
  *result.mutable_queue() = MetricsFromQueue(output_queue_);

  // Get chunk sources statistics and pool state.
  {
    absl::MutexLock lock(&chunk_sources_mutex_);
    AddSample(*result.mutable_chunk_sources_count(),
              static_cast<int64_t>(chunk_sources_.size()));

    // Calculate current chunks and set pool capacity.
    size_t current_chunks = 0;
    if (!chunk_sources_.empty()) {
      current_chunks = chunk_sources_.back().start_chunk_index +
                       chunk_sources_.back().source->GetChunkCount();
    }
    result.set_current_chunks(current_chunks);
    result.set_pool_capacity(chunk_pool_size_);
  }

  // Get anchor-related metrics.
  {
    absl::MutexLock lock(&anchor_mutex_);
    result.set_chunks_since_anchor(chunks_since_anchor_);
    result.set_anchor(anchor_);
  }

  return result;
}

std::string ShufflingChunkPool::ResetAnchor() {
  absl::MutexLock lock(&anchor_mutex_);
  // For ShufflingChunkPool, we'll use the latest chunk source's sort key
  std::string latest_chunk_key;
  {
    absl::MutexLock sources_lock(&chunk_sources_mutex_);
    if (!chunk_sources_.empty()) {
      latest_chunk_key = chunk_sources_.back().source->GetChunkSortKey();
    }
  }
  anchor_ = latest_chunk_key;
  chunks_since_anchor_ = 0;
  return anchor_;
}

int ShufflingChunkPool::ChunksSinceAnchor() { return chunks_since_anchor_; }

std::string ShufflingChunkPool::CurrentAnchor() {
  absl::MutexLock lock(&anchor_mutex_);
  return anchor_;
}

void ShufflingChunkPool::SetAnchor(std::string_view anchor) {
  absl::MutexLock lock(&anchor_mutex_);
  anchor_ = anchor;
}

void ShufflingChunkPool::Close() { output_queue_.Close(); }

}  // namespace training
}  // namespace lczero
