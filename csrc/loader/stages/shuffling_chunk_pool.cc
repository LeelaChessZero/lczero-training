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

ShufflingChunkPool::ShufflingChunkPool(const ShufflingChunkPoolConfig& config,
                                       const StageList& existing_stages)
    : SingleInputStage<ShufflingChunkPoolConfig, ChunkSourceWithPhase>(
          config, existing_stages),
      chunk_pool_size_(config.chunk_pool_size()),
      config_(config),
      indexing_pool_(config.indexing_threads(), ThreadPoolOptions{}),
      chunk_loading_pool_(config.chunk_loading_threads(), ThreadPoolOptions{}),
      output_queue_(config.queue_capacity()) {
  LOG(INFO) << "Initializing ShufflingChunkPool with pool size "
            << config.chunk_pool_size();
}

ShufflingChunkPool::~ShufflingChunkPool() { Stop(); }

Queue<std::string>* ShufflingChunkPool::output() { return &output_queue_; }

QueueBase* ShufflingChunkPool::GetOutput(std::string_view name) {
  (void)name;
  return &output_queue_;
}

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
    } catch (const QueueClosedException&) {
      LOG(INFO) << "ShufflingChunkPool initialization interrupted, input "
                   "queue closed.";
      output_queue_.Close();
    } catch (const std::exception& e) {
      LOG(ERROR) << "ShufflingChunkPool initialization failed: " << e.what();
      output_queue_.Close();
    }
  });
}

void ShufflingChunkPool::Stop() {
  bool expected = false;
  if (!stop_requested_.compare_exchange_strong(expected, true)) {
    return;
  }

  LOG(INFO) << "Stopping ShufflingChunkPool.";
  input_queue()->Close();
  output_queue_.Close();

  if (initialization_thread_.joinable()) {
    initialization_thread_.join();
  }

  indexing_pool_.WaitAll();
  chunk_loading_pool_.WaitAll();
  indexing_pool_.Shutdown();
  chunk_loading_pool_.Shutdown();
  LOG(INFO) << "ShufflingChunkPool stopped.";
}

std::vector<std::unique_ptr<ChunkSource>>
ShufflingChunkPool::InitializeChunkSources(size_t startup_indexing_threads) {
  std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources;

  // Read from input queue until kInitialScanComplete.
  while (true) {
    auto chunk_source_with_phase = input_queue()->Get();

    if (chunk_source_with_phase.message_type ==
        FilePathProvider::MessageType::kInitialScanComplete) {
      LOG(INFO)
          << "ShufflingChunkPool received initial scan completion marker.";
      break;
    }

    if (chunk_source_with_phase.message_type ==
        FilePathProvider::MessageType::kFile) {
      // Add ChunkSource to uninitialized sources.
      uninitialized_sources.push_back(
          std::move(chunk_source_with_phase.source));
    }
  }

  LOG(INFO) << "ShufflingChunkPool initial directory walk produced "
            << uninitialized_sources.size() << " chunk source candidate(s).";

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

    indexing_pool.Enqueue([&source, &total_chunks, &current_anchor, this]() {
      source->Index();
      const size_t chunk_count = source->GetChunkCount();
      total_chunks += chunk_count;

      // Count chunks since anchor during initial load.
      if (source->GetChunkSortKey() > current_anchor) {
        chunks_since_anchor_ += chunk_count;
      }

      LOG_EVERY_N_SEC(INFO, 4) << "Loaded so far: " << total_chunks.load()
                               << "; new: " << chunks_since_anchor_;
    });
    ++sources_to_keep;
  }
  indexing_pool.WaitAll();

  LOG(INFO) << "ShufflingChunkPool indexed " << total_chunks.load()
            << " chunk(s) across " << sources_to_keep
            << " source(s) during startup.";

  if (total_chunks < chunk_pool_size_ && !output_queue_.IsClosed()) {
    LOG(INFO) << "ShufflingChunkPool startup chunk requirement not met: "
              << total_chunks.load() << " < " << chunk_pool_size_;
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
  size_t initial_window_sources = 0;
  size_t initial_total_chunks = 0;
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
      initial_total_chunks = total_chunks;
    }
    initial_window_sources = chunk_sources_.size();
  }

  LOG(INFO) << "ShufflingChunkPool initial window ready with "
            << initial_window_sources << " source(s) totaling "
            << initial_total_chunks << " chunk(s).";
}

void ShufflingChunkPool::IndexingWorker(IndexingThreadContext* context) {
  try {
    while (true) {
      auto chunk_source_with_phase = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue()->Get();
      }();

      if (chunk_source_with_phase.message_type ==
          FilePathProvider::MessageType::kFile) {
        // Index the new chunk source.
        auto source = std::move(chunk_source_with_phase.source);
        source->Index();
        size_t chunk_count = source->GetChunkCount();
        chunks_since_anchor_ += chunk_count;
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

StageMetricProto ShufflingChunkPool::FlushMetrics() {
  StageMetricProto stage_metric;
  auto* metrics = stage_metric.mutable_shuffling_chunk_pool();

  // Aggregate indexing load metrics from all indexing threads.
  for (const auto& context : indexing_thread_contexts_) {
    UpdateFrom(*metrics->mutable_indexing_load(),
               context->load_metric_updater.FlushMetrics());
  }

  // Aggregate chunk loading load metrics from all chunk loading threads.
  for (const auto& context : chunk_loading_thread_contexts_) {
    UpdateFrom(*metrics->mutable_chunk_loading_load(),
               context->load_metric_updater.FlushMetrics());
  }

  // Get chunk sources statistics and pool state.
  {
    absl::MutexLock lock(&chunk_sources_mutex_);
    AddSample(*metrics->mutable_chunk_sources_count(),
              static_cast<int64_t>(chunk_sources_.size()));

    // Calculate current chunks and set pool capacity.
    size_t current_chunks = 0;
    if (!chunk_sources_.empty()) {
      current_chunks = chunk_sources_.back().start_chunk_index +
                       chunk_sources_.back().source->GetChunkCount();
    }
    metrics->set_current_chunks(current_chunks);
    metrics->set_pool_capacity(chunk_pool_size_);
  }

  // Get anchor-related metrics.
  {
    absl::MutexLock lock(&anchor_mutex_);
    metrics->set_chunks_since_anchor(chunks_since_anchor_);
    metrics->set_anchor(anchor_);
  }

  *stage_metric.add_output_queue_metrics() =
      MetricsFromQueue("output", output_queue_);
  return stage_metric;
}

std::pair<std::string, int> ShufflingChunkPool::ResetAnchor() {
  absl::MutexLock lock(&anchor_mutex_);
  // For ShufflingChunkPool, we'll use the latest chunk source's sort key
  std::string latest_chunk_key;
  {
    absl::MutexLock sources_lock(&chunk_sources_mutex_);
    if (chunk_sources_.empty()) return {"", 0};
    latest_chunk_key = chunk_sources_.back().source->GetChunkSortKey();
  }
  anchor_ = latest_chunk_key;
  int previous_count = chunks_since_anchor_.exchange(0);
  return {anchor_, previous_count};
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

std::optional<StageControlResponse> ShufflingChunkPool::Control(
    const StageControlRequest& request) {
  if (!request.has_chunk_pool_request()) {
    return std::nullopt;
  }

  const auto& chunk_request = request.chunk_pool_request();
  StageControlResponse response;
  auto* chunk_response = response.mutable_chunk_pool_response();

  if (chunk_request.reset_chunk_anchor()) {
    auto [anchor, chunks] = ResetAnchor();
    chunk_response->set_chunk_anchor(anchor);
    chunk_response->set_chunks_since_anchor(chunks);
    return response;
  }

  if (chunk_request.has_set_chunk_anchor()) {
    SetAnchor(chunk_request.set_chunk_anchor());
    chunk_response->set_chunk_anchor(chunk_request.set_chunk_anchor());
    chunk_response->set_chunks_since_anchor(ChunksSinceAnchor());
    return response;
  }

  chunk_response->set_chunk_anchor(CurrentAnchor());
  chunk_response->set_chunks_since_anchor(ChunksSinceAnchor());
  return response;
}

}  // namespace training
}  // namespace lczero
