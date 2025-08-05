#include "loader/chunk_feed/chunk_set.h"

#include <absl/algorithm/container.h>
#include <absl/base/thread_annotations.h>
#include <absl/log/log.h>
#include <absl/synchronization/mutex.h>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <thread>

#include "loader/chunk_feed/chunk_source.h"
#include "loader/chunk_feed/chunk_source_feed.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

ChunkSet::ChunkSet(Queue<ChunkSourceWithPhase>* input_queue,
                   const ChunkSetOptions& options)
    : chunk_pool_size_(options.chunk_pool_size),
      indexing_pool_(options.num_indexing_threads, ThreadPoolOptions{}),
      chunk_loading_pool_(options.num_chunk_loading_threads,
                          ThreadPoolOptions{}),
      input_queue_(input_queue),
      output_queue_(options.output_queue_size) {
  std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources =
      InitializeChunkSources(options.num_startup_indexing_threads);
  ProcessInputFiles(std::move(uninitialized_sources));

  // Start input processing worker that continuously processes new files.
  for (size_t i = 0; i < indexing_pool_.num_threads(); ++i) {
    indexing_pool_.Enqueue([this]() { IndexingWorker(); });
  }

  // Start output workers after everything is fully initialized.
  for (size_t i = 0; i < chunk_loading_pool_.num_threads(); ++i) {
    chunk_loading_pool_.Enqueue([this]() { OutputWorker(); });
  }
}

ChunkSet::~ChunkSet() {
  Close();
  indexing_pool_.WaitAll();
  chunk_loading_pool_.WaitAll();
}

Queue<std::string>* ChunkSet::output() { return &output_queue_; }

std::vector<std::unique_ptr<ChunkSource>> ChunkSet::InitializeChunkSources(
    size_t num_startup_indexing_threads) {
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

  ThreadPool indexing_pool(num_startup_indexing_threads);

  // Index sources ≈sequentially until we have enough chunks. It's fine to
  // overshoot a bit due to multiple threads.
  for (auto& source : uninitialized_sources) {
    indexing_pool.WaitForAvailableThread();
    if (total_chunks >= chunk_pool_size_) break;
    indexing_pool.Enqueue([&source, &total_chunks]() {
      source->Index();
      total_chunks += source->GetChunkCount();
    });
    ++sources_to_keep;
  }
  indexing_pool.WaitAll();

  if (total_chunks < chunk_pool_size_) {
    throw std::runtime_error("Not enough chunks to feed.");
  }

  // Trim the vector to only keep the sources we need.
  uninitialized_sources.resize(sources_to_keep);
  return uninitialized_sources;
}

void ChunkSet::ProcessInputFiles(
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

void ChunkSet::IndexingWorker() {
  try {
    while (true) {
      auto chunk_source_with_phase = input_queue_->Get();

      if (chunk_source_with_phase.message_type ==
          FilePathProvider::MessageType::kFile) {
        // Index the new chunk source.
        auto source = std::move(chunk_source_with_phase.source);
        source->Index();

        absl::MutexLock lock(&chunk_sources_mutex_);
        AddNewChunkSource(std::move(source));
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "Input queue closed, stopping input worker.";
  }
}

void ChunkSet::OutputWorker() {
  // Create a local producer for this worker
  auto producer = output_queue_.CreateProducer();

  try {
    while (true) producer.Put(GetNextChunkData());
  } catch (const QueueClosedException&) {
    // Output queue was closed, stop this worker
  }
}

std::string ChunkSet::GetNextChunkData() {
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

void ChunkSet::AddNewChunkSource(std::unique_ptr<ChunkSource> source)
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

void ChunkSet::Close() { output_queue_.Close(); }

}  // namespace training
}  // namespace lczero
