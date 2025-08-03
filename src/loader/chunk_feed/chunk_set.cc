#include "loader/chunk_feed/chunk_set.h"

#include <filesystem>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "loader/chunk_feed/chunk_source.h"
#include "loader/chunk_feed/chunk_source_feed.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

ChunkSet::ChunkSet(Queue<ChunkSourceWithPhase>* input_queue,
                   const ChunkSetOptions& options)
    : chunks_window_(options.chunks_window),
      input_processing_pool_(options.input_threads, ThreadPoolOptions{}),
      input_queue_(input_queue),
      output_queue_(options.output_queue_size) {
  auto uninitialized_sources = InitializeChunkSources();
  ProcessInputFiles(std::move(uninitialized_sources));
}

Queue<std::string>* ChunkSet::output() { return &output_queue_; }

std::vector<std::unique_ptr<ChunkSource>> ChunkSet::InitializeChunkSources() {
  std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources;

  // Read from input queue until kInitialScanComplete.
  while (true) {
    auto chunk_source_with_phase = input_queue_->Get();

    if (chunk_source_with_phase.phase ==
        FileDiscovery::Phase::kInitialScanComplete) {
      break;
    }

    if (chunk_source_with_phase.phase == FileDiscovery::Phase::kInitialScan) {
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

  ThreadPool indexing_pool(4);  // TODO make configurable

  for (auto& source : uninitialized_sources) {
    indexing_pool.WaitForAvailableThread();
    if (total_chunks >= chunks_window_) break;
    indexing_pool.Enqueue([&source, &total_chunks]() {
      source->Index();
      total_chunks += source->GetChunkCount();
    });
    ++sources_to_keep;
  }
  indexing_pool.WaitAll();

  if (total_chunks < chunks_window_) {
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
    std::for_each(
        uninitialized_sources.begin(), uninitialized_sources.end(),
        [this, &start_chunk_index](auto& source) {
          chunk_sources_.push_back({.start_chunk_index = start_chunk_index,
                                    .source = std::move(source)});
          start_chunk_index += chunk_sources_.back().source->GetChunkCount();
        });

    // Initialize stream shuffler with the initial bounds.
    if (!chunk_sources_.empty()) {
      size_t total_chunks = chunk_sources_.back().start_chunk_index +
                            chunk_sources_.back().source->GetChunkCount();
      // Set bounds to provide the last chunks_window_ chunks.
      size_t lower_bound =
          total_chunks > chunks_window_ ? total_chunks - chunks_window_ : 0;
      stream_shuffler_.SetLowerBound(lower_bound);
      stream_shuffler_.SetUpperBound(total_chunks);
    }
  }

  // Start input processing worker that continuously processes new files.
  input_processing_pool_.Enqueue([this]() { InputWorker(); });
}

void ChunkSet::InputWorker() {
  try {
    while (true) {
      auto chunk_source_with_phase = input_queue_->Get();

      if (chunk_source_with_phase.phase == FileDiscovery::Phase::kNewFile) {
        // Index the new chunk source.
        auto source = std::move(chunk_source_with_phase.source);
        source->Index();

        absl::MutexLock lock(&chunk_sources_mutex_);
        AddNewChunkSource(std::move(source));
      }
    }
  } catch (const QueueClosedException&) {
    // Queue is closed, stop processing.
    output_queue_.Close();
  }
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

  // Remove old chunks if window exceeds chunks_window_.
  while (!chunk_sources_.empty() && chunk_sources_.size() > 1) {
    size_t window_start = chunk_sources_.front().start_chunk_index;
    size_t window_size = new_upper_bound - window_start;

    if (window_size <= chunks_window_) break;

    // Remove the oldest chunk source (front of deque).
    chunk_sources_.pop_front();
  }

  // Update stream shuffler bounds with the sliding window.
  size_t window_start = chunk_sources_.front().start_chunk_index;
  size_t new_lower_bound = new_upper_bound > chunks_window_
                               ? new_upper_bound - chunks_window_
                               : window_start;
  stream_shuffler_.SetUpperBound(new_upper_bound);
  stream_shuffler_.SetLowerBound(new_lower_bound);
}

}  // namespace training
}  // namespace lczero
