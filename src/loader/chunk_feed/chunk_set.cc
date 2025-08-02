#include "loader/chunk_feed/chunk_set.h"

#include "loader/chunk_feed/chunk_source.h"

namespace lczero {
namespace training {

ChunkSet::ChunkSet(const ChunkSetOptions& options)
    : chunks_window_(options.chunks_window),
      thread_pool_(options.num_threads, ThreadPoolOptions{}) {}

void ChunkSet::AddChunkSource(std::unique_ptr<ChunkSource> source) {
  if (phase_ == Phase::kInitialization) {
    uninitialized_sources_.push_back(std::move(source));
  } else {
    // NotImplemented();
  }
}

void ChunkSet::StartFeeding() {
  if (phase_ == Phase::kFeeding) return;  // Already in feeding phase.

  std::sort(uninitialized_sources_.begin(), uninitialized_sources_.end(),
            [](const auto& a, const auto& b) {
              return a->GetChunkSortKey() < b->GetChunkSortKey();
            });
  std::atomic<size_t> total_chunks = 0;
  size_t source_index = uninitialized_sources_.size();

  // TODO If we need different number of threads for indexing, we can just
  // create a separate thread pool for indexing here.
  while (true) {
    thread_pool_.WaitForAvailableThread();
    if (source_index == 0 || total_chunks >= chunks_window_) break;
    auto& source = uninitialized_sources_[--source_index];
    thread_pool_.Enqueue([source = std::move(source), &total_chunks]() {
      source->Index();
      total_chunks += source->GetChunkCount();
    });
  }
  thread_pool_.WaitAll();

  if (total_chunks < chunks_window_) {
    throw std::runtime_error("Not enough chunks to feed.");
  }

  size_t start_chunk_index = 0;
  std::for_each(
      uninitialized_sources_.begin() + source_index,
      uninitialized_sources_.end(), [this, &start_chunk_index](auto& source) {
        chunk_sources_.push_back({.start_chunk_index = start_chunk_index,
                                  .source = std::move(source)});
        start_chunk_index += chunk_sources_.back().source->GetChunkCount();
      });

  uninitialized_sources_.clear();
  phase_ = Phase::kFeeding;
}

}  // namespace training
}  // namespace lczero
