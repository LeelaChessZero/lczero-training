#include "loader/chunk_feed/chunk_set.h"

#include <filesystem>

#include "absl/synchronization/mutex.h"
#include "loader/chunk_feed/chunk_source.h"
#include "loader/chunk_feed/discovery.h"
#include "loader/chunk_feed/rawfile_chunk_source.h"
#include "loader/chunk_feed/tar_chunk_source.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

std::unique_ptr<ChunkSource> CreateChunkSourceFromFile(
    const std::filesystem::path& filepath) {
  auto extension = filepath.extension();
  if (extension == ".gz") {
    return std::make_unique<RawFileChunkSource>(filepath);
  }
  if (extension == ".tar") {
    return std::make_unique<TarChunkSource>(filepath);
  }
  return nullptr;
}

ChunkSet::ChunkSet(Queue<FileDiscovery::File>* input_queue,
                   const ChunkSetOptions& options)
    : chunks_window_(options.chunks_window),
      thread_pool_(options.num_threads, ThreadPoolOptions{}),
      input_queue_(input_queue),
      output_queue_(options.output_queue_size) {
  InitializeChunkSources();
}

Queue<std::string>* ChunkSet::output() { return &output_queue_; }

void ChunkSet::InitializeChunkSources() {
  ThreadPool file_reader_pool(4);
  std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources;
  absl::Mutex sources_mutex;

  // Read from input queue until kInitialScanComplete
  while (true) {
    auto file = input_queue_->Get();

    if (file.phase == FileDiscovery::Phase::kInitialScanComplete) {
      break;
    }

    if (file.phase == FileDiscovery::Phase::kInitialScan) {
      // Create ChunkSource from file in thread pool
      file_reader_pool.Enqueue(
          [filepath = file.filepath, &uninitialized_sources, &sources_mutex]() {
            auto source = CreateChunkSourceFromFile(filepath);
            if (source) {
              absl::MutexLock lock(&sources_mutex);
              uninitialized_sources.push_back(std::move(source));
            }
          });
    }
  }

  // Wait for all file creation tasks to complete
  file_reader_pool.WaitAll();

  std::sort(uninitialized_sources.begin(), uninitialized_sources.end(),
            [](const auto& a, const auto& b) {
              return a->GetChunkSortKey() < b->GetChunkSortKey();
            });
  std::atomic<size_t> total_chunks = 0;
  size_t source_index = uninitialized_sources.size();

  // TODO If we need different number of threads for indexing, we can just
  // create a separate thread pool for indexing here.
  while (true) {
    file_reader_pool.WaitForAvailableThread();
    if (source_index == 0 || total_chunks >= chunks_window_) break;
    auto& source = uninitialized_sources[--source_index];
    file_reader_pool.Enqueue([source = std::move(source), &total_chunks]() {
      source->Index();
      total_chunks += source->GetChunkCount();
    });
  }
  file_reader_pool.WaitAll();

  if (total_chunks < chunks_window_) {
    throw std::runtime_error("Not enough chunks to feed.");
  }

  size_t start_chunk_index = 0;
  std::for_each(
      uninitialized_sources.begin() + source_index, uninitialized_sources.end(),
      [this, &start_chunk_index](auto& source) {
        chunk_sources_.push_back({.start_chunk_index = start_chunk_index,
                                  .source = std::move(source)});
        start_chunk_index += chunk_sources_.back().source->GetChunkCount();
      });
}

}  // namespace training
}  // namespace lczero
