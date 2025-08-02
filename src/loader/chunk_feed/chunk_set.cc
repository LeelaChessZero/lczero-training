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
      input_processing_pool_(options.input_threads, ThreadPoolOptions{}),
      input_queue_(input_queue),
      output_queue_(options.output_queue_size) {
  auto uninitialized_sources = InitializeChunkSources();
  ProcessInputFiles(std::move(uninitialized_sources));
}

Queue<std::string>* ChunkSet::output() { return &output_queue_; }

std::vector<std::unique_ptr<ChunkSource>> ChunkSet::InitializeChunkSources() {
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

  // Sort in descending order (newest first)
  std::sort(uninitialized_sources.begin(), uninitialized_sources.end(),
            [](const auto& a, const auto& b) {
              return a->GetChunkSortKey() > b->GetChunkSortKey();
            });
  std::atomic<size_t> total_chunks = 0;
  size_t sources_to_keep = 0;

  // TODO If we need different number of threads for indexing, we can just
  // create a separate thread pool for indexing here.
  for (auto& source : uninitialized_sources) {
    file_reader_pool.WaitForAvailableThread();
    if (total_chunks >= chunks_window_) break;
    file_reader_pool.Enqueue([&source, &total_chunks]() {
      source->Index();
      total_chunks += source->GetChunkCount();
    });
    ++sources_to_keep;
  }
  file_reader_pool.WaitAll();

  if (total_chunks < chunks_window_) {
    throw std::runtime_error("Not enough chunks to feed.");
  }

  // Trim the vector to only keep the sources we need
  uninitialized_sources.resize(sources_to_keep);
  return uninitialized_sources;
}

void ChunkSet::ProcessInputFiles(
    std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources) {
  // Initialize chunk sources from the initial scan
  size_t start_chunk_index = 0;
  std::for_each(
      uninitialized_sources.begin(), uninitialized_sources.end(),
      [this, &start_chunk_index](auto& source) {
        chunk_sources_.push_back({.start_chunk_index = start_chunk_index,
                                  .source = std::move(source)});
        start_chunk_index += chunk_sources_.back().source->GetChunkCount();
      });

  // Start input processing worker that continuously processes new files
  input_processing_pool_.Enqueue([this]() { InputWorker(); });
}

void ChunkSet::InputWorker() {
  try {
    while (true) {
      auto file = input_queue_->Get();

      if (file.phase == FileDiscovery::Phase::kNewFile) {
        // Create and index new chunk source
        auto source = CreateChunkSourceFromFile(file.filepath);
        if (source) {
          source->Index();
          // TODO: Add to chunk_sources_ with proper synchronization
          // TODO: Update stream_shuffler_ bounds
          // TODO: Remove old chunks if exceeding chunks_window_
        }
      }
    }
  } catch (const QueueClosedException&) {
    // Queue is closed, stop processing
    output_queue_.Close();
  }
}

}  // namespace training
}  // namespace lczero
