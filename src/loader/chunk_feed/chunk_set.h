#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "loader/chunk_feed/chunk_source.h"
#include "loader/chunk_feed/discovery.h"
#include "utils/queue.h"
#include "utils/stream_shuffler.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

// Creates a ChunkSource based on file extension. Returns RawFileChunkSource for
// .gz files, TarChunkSource for .tar files, or nullptr for unsupported types.
std::unique_ptr<ChunkSource> CreateChunkSourceFromFile(
    const std::filesystem::path& filepath);

struct ChunkSetOptions {
  size_t chunks_window;      // Number of chunks to keep in memory.
  size_t input_threads = 4;  // Number of threads to use for input processing.
  size_t output_queue_size = 16;  // Size of the output queue.
};

class ChunkSet {
 public:
  ChunkSet(Queue<FileDiscovery::File>* input_queue,
           const ChunkSetOptions& options);

  Queue<std::string>* output();

 private:
  struct ChunkSourceItem {
    size_t start_chunk_index;
    std::unique_ptr<ChunkSource> source;
  };

  std::vector<std::unique_ptr<ChunkSource>> InitializeChunkSources();
  void ProcessInputFiles(
      std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources);
  void InputWorker();
  void AddNewChunkSource(std::unique_ptr<ChunkSource> source)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(chunk_sources_mutex_);

  const size_t chunks_window_;
  ThreadPool input_processing_pool_;
  Queue<FileDiscovery::File>* input_queue_;
  Queue<std::string> output_queue_;

  absl::Mutex chunk_sources_mutex_;
  std::deque<ChunkSourceItem> chunk_sources_
      ABSL_GUARDED_BY(chunk_sources_mutex_);
  StreamShuffler stream_shuffler_ ABSL_GUARDED_BY(chunk_sources_mutex_);
};

}  // namespace training
}  // namespace lczero