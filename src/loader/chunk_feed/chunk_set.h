#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "loader/chunk_feed/chunk_source.h"
#include "loader/chunk_feed/discovery.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

std::unique_ptr<ChunkSource> CreateChunkSourceFromFile(
    const std::filesystem::path& filepath);

struct ChunkSetOptions {
  size_t chunks_window;    // Number of chunks to keep in memory.
  size_t num_threads = 4;  // Number of threads to use for feeding chunks.
};

class ChunkSet {
 public:
  ChunkSet(Queue<FileDiscovery::File>* input_queue,
           const ChunkSetOptions& options);

 private:
  struct ChunkSourceItem {
    size_t start_chunk_index;
    std::unique_ptr<ChunkSource> source;
  };

  void InitializeChunkSources();

  size_t chunks_window_;
  ThreadPool thread_pool_;
  Queue<FileDiscovery::File>* input_queue_;
  std::vector<ChunkSourceItem> chunk_sources_;
};

}  // namespace training
}  // namespace lczero