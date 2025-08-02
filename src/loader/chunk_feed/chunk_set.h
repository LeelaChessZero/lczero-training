#pragma once

#include <string>

#include "loader/chunk_feed/chunk_source.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

struct ChunkSetOptions {
  size_t chunks_window;    // Number of chunks to keep in memory.
  size_t num_threads = 4;  // Number of threads to use for feeding chunks.
};

class ChunkSet {
 public:
  ChunkSet(const ChunkSetOptions& options);

  enum class Phase {
    kInitialization,
    kFeeding,
  };

  void AddChunkSource(std::unique_ptr<ChunkSource> source);
  void StartFeeding();

 private:
  struct ChunkSourceItem {
    size_t start_chunk_index;
    std::unique_ptr<ChunkSource> source;
  };

  size_t chunks_window_;
  ThreadPool thread_pool_;
  std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources_;
  std::vector<ChunkSourceItem> chunk_sources_;

  Phase phase_ = Phase::kInitialization;
};

}  // namespace training
}  // namespace lczero