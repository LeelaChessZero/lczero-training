#pragma once

#include <filesystem>
#include <memory>

#include "loader/chunk_feed/chunk_source.h"
#include "loader/chunk_feed/discovery.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

// Creates a ChunkSource based on file extension. Returns RawFileChunkSource for
// .gz files, TarChunkSource for .tar files, or nullptr for unsupported types.
std::unique_ptr<ChunkSource> CreateChunkSourceFromFile(
    const std::filesystem::path& filepath);

struct ChunkSourceFeedOptions {
  size_t worker_threads = 1;      // Number of worker threads.
  size_t output_queue_size = 16;  // Size of the output queue.
};

struct ChunkSourceWithPhase {
  std::unique_ptr<ChunkSource> source;
  FileDiscovery::MessageType message_type;
};

// Worker pool that converts FileDiscovery output to ChunkSource objects.
// Takes FileDiscovery::File as input and outputs ChunkSourceWithPhase.
class ChunkSourceFeed {
 public:
  using InputType = FileDiscovery::File;
  using OutputType = ChunkSourceWithPhase;

  ChunkSourceFeed(Queue<InputType>* input_queue,
                  const ChunkSourceFeedOptions& options);

  Queue<OutputType>* output();

 private:
  void Worker();

  Queue<InputType>* input_queue_;
  Queue<OutputType> output_queue_;
  ThreadPool thread_pool_;
};

}  // namespace training
}  // namespace lczero