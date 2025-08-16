#pragma once

#include <filesystem>
#include <memory>

#include "loader/chunk_feed/chunk_source.h"
#include "loader/chunk_feed/file_path_provider.h"
#include "proto/training_config.pb.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

// Creates a ChunkSource based on file extension. Returns RawFileChunkSource for
// .gz files, TarChunkSource for .tar files, or nullptr for unsupported types.
std::unique_ptr<ChunkSource> CreateChunkSourceFromFile(
    const std::filesystem::path& filepath);

struct ChunkSourceWithPhase {
  std::unique_ptr<ChunkSource> source;
  FilePathProvider::MessageType message_type;
};

// Worker pool that converts FilePathProvider output to ChunkSource objects.
// Takes FilePathProvider::File as input and outputs ChunkSourceWithPhase.
class ChunkSourceLoader {
 public:
  using InputType = FilePathProvider::File;
  using OutputType = ChunkSourceWithPhase;

  ChunkSourceLoader(Queue<InputType>* input_queue,
                    const ChunkSourceLoaderConfig& config);

  Queue<OutputType>* output();

 private:
  void Worker();

  Queue<InputType>* input_queue_;
  Queue<OutputType> output_queue_;
  ThreadPool thread_pool_;
};

}  // namespace training
}  // namespace lczero