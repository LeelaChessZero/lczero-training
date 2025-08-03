#include "loader/chunk_feed/chunk_source_feed.h"

#include <filesystem>

#include "loader/chunk_feed/rawfile_chunk_source.h"
#include "loader/chunk_feed/tar_chunk_source.h"

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

ChunkSourceFeed::ChunkSourceFeed(Queue<InputType>* input_queue,
                                 const ChunkSourceFeedOptions& options)
    : input_queue_(input_queue),
      output_queue_(options.output_queue_size),
      thread_pool_(options.worker_threads, ThreadPoolOptions{}) {
  // Start the worker threads.
  for (size_t i = 0; i < options.worker_threads; ++i) {
    thread_pool_.Enqueue([this]() { Worker(); });
  }
}

Queue<ChunkSourceFeed::OutputType>* ChunkSourceFeed::output() {
  return &output_queue_;
}

void ChunkSourceFeed::Worker() {
  try {
    while (true) {
      auto file = input_queue_->Get();

      // Create ChunkSource from the file.
      auto source = CreateChunkSourceFromFile(file.filepath);
      if (source) {
        // Output the ChunkSource with its phase.
        ChunkSourceWithPhase output{.source = std::move(source),
                                    .phase = file.phase};
        output_queue_.Put(std::move(output));
      }
    }
  } catch (const QueueClosedException&) {
    // Input queue is closed, close output queue.
    output_queue_.Close();
  }
}

}  // namespace training
}  // namespace lczero