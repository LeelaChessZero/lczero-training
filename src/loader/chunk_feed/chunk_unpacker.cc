#include "loader/chunk_feed/chunk_unpacker.h"

#include <cstring>

#include "absl/log/log.h"

namespace lczero {
namespace training {

ChunkUnpacker::ChunkUnpacker(Queue<InputType>* input_queue,
                             const ChunkUnpackerOptions& options)
    : input_queue_(input_queue),
      output_queue_(options.output_queue_size),
      thread_pool_(options.worker_threads, ThreadPoolOptions{}) {
  LOG(INFO) << "Starting ChunkUnpacker with " << options.worker_threads
            << " worker threads";
  // Start the worker threads.
  for (size_t i = 0; i < options.worker_threads; ++i) {
    thread_pool_.Enqueue([this]() { Worker(); });
  }
}

Queue<ChunkUnpacker::OutputType>* ChunkUnpacker::output() {
  return &output_queue_;
}

void ChunkUnpacker::Worker() {
  // Create a local producer for this worker thread.
  auto producer = output_queue_.CreateProducer();

  try {
    while (true) {
      auto chunk = input_queue_->Get();

      // Check if chunk size is valid for V6TrainingData frames.
      if (chunk.size() % sizeof(V6TrainingData) != 0) {
        LOG(WARNING) << "Chunk size " << chunk.size()
                     << " is not a multiple of V6TrainingData size "
                     << sizeof(V6TrainingData) << ", skipping chunk";
        continue;
      }

      size_t num_frames = chunk.size() / sizeof(V6TrainingData);
      const char* data = chunk.data();

      // Unpack each frame from the chunk.
      for (size_t i = 0; i < num_frames; ++i) {
        V6TrainingData frame;
        std::memcpy(&frame, data + i * sizeof(V6TrainingData),
                    sizeof(V6TrainingData));
        producer.Put(std::move(frame));
      }
    }
  } catch (const QueueClosedException&) {
    // Input queue is closed, the local producer will be destroyed when this
    // function exits which may close the output queue if this is the last
    // producer.
  }
}

}  // namespace training
}  // namespace lczero