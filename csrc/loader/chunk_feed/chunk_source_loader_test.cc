#include "loader/chunk_feed/chunk_source_loader.h"

#include <gtest/gtest.h>

#include <filesystem>

#include "loader/chunk_feed/file_path_provider.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

TEST(ChunkSourceLoaderTest, ProcessesFiles) {
  Queue<FilePathProvider::File> input_queue(10);
  ChunkSourceLoaderConfig config;
  config.set_worker_threads(1);
  config.set_output_queue_size(10);
  ChunkSourceLoader feed(&input_queue, config);

  {
    auto producer = input_queue.CreateProducer();
    // Add a file with unsupported extension (should not create ChunkSource)
    producer.Put(FilePathProvider::File{
        .filepath =
            std::filesystem::path("/test.txt"),  // unsupported extension
        .message_type = FilePathProvider::MessageType::kFile});
  }  // Producer destroyed here, closing input queue

  // Try to get output - there should be no valid ChunkSources for unsupported
  // files
  try {
    while (true) {
      auto output = feed.output()->Get();
      // If we get output, it means a ChunkSource was created, which shouldn't
      // happen for unsupported files
      FAIL() << "Expected no output for unsupported file extension";
    }
  } catch (const QueueClosedException&) {
    // Expected: queue should be closed when input is done and no output
    // produced
    SUCCEED();
  }
}

TEST(ChunkSourceLoaderTest, HandlesPhases) {
  Queue<FilePathProvider::File> input_queue(10);
  ChunkSourceLoaderConfig config;
  config.set_worker_threads(1);
  config.set_output_queue_size(10);
  ChunkSourceLoader feed(&input_queue, config);

  {
    auto producer = input_queue.CreateProducer();
    // Test different phases - all should be passed through even if no
    // ChunkSource is created
    producer.Put(FilePathProvider::File{
        .filepath = std::filesystem::path("/test1.gz"),
        .message_type = FilePathProvider::MessageType::kFile});

    producer.Put(FilePathProvider::File{
        .filepath = std::filesystem::path("/test2.gz"),
        .message_type = FilePathProvider::MessageType::kFile});
  }  // Producer destroyed here, closing input queue

  // Queue should eventually close when input is done
  try {
    while (true) {
      feed.output()->Get();
    }
  } catch (const QueueClosedException&) {
    SUCCEED();
  }
}

TEST(ChunkSourceLoaderTest, PassesThroughInitialScanComplete) {
  Queue<FilePathProvider::File> input_queue(10);
  ChunkSourceLoaderConfig config;
  config.set_worker_threads(1);
  config.set_output_queue_size(10);
  ChunkSourceLoader feed(&input_queue, config);

  {
    auto producer = input_queue.CreateProducer();
    producer.Put(FilePathProvider::File{
        .filepath = std::filesystem::path(""),
        .message_type = FilePathProvider::MessageType::kInitialScanComplete});
  }  // Producer destroyed here, closing input queue

  // Should get kInitialScanComplete in output with null ChunkSource
  auto output = feed.output()->Get();
  EXPECT_EQ(output.message_type,
            FilePathProvider::MessageType::kInitialScanComplete);
  EXPECT_EQ(output.source, nullptr);

  // Queue should be closed after the single message
  try {
    feed.output()->Get();
    FAIL() << "Expected queue to be closed";
  } catch (const QueueClosedException&) {
    SUCCEED();
  }
}

}  // namespace training
}  // namespace lczero