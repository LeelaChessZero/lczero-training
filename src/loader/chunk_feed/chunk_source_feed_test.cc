#include "loader/chunk_feed/chunk_source_feed.h"

#include <gtest/gtest.h>

#include <filesystem>

#include "loader/chunk_feed/discovery.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

TEST(ChunkSourceFeedTest, ProcessesFiles) {
  Queue<FileDiscovery::File> input_queue(10);
  ChunkSourceFeedOptions options{.worker_threads = 1, .output_queue_size = 10};
  ChunkSourceFeed feed(&input_queue, options);

  // Add a file with unsupported extension (should not create ChunkSource)
  input_queue.Put(FileDiscovery::File{
      .filepath = std::filesystem::path("/test.txt"),  // unsupported extension
      .phase = FileDiscovery::Phase::kInitialScan});

  input_queue.Close();

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

TEST(ChunkSourceFeedTest, HandlesPhases) {
  Queue<FileDiscovery::File> input_queue(10);
  ChunkSourceFeedOptions options{.worker_threads = 1, .output_queue_size = 10};
  ChunkSourceFeed feed(&input_queue, options);

  // Test different phases - all should be passed through even if no ChunkSource
  // is created
  input_queue.Put(
      FileDiscovery::File{.filepath = std::filesystem::path("/test1.gz"),
                          .phase = FileDiscovery::Phase::kInitialScan});

  input_queue.Put(
      FileDiscovery::File{.filepath = std::filesystem::path("/test2.gz"),
                          .phase = FileDiscovery::Phase::kNewFile});

  input_queue.Close();

  // Queue should eventually close when input is done
  try {
    while (true) {
      feed.output()->Get();
    }
  } catch (const QueueClosedException&) {
    SUCCEED();
  }
}

}  // namespace training
}  // namespace lczero