// ABOUTME: Comprehensive unit tests for the ShufflingChunkPool class
// ABOUTME: Tests chunk source management, output workers, and dynamic windowing

#include "loader/stages/shuffling_chunk_pool.h"

#include <absl/cleanup/cleanup.h>
#include <absl/log/log.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace lczero {
namespace training {

// Mock ChunkSource for testing
class MockChunkSource : public ChunkSource {
 public:
  MockChunkSource(const std::string& sort_key, size_t chunk_count,
                  const std::string& chunk_prefix = "chunk")
      : sort_key_(sort_key),
        chunk_count_(chunk_count),
        chunk_prefix_(chunk_prefix) {}

  std::string GetChunkSortKey() const override { return sort_key_; }

  void Index() override { indexed_ = true; }

  size_t GetChunkCount() const override {
    if (!indexed_) {
      throw std::runtime_error("Index() must be called before GetChunkCount()");
    }
    return chunk_count_;
  }

  std::optional<std::string> GetChunkData(size_t index) override {
    if (!indexed_) {
      throw std::runtime_error("Index() must be called before GetChunkData()");
    }
    if (index >= chunk_count_) {
      throw std::out_of_range("Chunk index out of range");
    }
    return chunk_prefix_ + "_" + sort_key_ + "_" + std::to_string(index);
  }

  bool is_indexed() const { return indexed_; }

 private:
  std::string sort_key_;
  size_t chunk_count_;
  std::string chunk_prefix_;
  bool indexed_ = false;
};

class ShufflingChunkPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_queue_ = std::make_unique<Queue<ChunkSourceWithPhase>>(100);
    input_producer_ = std::make_unique<Queue<ChunkSourceWithPhase>::Producer>(
        input_queue_->CreateProducer());
  }

  void TearDown() override {
    // Close the producer to close the queue
    if (input_producer_) input_producer_.reset();
  }

  // Helper to add a mock chunk source to the input queue
  void AddMockChunkSourceToQueue(const std::string& sort_key,
                                 size_t chunk_count,
                                 FilePathProvider::MessageType message_type =
                                     FilePathProvider::MessageType::kFile,
                                 const std::string& chunk_prefix = "data") {
    ChunkSourceWithPhase item;
    item.source =
        std::make_unique<MockChunkSource>(sort_key, chunk_count, chunk_prefix);
    item.message_type = message_type;
    input_producer_->Put(std::move(item));
  }

  void MarkInitialScanComplete() {
    ChunkSourceWithPhase item;
    item.source = nullptr;  // No source for completion marker
    item.message_type = FilePathProvider::MessageType::kInitialScanComplete;
    input_producer_->Put(std::move(item));
  }

  void CloseInputQueue() {
    if (input_producer_) input_producer_.reset();
  }

  std::unique_ptr<Queue<ChunkSourceWithPhase>> input_queue_;
  std::unique_ptr<Queue<ChunkSourceWithPhase>::Producer> input_producer_;
};

TEST_F(ShufflingChunkPoolTest, ConstructorCreatesOutputQueue) {
  // Add some mock chunk sources with enough chunks
  AddMockChunkSourceToQueue("source1", 50);
  AddMockChunkSourceToQueue("source2", 60);
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(100);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);

  auto* output_queue = shuffling_chunk_pool.output();

  // Close input queue to stop input worker from waiting
  CloseInputQueue();

  EXPECT_NE(output_queue, nullptr);
  EXPECT_EQ(output_queue->Capacity(), 100);

  // Drain output queue to prevent workers from blocking
  try {
    while (output_queue->Size() > 0) {
      output_queue->Get();
    }
  } catch (const QueueClosedException&) {
    // Queue closed, that's fine
  }
}

TEST_F(ShufflingChunkPoolTest, HandlesEmptyInputQueue) {
  // Only mark scan complete, no chunk sources
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(100);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  // Constructor should now succeed (initialization is asynchronous)
  ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);

  // The initialization thread should handle the error case
  auto* output_queue = shuffling_chunk_pool.output();

  // Give the initialization thread time to complete and discover the error
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Close input queue to clean up
  CloseInputQueue();

  // Output queue should exist but should be closed due to initialization
  // failure
  EXPECT_NE(output_queue, nullptr);

  // Trying to get from the output queue should throw because it was closed due
  // to init failure
  EXPECT_THROW(output_queue->Get(), QueueClosedException);
}

TEST_F(ShufflingChunkPoolTest, ProcessesInitialScanChunkSources) {
  // Create mock chunk sources with enough chunks
  AddMockChunkSourceToQueue("source1", 30);
  AddMockChunkSourceToQueue("source2", 40);
  AddMockChunkSourceToQueue("source3", 50);
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(100);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  // Test that constructor completes and processes mock chunk sources
  EXPECT_NO_THROW({
    ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);

    // Close input queue to stop input worker from waiting
    CloseInputQueue();

    auto* output_queue = shuffling_chunk_pool.output();
    EXPECT_NE(output_queue, nullptr);
  });
}

TEST_F(ShufflingChunkPoolTest, OutputWorkerProducesChunks) {
  // Create mock chunk sources
  AddMockChunkSourceToQueue("source1", 10, FilePathProvider::MessageType::kFile,
                            "test");
  AddMockChunkSourceToQueue("source2", 15, FilePathProvider::MessageType::kFile,
                            "data");
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(20);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);

  // Close input queue to stop input worker from waiting
  CloseInputQueue();

  auto* output_queue = shuffling_chunk_pool.output();

  // Wait for output workers to produce at least one chunk
  output_queue->WaitForSizeAtLeast(1);

  // Should have some chunks available
  EXPECT_GT(output_queue->Size(), 0);

  // Get a chunk and verify it's from our mock sources
  auto chunk = output_queue->Get();
  EXPECT_FALSE(chunk.empty());
  // Should contain either "test_source1_" or "data_source2_"
  EXPECT_TRUE(chunk.find("source1") != std::string::npos ||
              chunk.find("source2") != std::string::npos);
}

TEST_F(ShufflingChunkPoolTest, NewChunkSourceProcessing) {
  // Start with initial scan and one chunk source - use enough chunks to satisfy
  // window
  AddMockChunkSourceToQueue("initial", 120);  // More chunks than window
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(100);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);

  // Verify chunks are being produced from initial sources
  auto* output_queue = shuffling_chunk_pool.output();
  output_queue->WaitForSizeAtLeast(1);
  EXPECT_NE(output_queue, nullptr);
  EXPECT_GT(output_queue->Size(), 0);

  // Add a new chunk source after initialization
  AddMockChunkSourceToQueue("new_source", 30,
                            FilePathProvider::MessageType::kFile);

  // Close input queue to stop input worker from waiting for more
  CloseInputQueue();

  // The chunk set should still be functional and continue producing chunks
  // from both the initial and new sources
  EXPECT_GT(output_queue->Size(), 0);
}

TEST_F(ShufflingChunkPoolTest, ChunkWindowManagement) {
  // Create more chunks than the window size
  AddMockChunkSourceToQueue("source1", 30);
  AddMockChunkSourceToQueue("source2", 30);
  AddMockChunkSourceToQueue("source3", 30);
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(50);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  // Should only keep sources that fit in the window
  EXPECT_NO_THROW({
    ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);

    // Close input queue to stop input worker from waiting
    CloseInputQueue();

    auto* output_queue = shuffling_chunk_pool.output();
    EXPECT_NE(output_queue, nullptr);
  });
}

// Test the ShufflingChunkPoolConfig structure
TEST_F(ShufflingChunkPoolTest, ShufflingChunkPoolConfigDefaults) {
  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(1000);

  EXPECT_EQ(config.chunk_pool_size(), 1000);
  EXPECT_EQ(config.startup_indexing_threads(), 4);  // Default value
  EXPECT_EQ(config.indexing_threads(), 4);          // Default value
  EXPECT_EQ(config.chunk_loading_threads(), 4);     // Default value
  EXPECT_EQ(config.queue_capacity(), 16);           // Default value
}

TEST_F(ShufflingChunkPoolTest, ShufflingChunkPoolConfigCustomValues) {
  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(500);
  config.set_startup_indexing_threads(2);
  config.set_indexing_threads(3);
  config.set_chunk_loading_threads(4);
  config.set_queue_capacity(25);

  EXPECT_EQ(config.chunk_pool_size(), 500);
  EXPECT_EQ(config.startup_indexing_threads(), 2);
  EXPECT_EQ(config.indexing_threads(), 3);
  EXPECT_EQ(config.chunk_loading_threads(), 4);
  EXPECT_EQ(config.queue_capacity(), 25);
}

TEST_F(ShufflingChunkPoolTest, ChunkSorting) {
  // Add chunk sources in non-sorted order (by sort key)
  AddMockChunkSourceToQueue("source_b", 20);
  AddMockChunkSourceToQueue("source_a", 25);
  AddMockChunkSourceToQueue("source_c", 30);
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(70);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  // ShufflingChunkPool should handle sorting internally (newest first)
  EXPECT_NO_THROW({
    ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);

    // Close input queue to stop input worker from waiting
    CloseInputQueue();

    auto* output_queue = shuffling_chunk_pool.output();
    EXPECT_NE(output_queue, nullptr);
  });
}

TEST_F(ShufflingChunkPoolTest, MultipleInitialIndexingThreads) {
  // Test with multiple indexing threads to ensure no crashes or hangs
  AddMockChunkSourceToQueue("source1", 30);
  AddMockChunkSourceToQueue("source2", 40);
  AddMockChunkSourceToQueue("source3", 50);
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(100);
  config.set_startup_indexing_threads(3);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  // Should work without hanging or crashing
  EXPECT_NO_THROW({
    ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);

    // Close input queue to stop input worker from waiting
    CloseInputQueue();

    auto* output_queue = shuffling_chunk_pool.output();
    EXPECT_NE(output_queue, nullptr);
  });
}

TEST_F(ShufflingChunkPoolTest, StreamShufflerResetWhenExhausted) {
  // Create a small chunk source to quickly exhaust the shuffler
  AddMockChunkSourceToQueue("source1", 3);  // Only 3 chunks for faster testing
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(3);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);  // Large enough to hold all chunks

  ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);

  auto* output_queue = shuffling_chunk_pool.output();

  // Collect chunks continuously and count total chunks received
  std::vector<std::string> all_chunks_received;

  // Wait for and collect chunks to test shuffler reset
  for (size_t i = 0; i < 8; ++i) {
    output_queue->WaitForSizeAtLeast(1);
    auto chunk = output_queue->Get();
    all_chunks_received.push_back(chunk);
  }

  // Debug output
  std::unordered_set<std::string> unique_chunks(all_chunks_received.begin(),
                                                all_chunks_received.end());

  // Close input queue to clean up
  try {
    CloseInputQueue();
  } catch (const QueueClosedException&) {
    // Already closed, that's fine
  }

  // We should see all 3 unique chunks from our source
  EXPECT_EQ(unique_chunks.size(), 3) << "Should see all unique chunks";

  // If reset works properly, we should receive more than 3 total chunks
  // (since chunks will repeat after shuffler reset)
  EXPECT_GT(all_chunks_received.size(), 3)
      << "Should get more than 3 chunks total due to shuffler reset, got "
      << all_chunks_received.size() << " chunks";
}

TEST_F(ShufflingChunkPoolTest, ExplicitClose) {
  // Create chunk sources
  AddMockChunkSourceToQueue("source1", 20);
  AddMockChunkSourceToQueue("source2", 30);
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(40);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);
  auto* output_queue = shuffling_chunk_pool.output();

  // Wait for workers to produce some chunks
  output_queue->WaitForSizeAtLeast(1);

  // Verify output queue is working before close
  EXPECT_GT(output_queue->Size(), 0);

  // Explicitly close the chunk set
  shuffling_chunk_pool.Close();

  // Drain all remaining items from the queue
  while (output_queue->Size() > 0) {
    output_queue->Get();
  }

  // Now the queue should be closed and empty, so Get() should throw
  EXPECT_THROW(output_queue->Get(), QueueClosedException);

  CloseInputQueue();
}

TEST_F(ShufflingChunkPoolTest, CloseStopsOutputWorkers) {
  // Create chunk sources
  AddMockChunkSourceToQueue("source1", 15);
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(15);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(2);
  config.set_queue_capacity(50);

  ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);
  auto* output_queue = shuffling_chunk_pool.output();

  // Wait for workers to produce chunks
  output_queue->WaitForSizeAtLeast(1);
  size_t chunks_before_close = output_queue->Size();

  // Close the chunk set
  shuffling_chunk_pool.Close();

  // Drain any remaining chunks from the queue
  try {
    while (output_queue->Size() > 0) {
      output_queue->Get();
    }
  } catch (const QueueClosedException&) {
    // Expected when queue is empty and closed
  }

  // Should have had chunks before close
  EXPECT_GT(chunks_before_close, 0) << "Should have had chunks before close";

  CloseInputQueue();
}

TEST_F(ShufflingChunkPoolTest, CloseIsIdempotent) {
  // Create chunk sources
  AddMockChunkSourceToQueue("source1", 20);
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(20);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);

  // Close multiple times - should not crash or cause issues
  EXPECT_NO_THROW(shuffling_chunk_pool.Close());
  EXPECT_NO_THROW(shuffling_chunk_pool.Close());
  EXPECT_NO_THROW(shuffling_chunk_pool.Close());

  CloseInputQueue();
}

TEST_F(ShufflingChunkPoolTest, DestructorCallsClose) {
  // Create chunk sources
  AddMockChunkSourceToQueue("source1", 20);
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(20);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  // Test that destructor calls Close() and properly shuts down
  {
    ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);
    auto* output_queue = shuffling_chunk_pool.output();

    // Wait for workers to produce some chunks
    output_queue->WaitForSizeAtLeast(1);
    EXPECT_GT(output_queue->Size(), 0);

    // Close input queue before destructor to allow threads to finish
    CloseInputQueue();

    // ShufflingChunkPool destructor should be called here, which calls Close()
    // and waits for all threads to finish
  }

  // Test passes if destructor completes without hanging
  // (we can't test the queue state after destruction since it's destroyed)
}

TEST_F(ShufflingChunkPoolTest, InputQueueClosureDoesNotCloseOutputQueue) {
  // Create chunk sources
  AddMockChunkSourceToQueue("source1", 30);
  MarkInitialScanComplete();

  ShufflingChunkPoolConfig config;
  config.set_chunk_pool_size(30);
  config.set_startup_indexing_threads(1);
  config.set_indexing_threads(1);
  config.set_chunk_loading_threads(1);
  config.set_queue_capacity(100);

  ShufflingChunkPool shuffling_chunk_pool(input_queue_.get(), config);
  auto* output_queue = shuffling_chunk_pool.output();

  // Wait for workers to produce some chunks
  output_queue->WaitForSizeAtLeast(1);
  EXPECT_GT(output_queue->Size(), 0);

  // Close input queue (simulating end of file discovery)
  CloseInputQueue();

  // Output queue should still be functional - workers should continue
  // producing chunks from existing chunk sources

  // Should still be able to get chunks (queue not closed)
  EXPECT_NO_THROW(output_queue->Get());

  // Explicitly close to clean up
  shuffling_chunk_pool.Close();
}

}  // namespace training
}  // namespace lczero