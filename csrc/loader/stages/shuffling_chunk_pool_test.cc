// ABOUTME: Comprehensive unit tests for the ShufflingChunkPool class
// ABOUTME: Tests chunk source management, output workers, and dynamic windowing

#include "loader/stages/shuffling_chunk_pool.h"

#include <absl/cleanup/cleanup.h>
#include <absl/log/log.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "loader/stages/training_chunk.h"

namespace lczero {
namespace training {

namespace {

template <typename T>
class PassthroughStage : public Stage {
 public:
  explicit PassthroughStage(Queue<T>* queue) : queue_(queue) {}

  void Start() override {}
  void Stop() override {}
  StageMetricProto FlushMetrics() override { return StageMetricProto(); }
  QueueBase* GetOutput(std::string_view name = "") override {
    (void)name;
    return queue_;
  }
  void SetInputs(absl::Span<QueueBase* const> inputs) override {
    if (!inputs.empty()) {
      throw std::runtime_error("PassthroughStage expects no inputs");
    }
  }

 private:
  Queue<T>* queue_;
};

}  // namespace

// Mock ChunkSource for testing
class MockChunkSource : public ChunkSource {
 public:
  MockChunkSource(const std::string& sort_key, size_t chunk_count)
      : sort_key_(sort_key), chunk_count_(chunk_count) {}

  std::string GetChunkSortKey() const override { return sort_key_; }
  size_t GetChunkCount() const override { return chunk_count_; }

  std::optional<std::vector<FrameType>> GetChunkData(size_t index) override {
    if (index >= chunk_count_) {
      throw std::out_of_range("Chunk index out of range");
    }
    FrameType frame{};
    frame.version = static_cast<uint32_t>(index);
    frame.input_format = 3;
    return std::vector<FrameType>{frame};
  }

 private:
  std::string sort_key_;
  size_t chunk_count_;
};

class InvalidChunkSource : public ChunkSource {
 public:
  explicit InvalidChunkSource(std::string sort_key)
      : sort_key_(std::move(sort_key)) {}

  std::string GetChunkSortKey() const override { return sort_key_; }
  size_t GetChunkCount() const override { return 2; }

  std::optional<std::vector<FrameType>> GetChunkData(size_t index) override {
    if (index >= 2) {
      throw std::out_of_range("Chunk index out of range");
    }
    if (index == 0) {
      return std::nullopt;
    }
    FrameType frame{};
    frame.version = 42;
    return std::vector<FrameType>{frame};
  }

 private:
  std::string sort_key_;
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
                                     FilePathProvider::MessageType::kFile) {
    ChunkSourceWithPhase item;
    item.source = std::make_unique<MockChunkSource>(sort_key, chunk_count);
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

  ShufflingChunkPoolConfig MakeConfig(int chunk_pool_size,
                                      int source_ingestion_threads = 1,
                                      int loading_threads = 1,
                                      int queue_capacity = 100) const {
    ShufflingChunkPoolConfig config;
    config.set_chunk_pool_size(chunk_pool_size);
    config.set_source_ingestion_threads(source_ingestion_threads);
    config.set_chunk_loading_threads(loading_threads);
    config.mutable_output()->set_queue_capacity(queue_capacity);
    return config;
  }

  std::unique_ptr<Queue<ChunkSourceWithPhase>> input_queue_;
  std::unique_ptr<Queue<ChunkSourceWithPhase>::Producer> input_producer_;
};

TEST_F(ShufflingChunkPoolTest, ConstructorCreatesOutputQueue) {
  // Add some mock chunk sources with enough chunks
  AddMockChunkSourceToQueue("source1", 50);
  AddMockChunkSourceToQueue("source2", 60);
  MarkInitialScanComplete();

  auto config = MakeConfig(20);

  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});

  auto* output_queue = shuffling_chunk_pool.output_queue();

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

  auto config = MakeConfig(20);

  // Constructor should now succeed (initialization is asynchronous)
  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});
  shuffling_chunk_pool.Start();

  // The initialization thread should handle the error case
  auto* output_queue = shuffling_chunk_pool.output_queue();

  // Give the initialization thread time to complete and discover the error
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Close input queue to clean up
  CloseInputQueue();

  // Output queue should exist but be closed to signal startup failure when no
  // chunks were found.
  EXPECT_NE(output_queue, nullptr);
  EXPECT_TRUE(output_queue->IsClosed());
  EXPECT_EQ(output_queue->Size(), 0u);
}

TEST_F(ShufflingChunkPoolTest, FlushMetricsHandlesEmptyChunkSources) {
  const int chunk_pool_size = 32;
  auto config = MakeConfig(chunk_pool_size);

  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});

  auto metrics = shuffling_chunk_pool.FlushMetrics();
  bool found_current = false;
  bool found_total = false;
  for (const auto& metric : metrics.gauge_metrics()) {
    if (metric.name() == "chunks_current") {
      found_current = true;
      EXPECT_EQ(metric.value(), 0u);
      EXPECT_EQ(metric.capacity(), static_cast<uint64_t>(chunk_pool_size));
    } else if (metric.name() == "chunks_total") {
      found_total = true;
      EXPECT_EQ(metric.value(), 0u);
    }
  }

  EXPECT_TRUE(found_current)
      << "FlushMetrics should emit chunks_current metric when empty.";
  EXPECT_TRUE(found_total)
      << "FlushMetrics should emit chunks_total metric when empty.";
}

TEST_F(ShufflingChunkPoolTest, FlushMetricsReportsWindowAndTotalCounts) {
  AddMockChunkSourceToQueue("initial", 30);
  MarkInitialScanComplete();

  const int chunk_pool_size = 20;
  ShufflingChunkPool shuffling_chunk_pool(MakeConfig(chunk_pool_size));
  shuffling_chunk_pool.SetInputs({input_queue_.get()});
  shuffling_chunk_pool.Start();

  auto* output_queue = shuffling_chunk_pool.output_queue();
  output_queue->WaitForSizeAtLeast(1);

  uint64_t current_count = 0;
  uint64_t total_count = 0;
  uint64_t current_capacity = 0;
  bool found_metrics = false;
  for (int attempt = 0; attempt < 50 && !found_metrics; ++attempt) {
    auto metrics = shuffling_chunk_pool.FlushMetrics();
    bool has_current = false;
    bool has_total = false;
    for (const auto& metric : metrics.gauge_metrics()) {
      if (metric.name() == "chunks_current") {
        has_current = true;
        current_count = metric.value();
        current_capacity = metric.capacity();
      } else if (metric.name() == "chunks_total") {
        has_total = true;
        total_count = metric.value();
      }
    }
    if (has_current && has_total) {
      found_metrics = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_TRUE(found_metrics)
      << "FlushMetrics should report both chunks_current and chunks_total.";
  EXPECT_EQ(current_count, 30u);
  EXPECT_EQ(current_capacity, static_cast<uint64_t>(chunk_pool_size));
  EXPECT_EQ(total_count, 30u);

  CloseInputQueue();
}

TEST_F(ShufflingChunkPoolTest, ProcessesInitialScanChunkSources) {
  // Create mock chunk sources with enough chunks
  AddMockChunkSourceToQueue("source1", 30);
  AddMockChunkSourceToQueue("source2", 40);
  AddMockChunkSourceToQueue("source3", 50);
  MarkInitialScanComplete();

  auto config = MakeConfig(20);

  // Test that constructor completes and processes mock chunk sources
  EXPECT_NO_THROW({
    ShufflingChunkPool shuffling_chunk_pool(config);

    shuffling_chunk_pool.SetInputs({input_queue_.get()});
    shuffling_chunk_pool.Start();

    // Close input queue to stop input worker from waiting
    CloseInputQueue();

    auto* output_queue = shuffling_chunk_pool.output_queue();
    EXPECT_NE(output_queue, nullptr);
  });
}

TEST_F(ShufflingChunkPoolTest, OutputWorkerProducesChunks) {
  // Create mock chunk sources
  AddMockChunkSourceToQueue("source1", 10,
                            FilePathProvider::MessageType::kFile);
  AddMockChunkSourceToQueue("source2", 15,
                            FilePathProvider::MessageType::kFile);
  MarkInitialScanComplete();

  auto config = MakeConfig(20);

  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});
  shuffling_chunk_pool.Start();

  // Close input queue to stop input worker from waiting
  CloseInputQueue();

  auto* output_queue = shuffling_chunk_pool.output_queue();

  // Wait for output workers to produce at least one chunk
  output_queue->WaitForSizeAtLeast(1);

  // Should have some chunks available
  EXPECT_GT(output_queue->Size(), 0);

  // Get a chunk and verify it's from our mock sources
  auto chunk = output_queue->Get();
  EXPECT_FALSE(chunk.frames.empty());
  EXPECT_TRUE(chunk.sort_key == "source1" || chunk.sort_key == "source2");
  EXPECT_EQ(chunk.frames.size(), 1);
  EXPECT_EQ(chunk.frames.front().version,
            static_cast<uint32_t>(chunk.index_within_sort_key));
  EXPECT_EQ(chunk.use_count, 0u);
}

TEST_F(ShufflingChunkPoolTest, DropsInvalidChunks) {
  ChunkSourceWithPhase invalid_source;
  invalid_source.source =
      std::make_unique<InvalidChunkSource>("invalid_source");
  invalid_source.message_type = FilePathProvider::MessageType::kFile;
  input_producer_->Put(std::move(invalid_source));
  MarkInitialScanComplete();

  auto config = MakeConfig(2, /*source_ingestion_threads=*/1,
                           /*loading_threads=*/1, /*queue_capacity=*/10);

  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});
  shuffling_chunk_pool.Start();

  // Close input queue to stop input worker from waiting
  CloseInputQueue();

  auto* output_queue = shuffling_chunk_pool.output_queue();
  output_queue->WaitForSizeAtLeast(1);
  auto chunk = output_queue->Get();

  EXPECT_EQ(chunk.sort_key, "invalid_source");
  EXPECT_EQ(chunk.index_within_sort_key, 1);
  EXPECT_EQ(chunk.use_count, 0u);
  ASSERT_EQ(chunk.frames.size(), 1);
  EXPECT_EQ(chunk.frames.front().version, 42);

  uint64_t dropped_latest = 0;
  bool found_dropped = false;
  for (int attempt = 0; attempt < 50 && !found_dropped; ++attempt) {
    auto metrics = shuffling_chunk_pool.FlushMetrics();
    for (const auto& metric : metrics.count_metrics()) {
      if (metric.name() == "dropped" && metric.count() > 0) {
        dropped_latest = metric.count();
        found_dropped = true;
        break;
      }
    }
    if (!found_dropped) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  ASSERT_TRUE(found_dropped) << "dropped chunk metrics should be reported";
  EXPECT_GE(dropped_latest, 1u);
}

TEST_F(ShufflingChunkPoolTest, NewChunkSourceProcessing) {
  // Start with initial scan and one chunk source - use enough chunks to satisfy
  // window
  AddMockChunkSourceToQueue("initial", 120);  // More chunks than window
  MarkInitialScanComplete();

  auto config = MakeConfig(20);

  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});
  shuffling_chunk_pool.Start();

  // Verify chunks are being produced from initial sources
  auto* output_queue = shuffling_chunk_pool.output_queue();
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

  auto config = MakeConfig(50);

  // Should only keep sources that fit in the window
  EXPECT_NO_THROW({
    ShufflingChunkPool shuffling_chunk_pool(config);

    shuffling_chunk_pool.SetInputs({input_queue_.get()});
    shuffling_chunk_pool.Start();

    // Close input queue to stop input worker from waiting
    CloseInputQueue();

    auto* output_queue = shuffling_chunk_pool.output_queue();
    EXPECT_NE(output_queue, nullptr);
  });
}

// Test the ShufflingChunkPoolConfig structure
TEST_F(ShufflingChunkPoolTest, ChunkSorting) {
  // Add chunk sources in non-sorted order (by sort key)
  AddMockChunkSourceToQueue("source_b", 20);
  AddMockChunkSourceToQueue("source_a", 25);
  AddMockChunkSourceToQueue("source_c", 30);
  MarkInitialScanComplete();

  auto config = MakeConfig(70);

  // ShufflingChunkPool should handle sorting internally (newest first)
  EXPECT_NO_THROW({
    ShufflingChunkPool shuffling_chunk_pool(config);

    shuffling_chunk_pool.SetInputs({input_queue_.get()});
    shuffling_chunk_pool.Start();

    // Close input queue to stop input worker from waiting
    CloseInputQueue();

    auto* output_queue = shuffling_chunk_pool.output_queue();
    EXPECT_NE(output_queue, nullptr);
  });
}

TEST_F(ShufflingChunkPoolTest, StreamShufflerResetWhenExhausted) {
  // Create a small chunk source to quickly exhaust the shuffler
  AddMockChunkSourceToQueue("source1", 3);  // Only 3 chunks for faster testing
  MarkInitialScanComplete();

  auto config = MakeConfig(3, /*source_ingestion_threads=*/1,
                           /*loading_threads=*/1,
                           /*queue_capacity=*/100);  // Large enough

  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});
  shuffling_chunk_pool.Start();

  auto* output_queue = shuffling_chunk_pool.output_queue();

  // Collect chunks continuously and count total chunks received
  struct ChunkRecord {
    std::string sort_key;
    size_t index;
    uint32_t use_count;
  };
  std::vector<ChunkRecord> all_chunks_received;

  // Wait for and collect chunks to test shuffler reset
  for (size_t i = 0; i < 8; ++i) {
    output_queue->WaitForSizeAtLeast(1);
    auto chunk = output_queue->Get();
    all_chunks_received.push_back(
        {chunk.sort_key, chunk.index_within_sort_key, chunk.use_count});
  }

  std::set<std::pair<std::string, size_t>> unique_chunks;
  bool seen_reuse = false;
  for (const auto& record : all_chunks_received) {
    unique_chunks.emplace(record.sort_key, record.index);
    if (record.use_count > 0) {
      seen_reuse = true;
    }
  }

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
  EXPECT_TRUE(seen_reuse)
      << "Expect at least one chunk to report a reuse count";
}

TEST_F(ShufflingChunkPoolTest, HanseMetrics_NoRejection_CacheAndReshuffles) {
  // Single chunk so we will continually reuse the same chunk.
  AddMockChunkSourceToQueue("source1", 1);
  MarkInitialScanComplete();

  auto config = MakeConfig(1, /*source_ingestion_threads=*/1,
                           /*loading_threads=*/1, /*queue_capacity=*/100);
  // Enable Hanse sampling with p == 1 to avoid rejections.
  config.set_hanse_sampling_threshold(1);

  ShufflingChunkPool pool(config);

  pool.SetInputs({input_queue_.get()});
  pool.Start();

  auto* output_queue = pool.output_queue();
  // Wait for multiple outputs to exercise cache hits and reshuffles.
  output_queue->WaitForSizeAtLeast(3);
  // Drain a few items.
  for (int i = 0; i < 3; ++i) {
    auto chunk = output_queue->Get();
    EXPECT_EQ(chunk.frames.size(), 1u);
  }

  // Close input to avoid lingering.
  CloseInputQueue();

  // Flush metrics and validate Hanse counters and reshuffles.
  auto metrics = pool.FlushMetrics();
  uint64_t cache_hits = 0, cache_misses = 0, rejected = 0, reshuffles = 0;
  for (const auto& m : metrics.count_metrics()) {
    if (m.name() == "hanse_cache_hits") cache_hits = m.count();
    if (m.name() == "hanse_cache_misses") cache_misses = m.count();
    if (m.name() == "hanse_rejected") rejected = m.count();
    if (m.name() == "reshuffles") reshuffles = m.count();
  }

  // First access computes and caches num_records => 1 miss, then hits.
  EXPECT_EQ(cache_misses, 1u);
  EXPECT_GE(cache_hits, 1u);
  // With threshold=1 and one frame, p = 1 => no rejections.
  EXPECT_EQ(rejected, 0u);
  // Single chunk repeatedly consumed forces reshuffles.
  EXPECT_GT(reshuffles, 0u);
}

TEST_F(ShufflingChunkPoolTest, ExplicitClose) {
  // Create chunk sources
  AddMockChunkSourceToQueue("source1", 20);
  AddMockChunkSourceToQueue("source2", 30);
  MarkInitialScanComplete();

  auto config = MakeConfig(40);

  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});
  shuffling_chunk_pool.Start();
  auto* output_queue = shuffling_chunk_pool.output_queue();

  // Wait for workers to produce some chunks
  output_queue->WaitForSizeAtLeast(1);

  // Verify output queue is working before close
  EXPECT_GT(output_queue->Size(), 0);

  // Explicitly stop the chunk set
  shuffling_chunk_pool.Stop();

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

  auto config = MakeConfig(15, /*source_ingestion_threads=*/1,
                           /*loading_threads=*/2, /*queue_capacity=*/50);

  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});
  shuffling_chunk_pool.Start();
  auto* output_queue = shuffling_chunk_pool.output_queue();

  // Wait for workers to produce chunks
  output_queue->WaitForSizeAtLeast(1);
  size_t chunks_before_close = output_queue->Size();

  // Stop the chunk set
  shuffling_chunk_pool.Stop();

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

  auto config = MakeConfig(20);

  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});
  shuffling_chunk_pool.Start();

  // Stop multiple times - should not crash or cause issues
  EXPECT_NO_THROW(shuffling_chunk_pool.Stop());
  EXPECT_NO_THROW(shuffling_chunk_pool.Stop());
  EXPECT_NO_THROW(shuffling_chunk_pool.Stop());

  CloseInputQueue();
}

TEST_F(ShufflingChunkPoolTest, DestructorCallsClose) {
  // Create chunk sources
  AddMockChunkSourceToQueue("source1", 20);
  MarkInitialScanComplete();

  auto config = MakeConfig(20);

  // Test that destructor calls Close() and properly shuts down
  {
    ShufflingChunkPool shuffling_chunk_pool(config);

    shuffling_chunk_pool.SetInputs({input_queue_.get()});
    shuffling_chunk_pool.Start();
    auto* output_queue = shuffling_chunk_pool.output_queue();

    // Wait for workers to produce some chunks
    output_queue->WaitForSizeAtLeast(1);
    EXPECT_GT(output_queue->Size(), 0);

    // Close input queue before destructor to allow threads to finish
    CloseInputQueue();

    // ShufflingChunkPool destructor should be called here, which calls Stop()
    // and waits for all threads to finish
  }

  // Test passes if destructor completes without hanging
  // (we can't test the queue state after destruction since it's destroyed)
}

TEST_F(ShufflingChunkPoolTest, InputQueueClosureDoesNotCloseOutputQueue) {
  // Create chunk sources
  AddMockChunkSourceToQueue("source1", 30);
  MarkInitialScanComplete();

  auto config = MakeConfig(30);

  ShufflingChunkPool shuffling_chunk_pool(config);

  shuffling_chunk_pool.SetInputs({input_queue_.get()});
  shuffling_chunk_pool.Start();
  auto* output_queue = shuffling_chunk_pool.output_queue();

  // Wait for workers to produce some chunks
  output_queue->WaitForSizeAtLeast(1);
  EXPECT_GT(output_queue->Size(), 0);

  // Close input queue (simulating end of file discovery)
  CloseInputQueue();

  // Output queue should still be functional - workers should continue
  // producing chunks from existing chunk sources

  // Should still be able to get chunks (queue not closed)
  EXPECT_NO_THROW(output_queue->Get());

  // Explicitly stop to clean up
  shuffling_chunk_pool.Stop();
}

TEST_F(ShufflingChunkPoolTest, BasicAnchorFunctionality) {
  AddMockChunkSourceToQueue("source1", 20);
  MarkInitialScanComplete();

  auto config = MakeConfig(20);

  ShufflingChunkPool pool(config);

  pool.SetInputs({input_queue_.get()});
  pool.Start();

  // Test initial state
  EXPECT_EQ(pool.ChunksSinceAnchor(), 0);
  EXPECT_EQ(pool.CurrentAnchor(), "");

  // Test SetAnchor and CurrentAnchor
  pool.SetAnchor("test_anchor_key");
  EXPECT_EQ(pool.CurrentAnchor(), "test_anchor_key");
  EXPECT_EQ(pool.ChunksSinceAnchor(), 0);  // Should still be 0

  // Test setting different anchor
  pool.SetAnchor("another_key");
  EXPECT_EQ(pool.CurrentAnchor(), "another_key");

  CloseInputQueue();
}

TEST_F(ShufflingChunkPoolTest, ResetAnchor) {
  AddMockChunkSourceToQueue("source1", 20);
  MarkInitialScanComplete();

  auto config = MakeConfig(20);

  ShufflingChunkPool pool(config);

  pool.SetInputs({input_queue_.get()});
  pool.Start();

  // Wait for initialization to complete
  pool.output_queue()->WaitForSizeAtLeast(1);

  // Now test ResetAnchor
  auto [anchor, count_before] = pool.ResetAnchor();
  EXPECT_FALSE(anchor.empty());  // Should have the chunk key
  EXPECT_EQ(pool.CurrentAnchor(), anchor);
  EXPECT_EQ(pool.ChunksSinceAnchor(), 0);  // Should be reset to 0

  CloseInputQueue();
}

TEST_F(ShufflingChunkPoolTest, AnchorCounterIncrement) {
  // Don't mark initial scan complete yet - we'll add sources one by one

  auto config = MakeConfig(20);

  // Start with some initial sources and complete scan
  AddMockChunkSourceToQueue("source1", 20);
  MarkInitialScanComplete();

  ShufflingChunkPool pool(config);

  pool.SetInputs({input_queue_.get()});
  pool.Start();

  // Set anchor to a key that won't match our new sources
  pool.SetAnchor("non_matching_key");

  // Wait for initial load to complete
  pool.output_queue()->WaitForSizeAtLeast(1);

  // Now add new sources (these should increment the counter)
  // Note: We can't add more sources after initial scan complete in the current
  // setup So we'll test the counter after the initial load

  int final_count = pool.ChunksSinceAnchor();

  // Counter should have incremented during initial load since anchor doesn't
  // match
  EXPECT_GT(final_count, 0);
  EXPECT_EQ(pool.CurrentAnchor(), "non_matching_key");  // Anchor unchanged

  CloseInputQueue();
}

TEST_F(ShufflingChunkPoolTest, AnchorCounterResetDuringInitialLoad) {
  // Test the special case where anchor is encountered during initial backward
  // processing
  AddMockChunkSourceToQueue("source_c", 10);  // newest
  AddMockChunkSourceToQueue("source_b", 15);  // middle
  AddMockChunkSourceToQueue("source_a", 20);  // oldest

  auto config = MakeConfig(45);

  ShufflingChunkPool pool(config);

  pool.SetInputs({input_queue_.get()});
  pool.Start();

  // Set anchor to middle source before marking scan complete
  pool.SetAnchor("source_b");

  // Mark scan complete to trigger initial processing
  MarkInitialScanComplete();

  // Wait for initial load to complete
  pool.output_queue()->WaitForSizeAtLeast(1);

  int final_count = pool.ChunksSinceAnchor();

  // Should only count chunks from source_c (10 chunks) since it is newer than
  // the anchor.
  EXPECT_EQ(final_count, 10);
  EXPECT_EQ(pool.CurrentAnchor(), "source_b");

  CloseInputQueue();
}

}  // namespace training
}  // namespace lczero
