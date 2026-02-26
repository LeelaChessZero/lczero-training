#include "loader/stages/simple_chunk_extractor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "loader/chunk_source/chunk_source.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/file_path_provider.h"
#include "loader/stages/stage.h"
#include "loader/stages/training_chunk.h"
#include "proto/data_loader_config.pb.h"
#include "utils/queue.h"

namespace lczero {
namespace training {
namespace {

// Mock chunk source for testing.
class MockChunkSource : public ChunkSource {
 public:
  MockChunkSource(std::string sort_key, size_t chunk_count)
      : sort_key_(std::move(sort_key)), chunk_count_(chunk_count) {
    // Pre-generate chunk data.
    for (size_t i = 0; i < chunk_count; ++i) {
      chunks_.emplace_back(10);  // 10 frames per chunk.
    }
  }

  std::string GetChunkSortKey() const override { return sort_key_; }
  size_t GetChunkCount() const override { return chunk_count_; }

  std::optional<std::vector<FrameType>> GetChunkData(size_t index) override {
    if (index >= chunks_.size()) return std::nullopt;
    return chunks_[index];
  }

 private:
  std::string sort_key_;
  size_t chunk_count_;
  std::vector<std::vector<FrameType>> chunks_;
};

class SimpleChunkExtractorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create input queue.
    input_queue_ = std::make_unique<Queue<ChunkSourceWithPhase>>(10);

    // Add a dummy stage to the registry.
    StageConfig dummy_config;
    dummy_config.set_name("dummy_input");
    registry_.AddStage("dummy_input",
                       std::make_unique<DummyStage>(input_queue_.get()));

    // Create the shuffler config.
    SimpleChunkExtractorConfig config;
    config.set_input("dummy_input");
    config.set_queue_capacity(10);

    // Create the shuffler stage.
    shuffler_ = std::make_unique<SimpleChunkExtractor>(config, registry_);
  }

  void TearDown() override {
    if (shuffler_) {
      shuffler_->Stop();
    }
    input_queue_->Close();
  }

  // Helper class to provide a dummy stage for the registry.
  class DummyStage : public Stage {
   public:
    explicit DummyStage(QueueBase* queue) : queue_(queue) {}
    void Start() override {}
    void Stop() override {}
    StageMetricProto FlushMetrics() override { return {}; }
    QueueBase* GetOutput(std::string_view name = "") override {
      (void)name;
      return queue_;
    }

   private:
    QueueBase* queue_;
  };

  StageRegistry registry_;
  std::unique_ptr<Queue<ChunkSourceWithPhase>> input_queue_;
  std::unique_ptr<SimpleChunkExtractor> shuffler_;
};

TEST_F(SimpleChunkExtractorTest, ProcessesSingleSource) {
  shuffler_->Start();

  auto producer = input_queue_->CreateProducer();

  // Send a chunk source with 5 chunks.
  auto source = std::make_unique<MockChunkSource>("source1", 5);
  producer.Put({.source = std::move(source),
                .message_type = FilePathProvider::MessageType::kFile});

  // Close input to signal completion.
  input_queue_->Close();

  // Collect all output chunks.
  auto* output = static_cast<Queue<TrainingChunk>*>(shuffler_->GetOutput());
  std::vector<TrainingChunk> chunks;
  while (true) {
    try {
      chunks.push_back(output->Get());
    } catch (const QueueClosedException&) {
      break;
    }
  }

  // Should receive exactly 5 chunks.
  EXPECT_EQ(chunks.size(), 5);

  // All chunks should have the same sort_key.
  for (const auto& chunk : chunks) {
    EXPECT_EQ(chunk.sort_key, "source1");
    EXPECT_EQ(chunk.frames.size(), 10);  // 10 frames per chunk.
  }

  // Check that all chunk indices are present (though order is shuffled).
  std::vector<size_t> indices;
  for (const auto& chunk : chunks) {
    indices.push_back(chunk.index_within_sort_key);
  }
  std::sort(indices.begin(), indices.end());
  EXPECT_THAT(indices, ::testing::ElementsAre(0, 1, 2, 3, 4));
}

TEST_F(SimpleChunkExtractorTest, ProcessesMultipleSources) {
  shuffler_->Start();

  auto producer = input_queue_->CreateProducer();

  // Send two chunk sources.
  producer.Put({.source = std::make_unique<MockChunkSource>("source1", 3),
                .message_type = FilePathProvider::MessageType::kFile});
  producer.Put({.source = std::make_unique<MockChunkSource>("source2", 2),
                .message_type = FilePathProvider::MessageType::kFile});

  input_queue_->Close();

  // Collect all output chunks.
  auto* output = static_cast<Queue<TrainingChunk>*>(shuffler_->GetOutput());
  std::vector<TrainingChunk> chunks;
  while (true) {
    try {
      chunks.push_back(output->Get());
    } catch (const QueueClosedException&) {
      break;
    }
  }

  // Should receive 3 + 2 = 5 chunks total.
  EXPECT_EQ(chunks.size(), 5);

  // Count chunks per source.
  size_t source1_count = 0;
  size_t source2_count = 0;
  for (const auto& chunk : chunks) {
    if (chunk.sort_key == "source1") {
      ++source1_count;
    } else if (chunk.sort_key == "source2") {
      ++source2_count;
    }
  }

  EXPECT_EQ(source1_count, 3);
  EXPECT_EQ(source2_count, 2);
}

TEST_F(SimpleChunkExtractorTest, SkipsNonFileMessages) {
  shuffler_->Start();

  auto producer = input_queue_->CreateProducer();

  // Send a non-file message.
  producer.Put(
      {.source = nullptr,
       .message_type = FilePathProvider::MessageType::kInitialScanComplete});

  // Send a file message.
  producer.Put({.source = std::make_unique<MockChunkSource>("source1", 2),
                .message_type = FilePathProvider::MessageType::kFile});

  input_queue_->Close();

  // Collect all output chunks.
  auto* output = static_cast<Queue<TrainingChunk>*>(shuffler_->GetOutput());
  std::vector<TrainingChunk> chunks;
  while (true) {
    try {
      chunks.push_back(output->Get());
    } catch (const QueueClosedException&) {
      break;
    }
  }

  // Should only receive 2 chunks from the file message.
  EXPECT_EQ(chunks.size(), 2);
}

TEST_F(SimpleChunkExtractorTest, MetricsAreRecorded) {
  shuffler_->Start();

  auto producer = input_queue_->CreateProducer();
  producer.Put({.source = std::make_unique<MockChunkSource>("source1", 3),
                .message_type = FilePathProvider::MessageType::kFile});

  input_queue_->Close();

  // Wait for processing to complete.
  auto* output = static_cast<Queue<TrainingChunk>*>(shuffler_->GetOutput());
  while (true) {
    try {
      output->Get();
    } catch (const QueueClosedException&) {
      break;
    }
  }

  // Flush metrics.
  auto metrics = shuffler_->FlushMetrics();

  EXPECT_GT(metrics.count_metrics_size(), 0);

  // Check that chunks_processed metric exists.
  bool found_chunks_processed = false;
  for (const auto& metric : metrics.count_metrics()) {
    if (metric.name() == "chunks_processed") {
      EXPECT_EQ(metric.count(), 3);
      found_chunks_processed = true;
    }
  }
  EXPECT_TRUE(found_chunks_processed);
}

}  // namespace
}  // namespace training
}  // namespace lczero
