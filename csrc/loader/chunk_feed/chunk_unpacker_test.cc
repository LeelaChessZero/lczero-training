#include "loader/chunk_feed/chunk_unpacker.h"

#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "proto/data_loader_config.pb.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

class ChunkUnpackerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_queue_ = std::make_unique<Queue<std::string>>(10);
    config_.set_worker_threads(1);
    config_.set_output_queue_size(10);
  }

  V6TrainingData CreateTestFrame(uint32_t version) {
    V6TrainingData frame{};
    frame.version = version;
    frame.input_format = 3;
    frame.root_q = 0.5f;
    return frame;
  }

  std::string PackFrames(const std::vector<V6TrainingData>& frames) {
    std::string chunk;
    chunk.resize(frames.size() * sizeof(V6TrainingData));
    char* data = chunk.data();
    for (size_t i = 0; i < frames.size(); ++i) {
      std::memcpy(data + i * sizeof(V6TrainingData), &frames[i],
                  sizeof(V6TrainingData));
    }
    return chunk;
  }

  std::unique_ptr<Queue<std::string>> input_queue_;
  ChunkUnpackerConfig config_;
};

TEST_F(ChunkUnpackerTest, UnpacksSingleFrame) {
  ChunkUnpacker unpacker(input_queue_.get(), config_);

  V6TrainingData test_frame = CreateTestFrame(6);
  std::string chunk = PackFrames({test_frame});

  auto producer = input_queue_->CreateProducer();
  producer.Put(chunk);
  producer.Close();

  auto output_frame = unpacker.output()->Get();
  EXPECT_EQ(output_frame.version, 6);
  EXPECT_EQ(output_frame.input_format, 3);
  EXPECT_EQ(output_frame.root_q, 0.5f);
}

TEST_F(ChunkUnpackerTest, UnpacksMultipleFrames) {
  ChunkUnpacker unpacker(input_queue_.get(), config_);

  std::vector<V6TrainingData> test_frames = {
      CreateTestFrame(6), CreateTestFrame(7), CreateTestFrame(8)};
  std::string chunk = PackFrames(test_frames);

  auto producer = input_queue_->CreateProducer();
  producer.Put(chunk);
  producer.Close();

  for (size_t i = 0; i < test_frames.size(); ++i) {
    auto output_frame = unpacker.output()->Get();
    EXPECT_EQ(output_frame.version, test_frames[i].version);
    EXPECT_EQ(output_frame.input_format, 3);
    EXPECT_EQ(output_frame.root_q, 0.5f);
  }
}

TEST_F(ChunkUnpackerTest, UnpacksMultipleChunks) {
  ChunkUnpacker unpacker(input_queue_.get(), config_);

  auto producer = input_queue_->CreateProducer();

  // Send first chunk with 2 frames
  std::vector<V6TrainingData> chunk1_frames = {CreateTestFrame(10),
                                               CreateTestFrame(11)};
  producer.Put(PackFrames(chunk1_frames));

  // Send second chunk with 1 frame
  std::vector<V6TrainingData> chunk2_frames = {CreateTestFrame(12)};
  producer.Put(PackFrames(chunk2_frames));

  producer.Close();

  // Verify all frames are output
  std::vector<uint32_t> expected_versions = {10, 11, 12};
  for (auto expected_version : expected_versions) {
    auto output_frame = unpacker.output()->Get();
    EXPECT_EQ(output_frame.version, expected_version);
  }
}

TEST_F(ChunkUnpackerTest, HandlesEmptyChunk) {
  ChunkUnpacker unpacker(input_queue_.get(), config_);

  auto producer = input_queue_->CreateProducer();
  producer.Put(std::string());  // Empty chunk
  producer.Close();

  // Should not produce any output frames, queue should close
  EXPECT_THROW(unpacker.output()->Get(), QueueClosedException);
}

TEST_F(ChunkUnpackerTest, SkipsInvalidSizeChunk) {
  ChunkUnpacker unpacker(input_queue_.get(), config_);

  auto producer = input_queue_->CreateProducer();
  // Create chunk with invalid size (not multiple of sizeof(V6TrainingData))
  std::string invalid_chunk(sizeof(V6TrainingData) + 1, 'x');
  producer.Put(invalid_chunk);
  producer.Close();

  // Should not produce any output frames, queue should close
  EXPECT_THROW(unpacker.output()->Get(), QueueClosedException);
}

TEST_F(ChunkUnpackerTest, HandlesQueueClosure) {
  ChunkUnpacker unpacker(input_queue_.get(), config_);

  // Close input queue without sending data
  input_queue_->Close();

  // Output queue should eventually close
  EXPECT_THROW(unpacker.output()->Get(), QueueClosedException);
}

}  // namespace training
}  // namespace lczero