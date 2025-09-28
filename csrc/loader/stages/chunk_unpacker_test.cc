#include "loader/stages/chunk_unpacker.h"

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "loader/stages/training_chunk.h"
#include "proto/data_loader_config.pb.h"
#include "utils/queue.h"

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

 private:
  Queue<T>* queue_;
};

}  // namespace

class ChunkUnpackerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_queue_ = std::make_unique<Queue<TrainingChunk>>(10);
    config_.set_threads(1);
    config_.set_queue_capacity(10);
    config_.set_input("source");
  }

  V6TrainingData CreateTestFrame(uint32_t version) {
    V6TrainingData frame{};
    frame.version = version;
    frame.input_format = 3;
    frame.root_q = 0.5f;
    return frame;
  }

  TrainingChunk MakeChunk(std::vector<V6TrainingData> frames,
                          std::string sort_key = "source", size_t index = 0,
                          uint32_t reshuffle = 0) {
    TrainingChunk chunk;
    chunk.sort_key = std::move(sort_key);
    chunk.index_within_sort_key = index;
    chunk.reshuffle_count = reshuffle;
    chunk.frames = std::move(frames);
    return chunk;
  }

  std::unique_ptr<Queue<TrainingChunk>> input_queue_;
  ChunkUnpackerConfig config_;
};

TEST_F(ChunkUnpackerTest, UnpacksSingleFrame) {
  PassthroughStage<TrainingChunk> source_stage(input_queue_.get());
  Stage::StageList stages{{"source", &source_stage}};
  ChunkUnpacker unpacker(config_, stages);
  unpacker.Start();

  V6TrainingData test_frame = CreateTestFrame(6);
  auto producer = input_queue_->CreateProducer();
  producer.Put(MakeChunk({test_frame}));
  producer.Close();

  auto output_frame = unpacker.output()->Get();
  EXPECT_EQ(output_frame.version, 6);
  EXPECT_EQ(output_frame.input_format, 3);
  EXPECT_EQ(output_frame.root_q, 0.5f);
}

TEST_F(ChunkUnpackerTest, UnpacksMultipleFrames) {
  PassthroughStage<TrainingChunk> source_stage(input_queue_.get());
  Stage::StageList stages{{"source", &source_stage}};
  ChunkUnpacker unpacker(config_, stages);
  unpacker.Start();

  std::vector<V6TrainingData> test_frames = {
      CreateTestFrame(6), CreateTestFrame(7), CreateTestFrame(8)};
  auto producer = input_queue_->CreateProducer();
  producer.Put(MakeChunk(test_frames));
  producer.Close();

  for (size_t i = 0; i < test_frames.size(); ++i) {
    auto output_frame = unpacker.output()->Get();
    EXPECT_EQ(output_frame.version, test_frames[i].version);
    EXPECT_EQ(output_frame.input_format, 3);
    EXPECT_EQ(output_frame.root_q, 0.5f);
  }
}

TEST_F(ChunkUnpackerTest, UnpacksMultipleChunks) {
  PassthroughStage<TrainingChunk> source_stage(input_queue_.get());
  Stage::StageList stages{{"source", &source_stage}};
  ChunkUnpacker unpacker(config_, stages);
  unpacker.Start();

  auto producer = input_queue_->CreateProducer();

  // Send first chunk with 2 frames
  std::vector<V6TrainingData> chunk1_frames = {CreateTestFrame(10),
                                               CreateTestFrame(11)};
  producer.Put(MakeChunk(chunk1_frames, "source", 0));

  // Send second chunk with 1 frame
  std::vector<V6TrainingData> chunk2_frames = {CreateTestFrame(12)};
  producer.Put(MakeChunk(chunk2_frames, "source", 1));

  producer.Close();

  // Verify all frames are output
  std::vector<uint32_t> expected_versions = {10, 11, 12};
  for (auto expected_version : expected_versions) {
    auto output_frame = unpacker.output()->Get();
    EXPECT_EQ(output_frame.version, expected_version);
  }
}

TEST_F(ChunkUnpackerTest, HandlesEmptyChunk) {
  PassthroughStage<TrainingChunk> source_stage(input_queue_.get());
  Stage::StageList stages{{"source", &source_stage}};
  ChunkUnpacker unpacker(config_, stages);
  unpacker.Start();

  auto producer = input_queue_->CreateProducer();
  TrainingChunk empty_chunk;
  empty_chunk.sort_key = "source";
  empty_chunk.index_within_sort_key = 0;
  producer.Put(std::move(empty_chunk));
  producer.Close();

  // Should not produce any output frames, queue should close
  EXPECT_THROW(unpacker.output()->Get(), QueueClosedException);
}

TEST_F(ChunkUnpackerTest, HandlesQueueClosure) {
  PassthroughStage<TrainingChunk> source_stage(input_queue_.get());
  Stage::StageList stages{{"source", &source_stage}};
  ChunkUnpacker unpacker(config_, stages);
  unpacker.Start();

  // Close input queue without sending data
  input_queue_->Close();

  // Output queue should eventually close
  EXPECT_THROW(unpacker.output()->Get(), QueueClosedException);
}

}  // namespace training
}  // namespace lczero
