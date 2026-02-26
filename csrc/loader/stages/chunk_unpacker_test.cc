#include "loader/stages/chunk_unpacker.h"

#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/random.h"
#include "absl/random/seed_sequences.h"
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
  void SetInputs(absl::Span<QueueBase* const> inputs) override {
    if (!inputs.empty()) {
      throw std::runtime_error("PassthroughStage expects no inputs");
    }
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
    config_.mutable_output()->set_queue_capacity(10);
    config_.set_position_sampling_rate(1.0f);
  }

  FrameType CreateTestFrame(uint32_t version) {
    FrameType frame{};
    frame.version = version;
    frame.input_format = 3;
    frame.root_q = 0.5f;
    return frame;
  }

  TrainingChunk MakeChunk(std::vector<FrameType> frames,
                          std::string sort_key = "source", size_t index = 0,
                          uint32_t use = 0) {
    TrainingChunk chunk;
    chunk.sort_key = std::move(sort_key);
    chunk.index_within_sort_key = index;
    chunk.use_count = use;
    chunk.frames = std::move(frames);
    return chunk;
  }

  std::unique_ptr<Queue<TrainingChunk>> input_queue_;
  ChunkUnpackerConfig config_;
};

TEST_F(ChunkUnpackerTest, UnpacksSingleFrame) {
  ChunkUnpacker unpacker(config_);
  unpacker.SetInputs({input_queue_.get()});
  unpacker.Start();

  FrameType test_frame = CreateTestFrame(6);
  auto producer = input_queue_->CreateProducer();
  producer.Put(MakeChunk({test_frame}));
  producer.Close();

  auto output_frame = unpacker.output_queue()->Get();
  EXPECT_EQ(output_frame.version, 6);
  EXPECT_EQ(output_frame.input_format, 3);
  EXPECT_EQ(output_frame.root_q, 0.5f);
}

TEST_F(ChunkUnpackerTest, UnpacksMultipleFrames) {
  ChunkUnpacker unpacker(config_);
  unpacker.SetInputs({input_queue_.get()});
  unpacker.Start();

  std::vector<FrameType> test_frames = {CreateTestFrame(6), CreateTestFrame(7),
                                        CreateTestFrame(8)};
  auto producer = input_queue_->CreateProducer();
  producer.Put(MakeChunk(test_frames));
  producer.Close();

  std::vector<uint32_t> actual_versions;
  actual_versions.reserve(test_frames.size());
  for (size_t i = 0; i < test_frames.size(); ++i) {
    auto output_frame = unpacker.output_queue()->Get();
    actual_versions.push_back(output_frame.version);
    EXPECT_EQ(output_frame.input_format, 3);
    EXPECT_EQ(output_frame.root_q, 0.5f);
  }

  std::vector<uint32_t> expected_versions;
  expected_versions.reserve(test_frames.size());
  for (const auto& frame : test_frames) {
    expected_versions.push_back(frame.version);
  }

  absl::c_sort(actual_versions);
  absl::c_sort(expected_versions);
  EXPECT_EQ(actual_versions, expected_versions);
}

TEST_F(ChunkUnpackerTest, UnpacksMultipleChunks) {
  ChunkUnpacker unpacker(config_);
  unpacker.SetInputs({input_queue_.get()});
  unpacker.Start();

  auto producer = input_queue_->CreateProducer();

  // Send first chunk with 2 frames
  std::vector<FrameType> chunk1_frames = {CreateTestFrame(10),
                                          CreateTestFrame(11)};
  producer.Put(MakeChunk(chunk1_frames, "source", 0));

  // Send second chunk with 1 frame
  std::vector<FrameType> chunk2_frames = {CreateTestFrame(12)};
  producer.Put(MakeChunk(chunk2_frames, "source", 1));

  producer.Close();

  // Verify all frames are output
  std::vector<uint32_t> expected_versions = {10, 11, 12};
  std::vector<uint32_t> actual_versions;
  actual_versions.reserve(expected_versions.size());
  for (size_t i = 0; i < expected_versions.size(); ++i) {
    auto output_frame = unpacker.output_queue()->Get();
    actual_versions.push_back(output_frame.version);
    EXPECT_EQ(output_frame.input_format, 3);
    EXPECT_EQ(output_frame.root_q, 0.5f);
  }

  absl::c_sort(actual_versions);
  absl::c_sort(expected_versions);
  EXPECT_EQ(actual_versions, expected_versions);
}

TEST_F(ChunkUnpackerTest, HandlesEmptyChunk) {
  ChunkUnpacker unpacker(config_);
  unpacker.SetInputs({input_queue_.get()});
  unpacker.Start();

  auto producer = input_queue_->CreateProducer();
  TrainingChunk empty_chunk;
  empty_chunk.sort_key = "source";
  empty_chunk.index_within_sort_key = 0;
  producer.Put(std::move(empty_chunk));
  producer.Close();

  // Should not produce any output frames, queue should close
  EXPECT_THROW(unpacker.output_queue()->Get(), QueueClosedException);
}

TEST_F(ChunkUnpackerTest, HandlesQueueClosure) {
  ChunkUnpacker unpacker(config_);
  unpacker.SetInputs({input_queue_.get()});
  unpacker.Start();

  // Close input queue without sending data
  input_queue_->Close();

  // Output queue should eventually close
  EXPECT_THROW(unpacker.output_queue()->Get(), QueueClosedException);
}

TEST(PickSampledPositionsTest, Deterministic) {
  absl::BitGen gen1(absl::SeedSeq{42});
  std::vector<uint32_t> result1 = PickSampledPositions(1000, 0.1, 5, gen1);

  absl::BitGen gen2(absl::SeedSeq{42});
  std::vector<uint32_t> result2 = PickSampledPositions(1000, 0.1, 5, gen2);

  EXPECT_EQ(result1, result2);
}

TEST(PickSampledPositionsTest, FullBucketFirstRound) {
  absl::BitGen gen(absl::SeedSeq{42});
  const uint32_t n = 10000;
  const double p = 0.1;
  std::vector<uint32_t> result = PickSampledPositions(n, p, 0, gen);
  // Expect size to be around n*p.
  EXPECT_NEAR(result.size(), n * p, n * p * 0.25);
}

TEST(PickSampledPositionsTest, DisjointBuckets) {
  absl::BitGen gen(absl::SeedSeq{42});
  const uint32_t n = 1000;
  const double p = 0.1;

  std::vector<uint32_t> bucket1 = PickSampledPositions(n, p, 0, gen);
  absl::c_sort(bucket1);

  // The generator state is now changed. For the next bucket, we need a fresh
  // one with the same seed to test the logic for a different iteration.
  absl::BitGen gen2(absl::SeedSeq{42});
  std::vector<uint32_t> bucket2 = PickSampledPositions(n, p, 1, gen2);
  absl::c_sort(bucket2);

  std::vector<uint32_t> intersection;
  absl::c_set_intersection(bucket1, bucket2, std::back_inserter(intersection));

  EXPECT_TRUE(intersection.empty());
}

TEST(PickSampledPositionsTest, PartialBucketElementsAreReturned) {
  absl::BitGen gen(absl::SeedSeq{42});
  const uint32_t n = 1000;
  const double p = 0.8;  // remainder 0.2

  // In round 1, elements with toss >= 0.8 are for iteration 1.
  absl::BitGen gen1(absl::SeedSeq{42});
  std::vector<uint32_t> expected_from_round1;
  for (uint32_t i = 0; i < n; ++i) {
    double toss = absl::Uniform<double>(gen1, 0.0, 1.0);
    if (toss >= 0.8) {
      expected_from_round1.push_back(i);
    }
  }
  absl::c_sort(expected_from_round1);

  std::vector<uint32_t> result = PickSampledPositions(n, p, 1, gen);
  absl::c_sort(result);

  // Check if all elements from round 1 are in the final result.
  // This will fail with the current implementation because they are discarded.
  std::vector<uint32_t> intersection;
  absl::c_set_intersection(expected_from_round1, result,
                           std::back_inserter(intersection));

  EXPECT_GT(expected_from_round1.size(), 50);  // High probability for n=1000
  EXPECT_EQ(intersection.size(), expected_from_round1.size());
}

TEST(PickSampledPositionsTest, PartialBucketCompletedSize) {
  absl::BitGen gen(absl::SeedSeq{42});
  const uint32_t n = 10000;
  const double p = 0.8;  // remainder 0.2

  std::vector<uint32_t> result = PickSampledPositions(n, p, 1, gen);

  // Expect size to be around n*p.
  // This will fail due to incorrect probability calculation for completion.
  EXPECT_NEAR(result.size(), n * p, n * p * 0.25);
}

}  // namespace training
}  // namespace lczero
