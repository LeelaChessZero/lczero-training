#include "loader/shuffling_frame_sampler.h"

#include <set>
#include <vector>

#include "gtest/gtest.h"
#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

class ShufflingFrameSamplerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_queue_ = std::make_unique<Queue<V6TrainingData>>(100);
    config_.set_reservoir_size_per_thread(10);  // Small size for testing
    config_.set_queue_capacity(20);
  }

  V6TrainingData CreateTestFrame(uint32_t version) {
    V6TrainingData frame{};
    frame.version = version;
    frame.input_format = 3;
    frame.root_q = 0.5f;
    return frame;
  }

  std::unique_ptr<Queue<V6TrainingData>> input_queue_;
  ShufflingFrameSamplerConfig config_;
};

TEST_F(ShufflingFrameSamplerTest, OutputsNoFramesWithSmallInput) {
  ShufflingFrameSampler sampler(input_queue_.get(), config_);

  // Send 5 frames (less than reservoir size)
  auto producer = input_queue_->CreateProducer();
  std::vector<uint32_t> input_versions = {1, 2, 3, 4, 5};
  for (auto version : input_versions) {
    producer.Put(CreateTestFrame(version));
  }
  producer.Close();

  // Collect all output frames
  std::set<uint32_t> output_versions;
  try {
    while (true) {
      auto frame = sampler.output()->Get();
      output_versions.insert(frame.version);
    }
  } catch (const QueueClosedException&) {
    // Expected when queue is closed
  }

  // With fewer inputs than reservoir size, no frames should be output
  // (they remain in the reservoir)
  EXPECT_EQ(output_versions.size(), 0);
}

TEST_F(ShufflingFrameSamplerTest, OutputsFramesWithLargeInput) {
  ShufflingFrameSampler sampler(input_queue_.get(), config_);

  // Send 20 frames (more than reservoir size of 10)
  auto producer = input_queue_->CreateProducer();
  std::vector<uint32_t> input_versions;
  for (uint32_t i = 1; i <= 20; ++i) {
    input_versions.push_back(i);
    producer.Put(CreateTestFrame(i));
  }
  producer.Close();

  // Collect all output frames
  std::set<uint32_t> output_versions;
  try {
    while (true) {
      auto frame = sampler.output()->Get();
      output_versions.insert(frame.version);
    }
  } catch (const QueueClosedException&) {
    // Expected when queue is closed
  }

  // Should output exactly 11 frames (10 during sampling + 1 final frame before
  // queue closes)
  EXPECT_EQ(output_versions.size(), 11);

  // All output frames should be from the input set
  for (auto version : output_versions) {
    EXPECT_TRUE(std::find(input_versions.begin(), input_versions.end(),
                          version) != input_versions.end());
  }
}

TEST_F(ShufflingFrameSamplerTest, HandlesEmptyInput) {
  ShufflingFrameSampler sampler(input_queue_.get(), config_);

  // Close input queue without sending data
  input_queue_->Close();

  // Should not output any frames
  EXPECT_THROW(sampler.output()->Get(), QueueClosedException);
}

TEST_F(ShufflingFrameSamplerTest, HandlesExactReservoirSize) {
  ShufflingFrameSampler sampler(input_queue_.get(), config_);

  // Send exactly reservoir_size_per_thread frames
  auto producer = input_queue_->CreateProducer();
  std::vector<uint32_t> input_versions;
  for (uint32_t i = 1; i <= config_.reservoir_size_per_thread(); ++i) {
    input_versions.push_back(i);
    producer.Put(CreateTestFrame(i));
  }
  producer.Close();

  // Collect all output frames
  std::set<uint32_t> output_versions;
  try {
    while (true) {
      auto frame = sampler.output()->Get();
      output_versions.insert(frame.version);
    }
  } catch (const QueueClosedException&) {
    // Expected when queue is closed
  }

  // With exactly reservoir size frames, 1 frame should be output
  // (fills reservoir, then queue closes during first sampling attempt)
  EXPECT_EQ(output_versions.size(), 1);
}

TEST_F(ShufflingFrameSamplerTest, PreservesFrameData) {
  config_.set_reservoir_size_per_thread(2);
  ShufflingFrameSampler sampler(input_queue_.get(), config_);

  auto producer = input_queue_->CreateProducer();

  // Create frames with specific data - need more than reservoir size
  V6TrainingData frame1 = CreateTestFrame(100);
  frame1.root_q = 0.1f;
  frame1.input_format = 1;

  V6TrainingData frame2 = CreateTestFrame(200);
  frame2.root_q = 0.2f;
  frame2.input_format = 2;

  V6TrainingData frame3 = CreateTestFrame(300);
  frame3.root_q = 0.3f;
  frame3.input_format = 3;

  producer.Put(frame1);
  producer.Put(frame2);
  producer.Put(frame3);  // This will cause frame1 to be output
  producer.Close();

  // Verify frame data is preserved
  std::vector<V6TrainingData> output_frames;
  try {
    while (true) {
      output_frames.push_back(sampler.output()->Get());
    }
  } catch (const QueueClosedException&) {
    // Expected
  }

  EXPECT_EQ(output_frames.size(), 2);
  // Should be frames that were displaced from the reservoir during sampling
  std::set<uint32_t> output_frame_versions;
  for (const auto& frame : output_frames) {
    output_frame_versions.insert(frame.version);
    // Verify frame data is preserved
    if (frame.version == 100) {
      EXPECT_EQ(frame.root_q, 0.1f);
      EXPECT_EQ(frame.input_format, 1);
    } else if (frame.version == 200) {
      EXPECT_EQ(frame.root_q, 0.2f);
      EXPECT_EQ(frame.input_format, 2);
    }
  }
}

}  // namespace training
}  // namespace lczero