// ABOUTME: Unit tests for TensorGenerator stage in training pipeline.
// ABOUTME: Tests tensor conversion, batching, and data format correctness.

#include "loader/stages/tensor_generator.h"

#include <cstring>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "utils/queue.h"
#include "utils/tensor.h"

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

class TensorGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_queue_ = std::make_unique<Queue<FrameType>>(100);
    config_.set_batch_size(4);
    config_.set_threads(1);
    config_.mutable_output()->set_queue_capacity(10);
  }

  FrameType CreateTestFrame() {
    FrameType frame{};
    std::memset(&frame, 0, sizeof(frame));

    frame.version = 6;
    frame.input_format = 3;

    // Fill probabilities with test values.
    for (ssize_t i = 0; i < 1858; ++i) {
      frame.probabilities[i] = static_cast<float>(i) / 1858.0f;
    }

    // Fill planes with test pattern.
    for (ssize_t i = 0; i < 104; ++i) {
      frame.planes[i] = 0x0F0F0F0F0F0F0F0FULL + i;  // Test pattern
    }

    // Set castling rights.
    frame.castling_us_ooo = 1;
    frame.castling_us_oo = 0;
    frame.castling_them_ooo = 1;
    frame.castling_them_oo = 1;

    // Set other fields.
    frame.side_to_move_or_enpassant = 1;
    frame.rule50_count = 50;

    // Set Q and D values.
    frame.result_q = 0.5f;
    frame.result_d = 0.2f;
    frame.best_q = 0.3f;
    frame.best_d = 0.1f;
    frame.best_m = 42.5f;
    frame.plies_left = 42.5f;

    return frame;
  }

  void VerifyTensorTuple(const TensorTuple& tensors,
                         const std::vector<FrameType>& frames) {
    const size_t batch_size = frames.size();

    // Verify tuple has 3 elements
    ASSERT_EQ(tensors.size(), 3);

    // Verify input tensor: (batch_size, 112, 8, 8)
    const auto* planes_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[0].get());
    ASSERT_NE(planes_tensor, nullptr);
    EXPECT_EQ(planes_tensor->shape().size(), 4);
    EXPECT_EQ(planes_tensor->shape()[0], batch_size);
    EXPECT_EQ(planes_tensor->shape()[1], 112);
    EXPECT_EQ(planes_tensor->shape()[2], 8);
    EXPECT_EQ(planes_tensor->shape()[3], 8);

    // Verify probabilities tensor: (batch_size, 1858)
    const auto* probs_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[1].get());
    ASSERT_NE(probs_tensor, nullptr);
    EXPECT_EQ(probs_tensor->shape().size(), 2);
    EXPECT_EQ(probs_tensor->shape()[0], batch_size);
    EXPECT_EQ(probs_tensor->shape()[1], 1858);

    // Verify values tensor: (batch_size, 6, 3)
    const auto* values_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[2].get());
    ASSERT_NE(values_tensor, nullptr);
    EXPECT_EQ(values_tensor->shape().size(), 3);
    EXPECT_EQ(values_tensor->shape()[0], batch_size);
    EXPECT_EQ(values_tensor->shape()[1], 6);
    EXPECT_EQ(values_tensor->shape()[2], 3);
  }

  void VerifyTensorData(const TensorTuple& tensors,
                        const std::vector<FrameType>& frames) {
    const size_t batch_size = frames.size();
    const auto* planes_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[0].get());
    const auto* probs_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[1].get());
    const auto* values_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[2].get());

    for (size_t i = 0; i < batch_size; ++i) {
      const auto& frame = frames[i];

      // Verify probabilities data.
      auto probs_slice = probs_tensor->slice({static_cast<ssize_t>(i)});
      for (ssize_t j = 0; j < 1858; ++j) {
        EXPECT_FLOAT_EQ(probs_slice[j], frame.probabilities[j]);
      }

      // Verify values tensor [batch, 6, 3] with raw q/d/m values
      // Index 0: result (q=0.5, d=0.2, m=42.5)
      auto values_slice = values_tensor->slice({static_cast<ssize_t>(i)});
      EXPECT_FLOAT_EQ(values_slice[0 * 3 + 0], 0.5f);   // result_q
      EXPECT_FLOAT_EQ(values_slice[0 * 3 + 1], 0.2f);   // result_d
      EXPECT_FLOAT_EQ(values_slice[0 * 3 + 2], 42.5f);  // result_m

      // Index 1: best (q=0.3, d=0.1, m=42.5)
      EXPECT_FLOAT_EQ(values_slice[1 * 3 + 0], 0.3f);   // best_q
      EXPECT_FLOAT_EQ(values_slice[1 * 3 + 1], 0.1f);   // best_d
      EXPECT_FLOAT_EQ(values_slice[1 * 3 + 2], 42.5f);  // best_m

      // Verify planes data - check first few planes and meta planes.
      auto planes_slice = planes_tensor->slice({static_cast<ssize_t>(i)});

      // Check first plane (plane 0).
      uint64_t expected_plane_0 = 0x0F0F0F0F0F0F0F0FULL;
      for (ssize_t square = 0; square < 64; ++square) {
        float expected =
            static_cast<float>((expected_plane_0 >> (63 - square)) & 1);
        EXPECT_FLOAT_EQ(planes_slice[square], expected);
      }

      // Check meta planes.
      // Plane 104: castling_us_ooo = 1
      for (ssize_t square = 104 * 64; square < 105 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 1.0f);
      }

      // Plane 105: castling_us_oo = 0
      for (ssize_t square = 105 * 64; square < 106 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 0.0f);
      }

      // Plane 109: rule50_count = 50, should be 50/99
      for (ssize_t square = 109 * 64; square < 110 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 50.0f / 99.0f);
      }

      // Plane 110: all zeros
      for (ssize_t square = 110 * 64; square < 111 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 0.0f);
      }

      // Plane 111: all ones
      for (ssize_t square = 111 * 64; square < 112 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 1.0f);
      }
    }
  }

  std::unique_ptr<Queue<FrameType>> input_queue_;
  TensorGeneratorConfig config_;
};

TEST_F(TensorGeneratorTest, GeneratesCorrectTensorShapes) {
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();
  std::vector<FrameType> frames;
  for (size_t i = 0; i < config_.batch_size(); ++i) {
    frames.push_back(CreateTestFrame());
    producer.Put(frames.back());
  }
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  VerifyTensorTuple(tensors, frames);
}

TEST_F(TensorGeneratorTest, GeneratesCorrectTensorData) {
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();
  std::vector<FrameType> frames;
  for (size_t i = 0; i < config_.batch_size(); ++i) {
    frames.push_back(CreateTestFrame());
    producer.Put(frames.back());
  }
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  VerifyTensorTuple(tensors, frames);
  VerifyTensorData(tensors, frames);
}

TEST_F(TensorGeneratorTest, HandlesMultipleBatches) {
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  // Send two full batches.
  std::vector<FrameType> all_frames;
  for (ssize_t batch = 0; batch < 2; ++batch) {
    for (size_t i = 0; i < config_.batch_size(); ++i) {
      auto frame = CreateTestFrame();
      frame.version = batch * 1000 + i;  // Unique version for each frame
      all_frames.push_back(frame);
      producer.Put(frame);
    }
  }
  producer.Close();

  // Get first batch.
  auto tensors1 = generator.output_queue()->Get();
  std::vector<FrameType> batch1_frames(
      all_frames.begin(), all_frames.begin() + config_.batch_size());
  VerifyTensorTuple(tensors1, batch1_frames);

  // Get second batch.
  auto tensors2 = generator.output_queue()->Get();
  std::vector<FrameType> batch2_frames(
      all_frames.begin() + config_.batch_size(), all_frames.end());
  VerifyTensorTuple(tensors2, batch2_frames);

  // No more batches should be available.
  EXPECT_THROW(generator.output_queue()->Get(), QueueClosedException);
}

TEST_F(TensorGeneratorTest, HandlesDifferentBatchSizes) {
  config_.set_batch_size(2);
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();
  std::vector<FrameType> frames;
  for (size_t i = 0; i < config_.batch_size(); ++i) {
    frames.push_back(CreateTestFrame());
    producer.Put(frames.back());
  }
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  VerifyTensorTuple(tensors, frames);
}

TEST_F(TensorGeneratorTest, HandlesEmptyInput) {
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  // Close input queue without sending data.
  input_queue_->Close();

  // Should not output any tensors.
  EXPECT_THROW(generator.output_queue()->Get(), QueueClosedException);
}

TEST_F(TensorGeneratorTest, VerifiesPlanesConversion) {
  config_.set_batch_size(1);
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  FrameType frame = CreateTestFrame();
  // Set specific bit pattern for plane 0.
  frame.planes[0] = 0xAAAAAAAAAAAAAAAAULL;  // Alternating bits
  // Set specific values for meta planes.
  frame.castling_us_ooo = 1;
  frame.castling_us_oo = 0;
  frame.rule50_count = 75;

  producer.Put(frame);
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  const auto* planes_tensor =
      dynamic_cast<const TypedTensor<float>*>(tensors[0].get());

  auto planes_slice = planes_tensor->slice({0});

  // Verify plane 0 bit conversion.
  for (ssize_t square = 0; square < 64; ++square) {
    float expected =
        static_cast<float>((0xAAAAAAAAAAAAAAAAULL >> (63 - square)) & 1);
    EXPECT_FLOAT_EQ(planes_slice[square], expected)
        << "Mismatch at square " << square;
  }

  // Verify rule50_count conversion: 75/99.
  for (ssize_t square = 109 * 64; square < 110 * 64; ++square) {
    EXPECT_FLOAT_EQ(planes_slice[square], 75.0f / 99.0f);
  }
}

TEST_F(TensorGeneratorTest, VerifiesQDConversion) {
  config_.set_batch_size(1);
  TensorGenerator generator(config_);
  generator.SetInputs({input_queue_.get()});
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  FrameType frame = CreateTestFrame();
  // Test specific Q/D values.
  frame.result_q = 0.4f;
  frame.result_d = 0.3f;
  frame.best_q = -0.2f;
  frame.best_d = 0.1f;

  producer.Put(frame);
  producer.Close();

  auto tensors = generator.output_queue()->Get();
  const auto* values_tensor =
      dynamic_cast<const TypedTensor<float>*>(tensors[2].get());

  auto values_slice = values_tensor->slice({0});

  // Verify result values: q=0.4, d=0.3 (raw values, no WDL conversion)
  EXPECT_FLOAT_EQ(values_slice[0 * 3 + 0], 0.4f);  // result_q
  EXPECT_FLOAT_EQ(values_slice[0 * 3 + 1], 0.3f);  // result_d

  // Verify best values: q=-0.2, d=0.1 (raw values, no WDL conversion)
  EXPECT_FLOAT_EQ(values_slice[1 * 3 + 0], -0.2f);  // best_q
  EXPECT_FLOAT_EQ(values_slice[1 * 3 + 1], 0.1f);   // best_d
}

}  // namespace training
}  // namespace lczero
