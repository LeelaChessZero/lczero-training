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

class TensorGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_queue_ = std::make_unique<Queue<V6TrainingData>>(100);
    config_.set_batch_size(4);
    config_.set_threads(1);
    config_.set_queue_capacity(10);
  }

  V6TrainingData CreateTestFrame() {
    V6TrainingData frame{};
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
    frame.plies_left = 42.5f;

    return frame;
  }

  void VerifyTensorTuple(const TensorTuple& tensors,
                         const std::vector<V6TrainingData>& frames) {
    ASSERT_EQ(tensors.size(), 5);
    const size_t batch_size = frames.size();

    // 1. Verify planes tensor: (batch_size, 112, 8, 8)
    const auto* planes_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[0].get());
    ASSERT_NE(planes_tensor, nullptr);
    EXPECT_EQ(planes_tensor->shape().size(), 4);
    EXPECT_EQ(planes_tensor->shape()[0], batch_size);
    EXPECT_EQ(planes_tensor->shape()[1], 112);
    EXPECT_EQ(planes_tensor->shape()[2], 8);
    EXPECT_EQ(planes_tensor->shape()[3], 8);

    // 2. Verify probabilities tensor: (batch_size, 1858)
    const auto* probs_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[1].get());
    ASSERT_NE(probs_tensor, nullptr);
    EXPECT_EQ(probs_tensor->shape().size(), 2);
    EXPECT_EQ(probs_tensor->shape()[0], batch_size);
    EXPECT_EQ(probs_tensor->shape()[1], 1858);

    // 3. Verify winner tensor: (batch_size, 3)
    const auto* winner_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[2].get());
    ASSERT_NE(winner_tensor, nullptr);
    EXPECT_EQ(winner_tensor->shape().size(), 2);
    EXPECT_EQ(winner_tensor->shape()[0], batch_size);
    EXPECT_EQ(winner_tensor->shape()[1], 3);

    // 4. Verify best_q tensor: (batch_size, 3)
    const auto* best_q_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[3].get());
    ASSERT_NE(best_q_tensor, nullptr);
    EXPECT_EQ(best_q_tensor->shape().size(), 2);
    EXPECT_EQ(best_q_tensor->shape()[0], batch_size);
    EXPECT_EQ(best_q_tensor->shape()[1], 3);

    // 5. Verify plies_left tensor: (batch_size,)
    const auto* plies_left_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[4].get());
    ASSERT_NE(plies_left_tensor, nullptr);
    EXPECT_EQ(plies_left_tensor->shape().size(), 1);
    EXPECT_EQ(plies_left_tensor->shape()[0], batch_size);
  }

  void VerifyTensorData(const TensorTuple& tensors,
                        const std::vector<V6TrainingData>& frames) {
    const size_t batch_size = frames.size();
    const auto* planes_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[0].get());
    const auto* probs_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[1].get());
    const auto* winner_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[2].get());
    const auto* best_q_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[3].get());
    const auto* plies_left_tensor =
        dynamic_cast<const TypedTensor<float>*>(tensors[4].get());

    for (size_t i = 0; i < batch_size; ++i) {
      const auto& frame = frames[i];

      // Verify probabilities data.
      auto probs_slice = probs_tensor->slice({static_cast<ssize_t>(i)});
      for (ssize_t j = 0; j < 1858; ++j) {
        EXPECT_FLOAT_EQ(probs_slice[j], frame.probabilities[j]);
      }

      // Verify winner conversion: result_q=0.5, result_d=0.2
      // win = (1.0 + 0.5 - 0.2) / 2.0 = 0.65
      // draw = 0.2
      // loss = (1.0 - 0.5 - 0.2) / 2.0 = 0.15
      auto winner_slice = winner_tensor->slice({static_cast<ssize_t>(i)});
      EXPECT_FLOAT_EQ(winner_slice[0], 0.65f);  // win
      EXPECT_FLOAT_EQ(winner_slice[1], 0.2f);   // draw
      EXPECT_FLOAT_EQ(winner_slice[2], 0.15f);  // loss

      // Verify best_q conversion: best_q=0.3, best_d=0.1
      // win = (1.0 + 0.3 - 0.1) / 2.0 = 0.6
      // draw = 0.1
      // loss = (1.0 - 0.3 - 0.1) / 2.0 = 0.3
      auto best_q_slice = best_q_tensor->slice({static_cast<ssize_t>(i)});
      EXPECT_FLOAT_EQ(best_q_slice[0], 0.6f);  // win
      EXPECT_FLOAT_EQ(best_q_slice[1], 0.1f);  // draw
      EXPECT_FLOAT_EQ(best_q_slice[2], 0.3f);  // loss

      // Verify plies_left.
      auto plies_left_slice =
          plies_left_tensor->slice({static_cast<ssize_t>(i)});
      EXPECT_FLOAT_EQ(plies_left_slice[0], 42.5f);

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

      // Plane 109: rule50_count = 50, should be 50/100 = 0.5
      for (ssize_t square = 109 * 64; square < 110 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 0.5f);
      }

      // Plane 110: all ones
      for (ssize_t square = 110 * 64; square < 111 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 1.0f);
      }

      // Plane 111: all zeros
      for (ssize_t square = 111 * 64; square < 112 * 64; ++square) {
        EXPECT_FLOAT_EQ(planes_slice[square], 0.0f);
      }
    }
  }

  std::unique_ptr<Queue<V6TrainingData>> input_queue_;
  TensorGeneratorConfig config_;
};

TEST_F(TensorGeneratorTest, GeneratesCorrectTensorShapes) {
  TensorGenerator generator(input_queue_.get(), config_);
  generator.Start();

  auto producer = input_queue_->CreateProducer();
  std::vector<V6TrainingData> frames;
  for (size_t i = 0; i < config_.batch_size(); ++i) {
    frames.push_back(CreateTestFrame());
    producer.Put(frames.back());
  }
  producer.Close();

  auto tensors = generator.output()->Get();
  VerifyTensorTuple(tensors, frames);
}

TEST_F(TensorGeneratorTest, GeneratesCorrectTensorData) {
  TensorGenerator generator(input_queue_.get(), config_);
  generator.Start();

  auto producer = input_queue_->CreateProducer();
  std::vector<V6TrainingData> frames;
  for (size_t i = 0; i < config_.batch_size(); ++i) {
    frames.push_back(CreateTestFrame());
    producer.Put(frames.back());
  }
  producer.Close();

  auto tensors = generator.output()->Get();
  VerifyTensorTuple(tensors, frames);
  VerifyTensorData(tensors, frames);
}

TEST_F(TensorGeneratorTest, HandlesMultipleBatches) {
  TensorGenerator generator(input_queue_.get(), config_);
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  // Send two full batches.
  std::vector<V6TrainingData> all_frames;
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
  auto tensors1 = generator.output()->Get();
  std::vector<V6TrainingData> batch1_frames(
      all_frames.begin(), all_frames.begin() + config_.batch_size());
  VerifyTensorTuple(tensors1, batch1_frames);

  // Get second batch.
  auto tensors2 = generator.output()->Get();
  std::vector<V6TrainingData> batch2_frames(
      all_frames.begin() + config_.batch_size(), all_frames.end());
  VerifyTensorTuple(tensors2, batch2_frames);

  // No more batches should be available.
  EXPECT_THROW(generator.output()->Get(), QueueClosedException);
}

TEST_F(TensorGeneratorTest, HandlesDifferentBatchSizes) {
  config_.set_batch_size(2);
  TensorGenerator generator(input_queue_.get(), config_);
  generator.Start();

  auto producer = input_queue_->CreateProducer();
  std::vector<V6TrainingData> frames;
  for (size_t i = 0; i < config_.batch_size(); ++i) {
    frames.push_back(CreateTestFrame());
    producer.Put(frames.back());
  }
  producer.Close();

  auto tensors = generator.output()->Get();
  VerifyTensorTuple(tensors, frames);
}

TEST_F(TensorGeneratorTest, HandlesEmptyInput) {
  TensorGenerator generator(input_queue_.get(), config_);
  generator.Start();

  // Close input queue without sending data.
  input_queue_->Close();

  // Should not output any tensors.
  EXPECT_THROW(generator.output()->Get(), QueueClosedException);
}

TEST_F(TensorGeneratorTest, VerifiesPlanesConversion) {
  config_.set_batch_size(1);
  TensorGenerator generator(input_queue_.get(), config_);
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  V6TrainingData frame = CreateTestFrame();
  // Set specific bit pattern for plane 0.
  frame.planes[0] = 0xAAAAAAAAAAAAAAAAULL;  // Alternating bits
  // Set specific values for meta planes.
  frame.castling_us_ooo = 1;
  frame.castling_us_oo = 0;
  frame.rule50_count = 75;

  producer.Put(frame);
  producer.Close();

  auto tensors = generator.output()->Get();
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

  // Verify rule50_count conversion: 75/100 = 0.75.
  for (ssize_t square = 109 * 64; square < 110 * 64; ++square) {
    EXPECT_FLOAT_EQ(planes_slice[square], 0.75f);
  }
}

TEST_F(TensorGeneratorTest, VerifiesQDConversion) {
  config_.set_batch_size(1);
  TensorGenerator generator(input_queue_.get(), config_);
  generator.Start();

  auto producer = input_queue_->CreateProducer();

  V6TrainingData frame = CreateTestFrame();
  // Test specific Q/D values.
  frame.result_q = 0.4f;
  frame.result_d = 0.3f;
  frame.best_q = -0.2f;
  frame.best_d = 0.1f;

  producer.Put(frame);
  producer.Close();

  auto tensors = generator.output()->Get();
  const auto* winner_tensor =
      dynamic_cast<const TypedTensor<float>*>(tensors[2].get());
  const auto* best_q_tensor =
      dynamic_cast<const TypedTensor<float>*>(tensors[3].get());

  auto winner_slice = winner_tensor->slice({0});
  auto best_q_slice = best_q_tensor->slice({0});

  // Verify winner: q=0.4, d=0.3
  // win = (1.0 + 0.4 - 0.3) / 2.0 = 0.55
  // draw = 0.3
  // loss = (1.0 - 0.4 - 0.3) / 2.0 = 0.15
  EXPECT_FLOAT_EQ(winner_slice[0], 0.55f);
  EXPECT_FLOAT_EQ(winner_slice[1], 0.3f);
  EXPECT_FLOAT_EQ(winner_slice[2], 0.15f);

  // Verify best_q: q=-0.2, d=0.1
  // win = (1.0 + (-0.2) - 0.1) / 2.0 = 0.35
  // draw = 0.1
  // loss = (1.0 - (-0.2) - 0.1) / 2.0 = 0.55
  EXPECT_FLOAT_EQ(best_q_slice[0], 0.35f);
  EXPECT_FLOAT_EQ(best_q_slice[1], 0.1f);
  EXPECT_FLOAT_EQ(best_q_slice[2], 0.55f);
}

}  // namespace training
}  // namespace lczero