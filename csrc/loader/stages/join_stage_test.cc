#include "loader/stages/join_stage.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "gtest/gtest.h"
#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "proto/data_loader_config.pb.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

class JoinStageTest : public ::testing::Test {
 protected:
  void SetUp() override { config_.mutable_output()->set_queue_capacity(100); }

  FrameType CreateTestFrame(uint32_t version) {
    FrameType frame{};
    frame.version = version;
    frame.input_format = 3;
    frame.root_q = 0.5f;
    return frame;
  }

  JoinPositionsConfig config_;
};

TEST_F(JoinStageTest, JoinsTwoInputs) {
  auto input_queue_1 = std::make_unique<Queue<FrameType>>(10);
  auto input_queue_2 = std::make_unique<Queue<FrameType>>(10);

  JoinPositions join_stage(config_);
  join_stage.SetInputs({input_queue_1.get(), input_queue_2.get()});
  join_stage.Start();

  auto producer_1 = input_queue_1->CreateProducer();
  auto producer_2 = input_queue_2->CreateProducer();

  producer_1.Put(CreateTestFrame(1));
  producer_1.Put(CreateTestFrame(2));
  producer_2.Put(CreateTestFrame(3));
  producer_2.Put(CreateTestFrame(4));

  absl::flat_hash_set<uint32_t> received_versions;
  for (int i = 0; i < 4; ++i) {
    auto frame = join_stage.output_queue()->Get();
    received_versions.insert(frame.version);
  }

  producer_1.Close();
  producer_2.Close();
  join_stage.Stop();

  EXPECT_EQ(received_versions.size(), 4u);
  EXPECT_TRUE(received_versions.contains(1));
  EXPECT_TRUE(received_versions.contains(2));
  EXPECT_TRUE(received_versions.contains(3));
  EXPECT_TRUE(received_versions.contains(4));
}

TEST_F(JoinStageTest, JoinsThreeInputs) {
  auto input_queue_1 = std::make_unique<Queue<FrameType>>(10);
  auto input_queue_2 = std::make_unique<Queue<FrameType>>(10);
  auto input_queue_3 = std::make_unique<Queue<FrameType>>(10);

  JoinPositions join_stage(config_);
  join_stage.SetInputs(
      {input_queue_1.get(), input_queue_2.get(), input_queue_3.get()});
  join_stage.Start();

  auto producer_1 = input_queue_1->CreateProducer();
  auto producer_2 = input_queue_2->CreateProducer();
  auto producer_3 = input_queue_3->CreateProducer();

  producer_1.Put(CreateTestFrame(10));
  producer_2.Put(CreateTestFrame(20));
  producer_3.Put(CreateTestFrame(30));

  absl::flat_hash_set<uint32_t> received_versions;
  for (int i = 0; i < 3; ++i) {
    auto frame = join_stage.output_queue()->Get();
    received_versions.insert(frame.version);
  }

  producer_1.Close();
  producer_2.Close();
  producer_3.Close();
  join_stage.Stop();

  EXPECT_EQ(received_versions.size(), 3u);
  EXPECT_TRUE(received_versions.contains(10));
  EXPECT_TRUE(received_versions.contains(20));
  EXPECT_TRUE(received_versions.contains(30));
}

TEST_F(JoinStageTest, HandlesEmptyInputs) {
  auto input_queue_1 = std::make_unique<Queue<FrameType>>(10);
  auto input_queue_2 = std::make_unique<Queue<FrameType>>(10);

  JoinPositions join_stage(config_);
  join_stage.SetInputs({input_queue_1.get(), input_queue_2.get()});
  join_stage.Start();

  auto producer_1 = input_queue_1->CreateProducer();
  auto producer_2 = input_queue_2->CreateProducer();

  producer_1.Close();
  producer_2.Close();

  auto maybe_frame = join_stage.output_queue()->MaybeGet();
  EXPECT_FALSE(maybe_frame.has_value());

  join_stage.Stop();
}

TEST_F(JoinStageTest, FlushesMetrics) {
  auto input_queue = std::make_unique<Queue<FrameType>>(10);

  JoinPositions join_stage(config_);
  join_stage.SetInputs({input_queue.get()});
  join_stage.Start();

  auto producer = input_queue->CreateProducer();
  producer.Put(CreateTestFrame(1));

  auto frame = join_stage.output_queue()->Get();
  EXPECT_EQ(frame.version, 1u);

  producer.Close();
  join_stage.Stop();

  auto metrics = join_stage.FlushMetrics();
  EXPECT_EQ(metrics.load_metrics_size(), 1);
  EXPECT_EQ(metrics.queue_metrics_size(), 1);
}

}  // namespace training
}  // namespace lczero
