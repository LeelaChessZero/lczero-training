#include "loader/stages/chunk_source_splitter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "gtest/gtest.h"
#include "loader/chunk_source/chunk_source.h"
#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/stage.h"
#include "proto/data_loader_config.pb.h"
#include "utils/queue.h"

namespace lczero {
namespace training {
namespace {

// Simple fixed-count chunk source for testing.
class FixedCountChunkSource : public ChunkSource {
 public:
  FixedCountChunkSource(std::string sort_key, size_t count)
      : key_(std::move(sort_key)), count_(count) {}

 private:
  std::string GetChunkSortKey() const override { return key_; }
  size_t GetChunkCount() const override { return count_; }
  std::optional<std::vector<FrameType>> GetChunkData(size_t) override {
    return std::vector<FrameType>{FrameType{}};
  }

  std::string key_;
  size_t count_;
};

template <typename T>
class PassthroughStage : public Stage {
 public:
  explicit PassthroughStage(Queue<T>* queue) : queue_(queue) {}

  void Start() override {}
  void Stop() override {}
  StageMetricProto FlushMetrics() override { return StageMetricProto(); }
  QueueBase* GetOutput(std::string_view) override { return queue_; }
  void SetInputs(absl::Span<QueueBase* const> inputs) override {
    if (!inputs.empty()) {
      throw std::runtime_error("PassthroughStage expects no inputs");
    }
  }

 private:
  Queue<T>* queue_;
};

}  // namespace

TEST(ChunkSourceSplitterTest, SplitsByHashAndWeight) {
  // Upstream queue.
  auto input_queue = std::make_unique<Queue<ChunkSourceWithPhase>>(8);

  // Configure splitter with two outputs A:1, B:2.
  ChunkSourceSplitterConfig cfg;
  auto* outA = cfg.add_output();
  outA->set_name("A");
  outA->set_queue_capacity(8);
  cfg.add_weight(1);
  auto* outB = cfg.add_output();
  outB->set_name("B");
  outB->set_queue_capacity(8);
  cfg.add_weight(2);

  ChunkSourceSplitter splitter(cfg);
  splitter.SetInputs({input_queue.get()});

  splitter.Start();

  // Send a source with known key and count.
  const std::string key = "skey";
  const size_t count = 100;
  ChunkSourceWithPhase item;
  item.source = std::make_unique<FixedCountChunkSource>(key, count);
  item.message_type = FilePathProvider::MessageType::kFile;
  auto producer = input_queue->CreateProducer();
  producer.Put(std::move(item));
  producer.Close();

  // Compute expected assignment counts using the same hash/weights.
  const uint64_t total_weight = 3;
  uint64_t cumA = 1;  // [0]
  size_t expectedA = 0;
  size_t expectedB = 0;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t h = static_cast<uint64_t>(
        absl::Hash<std::pair<std::string, size_t>>{}(std::make_pair(key, i)));
    const uint64_t r = h % total_weight;
    if (r < cumA)
      ++expectedA;
    else
      ++expectedB;
  }

  // Read outputs and verify view sizes.
  auto* qa =
      dynamic_cast<Queue<ChunkSourceWithPhase>*>(splitter.GetOutput("A"));
  auto* qb =
      dynamic_cast<Queue<ChunkSourceWithPhase>*>(splitter.GetOutput("B"));

  ASSERT_NE(qa, nullptr);
  ASSERT_NE(qb, nullptr);

  auto msgA = qa->Get();
  auto msgB = qb->Get();
  ASSERT_NE(msgA.source, nullptr);
  ASSERT_NE(msgB.source, nullptr);
  EXPECT_EQ(msgA.source->GetChunkCount(), expectedA);
  EXPECT_EQ(msgB.source->GetChunkCount(), expectedB);

  splitter.Stop();
}

TEST(ChunkSourceSplitterTest, BroadcastsInitialScanComplete) {
  auto input_queue = std::make_unique<Queue<ChunkSourceWithPhase>>(4);

  ChunkSourceSplitterConfig cfg;
  auto* outA = cfg.add_output();
  outA->set_name("A");
  cfg.add_weight(1);
  auto* outB = cfg.add_output();
  outB->set_name("B");
  cfg.add_weight(1);

  ChunkSourceSplitter splitter(cfg);
  splitter.SetInputs({input_queue.get()});
  splitter.Start();

  ChunkSourceWithPhase marker;
  marker.source = nullptr;
  marker.message_type = FilePathProvider::MessageType::kInitialScanComplete;
  auto producer = input_queue->CreateProducer();
  producer.Put(std::move(marker));
  producer.Close();

  auto* qa =
      dynamic_cast<Queue<ChunkSourceWithPhase>*>(splitter.GetOutput("A"));
  auto* qb =
      dynamic_cast<Queue<ChunkSourceWithPhase>*>(splitter.GetOutput("B"));

  auto m1 = qa->Get();
  auto m2 = qb->Get();
  EXPECT_EQ(m1.message_type,
            FilePathProvider::MessageType::kInitialScanComplete);
  EXPECT_EQ(m2.message_type,
            FilePathProvider::MessageType::kInitialScanComplete);
  EXPECT_EQ(m1.source, nullptr);
  EXPECT_EQ(m2.source, nullptr);

  splitter.Stop();
}

}  // namespace training
}  // namespace lczero
