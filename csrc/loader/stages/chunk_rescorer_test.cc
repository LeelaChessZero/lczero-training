#include "loader/stages/chunk_rescorer.h"

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "loader/stages/training_chunk.h"
#include "proto/data_loader_config.pb.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

namespace {

std::vector<V6TrainingData> StubRescore(std::vector<V6TrainingData> frames,
                                        SyzygyTablebase*, float dist_temp,
                                        float dist_offset, float dtz_boost,
                                        int new_input_format) {
  (void)dist_offset;
  (void)dtz_boost;
  (void)new_input_format;
  for (auto& frame : frames) {
    frame.result_q = dist_temp;
  }
  return frames;
}

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

class ChunkRescorerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_queue_ = std::make_unique<Queue<TrainingChunk>>(10);
    config_.set_threads(1);
    config_.set_queue_capacity(10);
    config_.set_input("source");
    config_.set_syzygy_paths("");
    config_.set_dist_temp(0.75f);
    config_.set_dist_offset(0.1f);
    config_.set_dtz_boost(0.2f);
    config_.set_new_input_format(-1);
  }

  TrainingChunk MakeChunk(std::vector<V6TrainingData> frames,
                          std::string sort_key = "alpha", size_t index = 3,
                          uint32_t use = 7) {
    TrainingChunk chunk;
    chunk.sort_key = std::move(sort_key);
    chunk.index_within_sort_key = index;
    chunk.use_count = use;
    chunk.frames = std::move(frames);
    return chunk;
  }

  std::unique_ptr<Queue<TrainingChunk>> input_queue_;
  ChunkRescorerConfig config_;
};

TEST_F(ChunkRescorerTest, AppliesInjectedRescoreFunction) {
  StageRegistry registry;
  registry.AddStage("source", std::make_unique<PassthroughStage<TrainingChunk>>(
                                  input_queue_.get()));
  ChunkRescorer rescorer(config_, registry, StubRescore);

  rescorer.Start();

  V6TrainingData frame{};
  frame.result_q = 0.1f;
  auto producer = input_queue_->CreateProducer();
  producer.Put(MakeChunk({frame}));
  producer.Close();

  auto output_chunk = rescorer.output()->Get();
  ASSERT_EQ(output_chunk.frames.size(), 1);
  EXPECT_FLOAT_EQ(output_chunk.frames[0].result_q, config_.dist_temp());
  EXPECT_EQ(output_chunk.sort_key, "alpha");
  EXPECT_EQ(output_chunk.index_within_sort_key, 3u);
  EXPECT_EQ(output_chunk.use_count, 7u);

  rescorer.Stop();
}

TEST_F(ChunkRescorerTest, HandlesInputQueueClosure) {
  StageRegistry registry;
  registry.AddStage("source", std::make_unique<PassthroughStage<TrainingChunk>>(
                                  input_queue_.get()));
  ChunkRescorer rescorer(config_, registry, StubRescore);

  rescorer.Start();

  input_queue_->Close();

  EXPECT_THROW(rescorer.output()->Get(), QueueClosedException);

  rescorer.Stop();
}

}  // namespace training
}  // namespace lczero
