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

class ChunkRescorerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    input_queue_ = std::make_unique<Queue<TrainingChunk>>(10);
    config_.set_threads(1);
    config_.mutable_output()->set_queue_capacity(10);
    config_.set_syzygy_paths("");
    config_.set_dist_temp(0.75f);
    config_.set_dist_offset(0.1f);
    config_.set_dtz_boost(0.2f);
    config_.set_new_input_format(-1);
  }

  TrainingChunk MakeChunk(std::vector<FrameType> frames,
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

TEST_F(ChunkRescorerTest, HandlesInputQueueClosure) {
  ChunkRescorer rescorer(config_);
  rescorer.SetInputs({input_queue_.get()});
  rescorer.Start();

  input_queue_->Close();

  EXPECT_THROW(rescorer.output_queue()->Get(), QueueClosedException);

  rescorer.Stop();
}

}  // namespace training
}  // namespace lczero
