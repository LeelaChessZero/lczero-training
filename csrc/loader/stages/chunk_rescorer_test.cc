#include "loader/stages/chunk_rescorer.h"

#include <span>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "loader/stages/training_chunk.h"
#include "proto/data_loader_config.pb.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

// Declared here rather than in chunk_rescorer.h because it is exposed only for
// testing; the definition lives in chunk_rescorer.cc with external linkage.
// Fills `plies_until_progress` on each frame with the number of plies until the
// next frame (strictly after the current one) whose `rule50_count` is 0. The
// tail is filled based on the adjudication bit (invariance_info bit 5) of the
// last frame: adjudicated -> 0xff sentinel; otherwise behaves as if a virtual
// progress frame sat one past the end (last -> 1, second-to-last -> 2, ...).
void FillPliesUntilProgress(std::span<FrameType> data);

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

namespace {

constexpr uint8_t kAdjudicatedBit = 1u << 5;

std::vector<FrameType> MakeFrames(std::vector<uint8_t> rule50_counts,
                                  bool adjudicated) {
  std::vector<FrameType> frames(rule50_counts.size());
  for (size_t i = 0; i < rule50_counts.size(); ++i) {
    frames[i].rule50_count = rule50_counts[i];
    frames[i].invariance_info = 0;
  }
  if (adjudicated && !frames.empty()) {
    frames.back().invariance_info |= kAdjudicatedBit;
  }
  return frames;
}

std::vector<uint8_t> ExtractPpp(const std::vector<FrameType>& frames) {
  std::vector<uint8_t> out;
  out.reserve(frames.size());
  for (const auto& f : frames) out.push_back(f.plies_until_progress);
  return out;
}

}  // namespace

TEST(FillPliesUntilProgressTest, MixedSequenceNotAdjudicated) {
  // Zeros at indices 1 and 3. Last frame has no zero strictly to its right,
  // and is not adjudicated, so it falls back to the "virtual zero" rule.
  auto frames = MakeFrames({3, 0, 2, 0}, /*adjudicated=*/false);
  FillPliesUntilProgress(frames);
  EXPECT_EQ(ExtractPpp(frames), (std::vector<uint8_t>{1, 2, 1, 1}));
}

TEST(FillPliesUntilProgressTest, MixedSequenceAdjudicated) {
  // Same sequence but adjudicated on the last frame; tail (no zero to the
  // right of index 3) stays at the 0xff sentinel.
  auto frames = MakeFrames({3, 0, 2, 0}, /*adjudicated=*/true);
  FillPliesUntilProgress(frames);
  EXPECT_EQ(ExtractPpp(frames), (std::vector<uint8_t>{1, 2, 1, 0xff}));
}

TEST(FillPliesUntilProgressTest, AllNonZeroNotAdjudicated) {
  auto frames = MakeFrames({5, 5, 5, 5, 5}, /*adjudicated=*/false);
  FillPliesUntilProgress(frames);
  EXPECT_EQ(ExtractPpp(frames), (std::vector<uint8_t>{5, 4, 3, 2, 1}));
}

TEST(FillPliesUntilProgressTest, AllNonZeroAdjudicated) {
  auto frames = MakeFrames({5, 5, 5, 5, 5}, /*adjudicated=*/true);
  FillPliesUntilProgress(frames);
  EXPECT_EQ(ExtractPpp(frames),
            (std::vector<uint8_t>{0xff, 0xff, 0xff, 0xff, 0xff}));
}

TEST(FillPliesUntilProgressTest, AllNonZeroLongClampsAtFf) {
  // Length 300, no zeros, not adjudicated. Distance from index i is N-i,
  // clamped at 0xff.
  auto frames = MakeFrames(std::vector<uint8_t>(300, 7),
                           /*adjudicated=*/false);
  FillPliesUntilProgress(frames);
  for (size_t i = 0; i < frames.size(); ++i) {
    const uint8_t expected =
        static_cast<uint8_t>(std::min<size_t>(frames.size() - i, 0xff));
    EXPECT_EQ(frames[i].plies_until_progress, expected) << "i=" << i;
  }
}

TEST(FillPliesUntilProgressTest, EmptyChunk) {
  std::vector<FrameType> frames;
  FillPliesUntilProgress(frames);  // should not crash
  EXPECT_TRUE(frames.empty());
}

}  // namespace training
}  // namespace lczero
