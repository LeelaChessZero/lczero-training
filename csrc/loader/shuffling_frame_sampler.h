// ABOUTME: Stage that provides shuffled frames using reservoir sampling.
// ABOUTME: Takes V6TrainingData frames and outputs them in randomized order.
#pragma once

#include <cstddef>

#include "absl/container/fixed_array.h"
#include "absl/random/random.h"
#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "proto/data_loader_config.pb.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

using FrameType = V6TrainingData;

// Worker that implements reservoir sampling for training frames.
// Takes V6TrainingData frames as input and outputs them in shuffled order
// using reservoir sampling algorithm.
class ShufflingFrameSampler {
 public:
  using InputType = FrameType;
  using OutputType = FrameType;

  ShufflingFrameSampler(Queue<InputType>* input_queue,
                        const ShufflingFrameSamplerConfig& config);

  Queue<OutputType>* output();

 private:
  void Worker();
  void MainSamplingLoop(absl::FixedArray<FrameType>& reservoir,
                        Queue<OutputType>::Producer& producer);

  Queue<InputType>* input_queue_;
  Queue<OutputType> output_queue_;
  ThreadPool thread_pool_;
  size_t reservoir_size_per_thread_;
  absl::BitGen gen_;
};

}  // namespace training
}  // namespace lczero