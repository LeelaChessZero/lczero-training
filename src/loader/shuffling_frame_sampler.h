// ABOUTME: Stage that provides shuffled frames using reservoir sampling.
// ABOUTME: Takes V6TrainingData frames and outputs them in randomized order.
#pragma once

#include <cstddef>

#include "absl/container/fixed_array.h"
#include "absl/random/random.h"
#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

using FrameType = V6TrainingData;

struct ShufflingFrameSamplerOptions {
  size_t num_worker_threads = 1;  // Number of worker threads.
  size_t reservoir_size_per_thread =
      1000000;                    // Size of the reservoir for sampling.
  size_t output_queue_size = 16;  // Size of the output queue.
};

// Worker that implements reservoir sampling for training frames.
// Takes V6TrainingData frames as input and outputs them in shuffled order
// using reservoir sampling algorithm.
class ShufflingFrameSampler {
 public:
  using InputType = FrameType;
  using OutputType = FrameType;

  ShufflingFrameSampler(Queue<InputType>* input_queue,
                        const ShufflingFrameSamplerOptions& options);

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