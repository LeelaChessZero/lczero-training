// ABOUTME: Stage that converts V6TrainingData frames into tensor batches.
// ABOUTME: Produces TensorTuple with tensors for training pipeline.
#pragma once

#include <cstddef>

#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "proto/data_loader_config.pb.h"
#include "utils/queue.h"
#include "utils/tensor.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

using FrameType = V6TrainingData;

// Worker pool that converts V6TrainingData frames into tensor batches.
// Takes individual V6TrainingData frames as input and outputs TensorTuple
// containing batched tensors in the format required for training.
class TensorGenerator {
 public:
  using InputType = FrameType;
  using OutputType = TensorTuple;

  TensorGenerator(Queue<InputType>* input_queue,
                  const TensorGeneratorConfig& config);

  Queue<OutputType>* output();

 private:
  void Worker();
  void ConvertFramesToTensors(const std::vector<FrameType>& frames,
                              TensorTuple& tensors);
  void ProcessPlanes(const std::vector<FrameType>& frames,
                     TypedTensor<float>& planes_tensor);

  Queue<InputType>* input_queue_;
  Queue<OutputType> output_queue_;
  ThreadPool thread_pool_;
  size_t batch_size_;
};

}  // namespace training
}  // namespace lczero