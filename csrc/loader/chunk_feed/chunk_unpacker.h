// ABOUTME: Stage that unpacks chunks into V6TrainingData frames.
// ABOUTME: Converts stream of std::string chunks to V6TrainingData stream.
#pragma once

#include <string>

#include "libs/lc0/src/trainingdata/trainingdata_v6.h"
#include "proto/data_loader_config.pb.h"
#include "utils/queue.h"
#include "utils/thread_pool.h"

namespace lczero {
namespace training {

using FrameType = V6TrainingData;

// Worker pool that unpacks chunks into frames.
// Takes std::string chunks containing packed V6TrainingData as input and
// outputs individual V6TrainingData frames.
class ChunkUnpacker {
 public:
  using InputType = std::string;
  using OutputType = FrameType;

  ChunkUnpacker(Queue<InputType>* input_queue,
                const ChunkUnpackerConfig& config);

  Queue<OutputType>* output();

 private:
  void Worker();

  Queue<InputType>* input_queue_;
  Queue<OutputType> output_queue_;
  ThreadPool thread_pool_;
};

}  // namespace training
}  // namespace lczero