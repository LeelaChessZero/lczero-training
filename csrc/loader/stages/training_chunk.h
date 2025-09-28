#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "libs/lc0/src/trainingdata/trainingdata_v6.h"

namespace lczero {
namespace training {

using FrameType = V6TrainingData;

struct TrainingChunk {
  std::vector<FrameType> frames;
  std::string sort_key;
  size_t index_within_sort_key = 0;
  uint32_t reshuffle_count = 0;
};

}  // namespace training
}  // namespace lczero
