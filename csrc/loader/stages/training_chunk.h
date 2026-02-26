#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "loader/frame_type.h"

namespace lczero {
namespace training {

struct TrainingChunk {
  std::vector<FrameType> frames;
  std::string sort_key;
  size_t index_within_sort_key = 0;
  size_t global_index = 0;
  uint32_t use_count = 0;
};

struct CacheRequest {
  size_t global_index = 0;
  uint16_t next_use = 0;
  std::vector<FrameType> items;
};

}  // namespace training
}  // namespace lczero
