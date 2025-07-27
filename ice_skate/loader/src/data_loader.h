#pragma once
#include <cstddef>

namespace lczero {
namespace ice_skate {

class DataLoader {
 public:
  void SetTargetNumChunks(size_t target_num_chunks);
  void SetShuffleBufferSizeFrames(size_t size);
};

}  // namespace ice_skate
}  // namespace lczero
