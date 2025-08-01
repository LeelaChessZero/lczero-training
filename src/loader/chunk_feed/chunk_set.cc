#include "loader/chunk_feed/chunk_source.h"

namespace lczero {
namespace training {

void ChunkSet::AddChunkSource(std::unique_ptr<ChunkSource> source) {
  if (phase_ == Phase::kInitialization) {
    uninitialized_sources_.push_back(std::move(source));
  } else {
    // Handle adding chunk sources in the feeding phase
  }
}

}  // namespace training
}  // namespace lczero
