#pragma once

#include "loader/chunk_feed/chunk_source.h"

namespace lczero {
namespace training {

class ChunkSet {
 public:
  enum class Phase {
    kInitialization,
    kFeeding,
  };

  void AddChunkSource(std::unique_ptr<ChunkSource> source);

 private:
  struct ChunkSourceItem {
    size_t start_chunk_index;
    std::unique_ptr<ChunkSource> source;
  };

  std::vector<std::unique_ptr<ChunkSource>> uninitialized_sources_;
  std::vector<ChunkSourceItem> chunk_sources_;

  Phase phase_ = Phase::kInitialization;
};

}  // namespace training
}  // namespace lczero