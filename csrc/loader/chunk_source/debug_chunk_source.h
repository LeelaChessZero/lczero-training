#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>

#include "loader/chunk_source/chunk_source.h"

namespace lczero {
namespace training {

// DebugChunkSource synthesizes deterministic pseudo-random chunks for loader
// debugging. Each instance is identified by an integer id. The class produces
// a chunk count sampled from a normal distribution with the provided mean and
// mean / 4 standard deviation. The id serves as the seed, which keeps the
// number of chunks stable across runs. Individual chunks contain a
// pseudo-random number of FrameType frames (between one and 200) that are
// generated on demand. The generation seed depends on both the source id and
// chunk index. This lets shuffling logic exercise variable chunk sizes while
// keeping the content reproducible. Each generated frame is zero-initialized,
// but the first three entries of the planes array encode, respectively, the
// source id, the chunk index, and the frame index within the chunk. This makes
// it easy to reason about ordering and grouping when inspecting chunk payloads.
class DebugChunkSource : public ChunkSource {
 public:
  DebugChunkSource(uint64_t id, double mean_chunk_count);

 private:
  std::string GetChunkSortKey() const override;
  size_t GetChunkCount() const override;
  std::optional<std::vector<FrameType>> GetChunkData(size_t index) override;

  uint64_t id_;
  double mean_chunk_count_;
  mutable std::optional<size_t> cached_chunk_count_;
};

}  // namespace training
}  // namespace lczero
