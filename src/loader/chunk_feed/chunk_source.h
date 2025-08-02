#pragma once

#include <cstddef>
#include <string>

namespace lczero {
namespace training {

// Interface for providing training data chunks.
// A chunk source provides access to one or more chunks of training data.
// It's assumed that all chunks in a source for one group for sorting purposes,
// therefore GetChunkSortKey() returns just one key for the entire source. This
// allows to know the key before reading/indexing the chunks.
class ChunkSource {
 public:
  virtual ~ChunkSource() = default;

  // Returns a sort key (e.g. filename or a timestamp).
  virtual std::string GetChunkSortKey() const = 0;

  // Indexes the chunks in the source. This is called before any chunk data is
  // requested.
  virtual void Index() = 0;

  // Returns the number of chunks in this source. Only called after Index().
  virtual size_t GetChunkCount() const = 0;

  // Returns the data for the chunk at the given index. Only called after
  // Index().
  virtual std::string GetChunkData(size_t index) = 0;
};

}  // namespace training
}  // namespace lczero