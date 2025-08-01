#pragma once

namespace lczero {
namespace ice_skate {

class ChunkSource {
 public:
  virtual ~ChunkSource() = default;
  virtual uint64_t GetChunkSortKey() const = 0;
  virtual void Index() = 0;
  virtual size_t GetChunkCount() const = 0;
  virtual std::string GetChunkData(size_t index) = 0;
};

}  // namespace ice_skate
}  // namespace lczero