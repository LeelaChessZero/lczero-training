#pragma once

namespace lczero {
namespace training {

class ChunkSource {
 public:
  virtual ~ChunkSource() = default;
  virtual std::string GetChunkSortKey() const = 0;
  virtual void Index() = 0;
  virtual size_t GetChunkCount() const = 0;
  virtual std::string GetChunkData(size_t index) = 0;
};

}  // namespace training
}  // namespace lczero