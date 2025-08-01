#pragma once

#include <string>
#include <string_view>

#include "chunk_feed/chunk_source.h"

namespace lczero {
namespace ice_skate {

class GzChunkSource : public ChunkSource {
 public:
  GzChunkSource(const std::string_view filename);
  ~GzChunkSource();

 private:
  std::string GetChunkSortKey() const override;
  void Index() override;
  size_t GetChunkCount() const override;
  std::string GetChunkData(size_t index) override;

  std::string filename_;
};

}  // namespace ice_skate
}  // namespace lczero