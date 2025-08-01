#pragma once

#include <string>
#include <string_view>

#include "loader/chunk_feed/chunk_source.h"

namespace lczero {
namespace training {

class RawFileChunkSource : public ChunkSource {
 public:
  RawFileChunkSource(const std::string_view filename);
  ~RawFileChunkSource();

 private:
  std::string GetChunkSortKey() const override;
  void Index() override;
  size_t GetChunkCount() const override;
  std::string GetChunkData(size_t index) override;

  std::string filename_;
};

}  // namespace training
}  // namespace lczero