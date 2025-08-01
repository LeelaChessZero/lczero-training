#pragma once

#include <archive.h>

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "chunk_feed/chunk_source.h"

namespace lczero {
namespace ice_skate {

class TarChunkSource : public ChunkSource {
 public:
  TarChunkSource(const std::string_view filename);
  ~TarChunkSource();

 private:
  struct FileEntry {
    size_t offset;
    size_t size;
    bool is_gzip;
  };

  void Index() override;
  size_t GetChunkCount() const override;
  std::string GetChunkData(size_t index) override;

  archive* archive_ = nullptr;
  std::vector<FileEntry> files_;
  std::string filename_;
};

}  // namespace ice_skate
}  // namespace lczero