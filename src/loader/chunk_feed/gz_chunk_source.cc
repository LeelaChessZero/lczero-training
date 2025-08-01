#include "chunk_feed/gz_chunk_source.h"

#include <absl/log/log.h>

#include <fstream>
#include <stdexcept>

#include "utils/gz.h"
#include "utils/readfile.h"

namespace lczero {
namespace training {

GzChunkSource::GzChunkSource(const std::string_view filename)
    : filename_(filename) {}

GzChunkSource::~GzChunkSource() = default;

void GzChunkSource::Index() {}

std::string GzChunkSource::GetChunkSortKey() const { return filename_; }

size_t GzChunkSource::GetChunkCount() const { return 1; }

std::string GzChunkSource::GetChunkData(size_t index) {
  if (index != 0) {
    throw std::out_of_range("GzChunkSource only has one chunk (index 0)");
  }
  std::string compressed_content = ReadFile(filename_);
  return GunzipBuffer(compressed_content);
}

}  // namespace training
}  // namespace lczero