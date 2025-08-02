#include "loader/chunk_feed/rawfile_chunk_source.h"

#include <absl/log/log.h>

#include <fstream>
#include <stdexcept>

#include "utils/files.h"
#include "utils/gz.h"

namespace lczero {
namespace training {

RawFileChunkSource::RawFileChunkSource(const std::string_view filename)
    : filename_(filename) {}

RawFileChunkSource::~RawFileChunkSource() = default;

void RawFileChunkSource::Index() {}

std::string RawFileChunkSource::GetChunkSortKey() const { return filename_; }

size_t RawFileChunkSource::GetChunkCount() const { return 1; }

std::string RawFileChunkSource::GetChunkData(size_t index) {
  if (index != 0) {
    throw std::out_of_range("RawFileChunkSource only has one chunk (index 0)");
  }
  return ReadFileToString(filename_);
}

}  // namespace training
}  // namespace lczero