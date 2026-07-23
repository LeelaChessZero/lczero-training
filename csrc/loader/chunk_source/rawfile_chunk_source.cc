#include "loader/chunk_source/rawfile_chunk_source.h"

#include <absl/log/log.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

#include "trainingdata/trainingdata_v6.h"
#include "utils/files.h"
#include "utils/gz.h"

namespace lczero {
namespace training {

RawFileChunkSource::RawFileChunkSource(
    const std::filesystem::path& filename,
    ChunkSourceLoaderConfig::FrameFormat frame_format)
    : filename_(filename), frame_format_(frame_format) {}

RawFileChunkSource::~RawFileChunkSource() = default;

std::string RawFileChunkSource::GetChunkSortKey() const {
  return std::filesystem::path(filename_).filename().string();
}

size_t RawFileChunkSource::GetChunkCount() const { return 1; }

std::optional<std::vector<FrameType>> RawFileChunkSource::GetChunkData(
    size_t index) {
  if (index != 0) return std::nullopt;
  std::string data = ReadFileToString(filename_);
  if (data.empty()) return std::nullopt;

  // FrameType has a trailing in-memory byte; on-disk records are V6/V7 sized.
  // Copy each record into the V6/V7 base of its FrameType slot.
  const auto parse = [&](auto tag) -> std::optional<std::vector<FrameType>> {
    using Record = typename decltype(tag)::type;
    if (data.size() % sizeof(Record) != 0) {
      LOG(WARNING) << "File " << filename_ << " size " << data.size()
                   << " is not a multiple of input frame size "
                   << sizeof(Record);
      return std::nullopt;
    }
    const size_t num_frames = data.size() / sizeof(Record);
    const auto* records = reinterpret_cast<const Record*>(data.data());
    std::vector<FrameType> result(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
      std::memcpy(static_cast<Record*>(&result[i]), &records[i],
                  sizeof(Record));
    }
    return result;
  };

  switch (frame_format_) {
    case ChunkSourceLoaderConfig::V7TrainingData:
      return parse(std::type_identity<V7TrainingData>{});
    default:
      return parse(std::type_identity<V6TrainingData>{});
  }
}

}  // namespace training
}  // namespace lczero
