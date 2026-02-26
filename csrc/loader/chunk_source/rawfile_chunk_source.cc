#include "loader/chunk_source/rawfile_chunk_source.h"

#include <absl/log/log.h>

#include <fstream>
#include <stdexcept>

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

  const size_t input_size =
      frame_format_ == ChunkSourceLoaderConfig::V7TrainingData
          ? sizeof(V7TrainingData)
          : sizeof(V6TrainingData);
  if (data.size() % input_size != 0) {
    LOG(WARNING) << "File " << filename_ << " size " << data.size()
                 << " is not a multiple of input frame size " << input_size;
    return std::nullopt;
  }

  const size_t num_frames = data.size() / input_size;
  std::vector<V7TrainingData> result(num_frames);

  if (frame_format_ == ChunkSourceLoaderConfig::V7TrainingData) {
    std::memcpy(result.data(), data.data(), data.size());
  } else {
    const auto* v6_data = reinterpret_cast<const V6TrainingData*>(data.data());
    for (size_t i = 0; i < num_frames; ++i) {
      std::memcpy(&result[i], &v6_data[i], sizeof(V6TrainingData));
    }
  }
  return result;
}

}  // namespace training
}  // namespace lczero
