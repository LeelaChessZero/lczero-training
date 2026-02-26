#pragma once

#include <filesystem>
#include <string>

#include "loader/chunk_source/chunk_source.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {

// A chunk source that reads a single (potentially gzipped) file as a single
// chunk.
class RawFileChunkSource : public ChunkSource {
 public:
  RawFileChunkSource(const std::filesystem::path& filename,
                     ChunkSourceLoaderConfig::FrameFormat frame_format);
  ~RawFileChunkSource();

 private:
  std::string GetChunkSortKey() const override;
  size_t GetChunkCount() const override;
  std::optional<std::vector<FrameType>> GetChunkData(size_t index) override;

  std::string filename_;
  ChunkSourceLoaderConfig::FrameFormat frame_format_;
};

}  // namespace training
}  // namespace lczero
