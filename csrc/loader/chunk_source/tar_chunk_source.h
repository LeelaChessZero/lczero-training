#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "loader/chunk_source/chunk_source.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {

// A chunk source that reads a tar archive and provides access to its files as
// chunks. Each file in the tar is treated as a separate chunk.
class TarChunkSource : public ChunkSource {
 public:
  TarChunkSource(const std::filesystem::path& filename,
                 ChunkSourceLoaderConfig::FrameFormat frame_format);
  ~TarChunkSource() override;
  std::string GetChunkSortKey() const override;
  size_t GetChunkCount() const override;
  std::optional<std::vector<FrameType>> GetChunkData(size_t index) override;
  std::optional<std::string> GetChunkPrefix(size_t index, size_t max_bytes);

 private:
  // Performs one-time indexing during construction. Not part of the interface.
  void Index();
  struct FileEntry {
    long int offset;
    long int size;
    bool is_gzip;
  };

  FILE* file_ = nullptr;
  std::vector<FileEntry> files_;
  std::string filename_;
  ChunkSourceLoaderConfig::FrameFormat frame_format_;
};

}  // namespace training
}  // namespace lczero
