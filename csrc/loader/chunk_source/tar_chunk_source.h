#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "loader/chunk_source/chunk_source.h"

namespace lczero {
namespace training {

// A chunk source that reads a tar archive and provides access to its files as
// chunks. Each file in the tar is treated as a separate chunk.
class TarChunkSource : public ChunkSource {
 public:
  TarChunkSource(const std::filesystem::path& filename);
  ~TarChunkSource() override;
  std::string GetChunkSortKey() const override;
  void Index() override;
  size_t GetChunkCount() const override;
  std::optional<std::string> GetChunkData(size_t index) override;
  std::optional<std::string> GetChunkPrefix(size_t index, size_t max_bytes);

 private:
  struct FileEntry {
    long int offset;
    long int size;
    bool is_gzip;
  };

  FILE* file_ = nullptr;
  std::vector<FileEntry> files_;
  std::string filename_;
};

}  // namespace training
}  // namespace lczero
