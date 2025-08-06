#pragma once

#include <cstddef>
#include <string>

#include "loader/chunk_feed/chunk_source_loader.h"
#include "loader/chunk_feed/chunk_unpacker.h"
#include "loader/chunk_feed/file_path_provider.h"
#include "loader/chunk_feed/shuffling_chunk_pool.h"

namespace lczero {
namespace training {

struct DataLoaderConfig {
  std::string training_data_path;
  size_t num_chunks_window;
};

class DataLoader {
 public:
  DataLoader(const DataLoaderConfig& config);

 private:
  DataLoaderConfig config_;
  FilePathProvider file_path_provifer_;
  ChunkSourceLoader chunk_source_loader_;
  ShufflingChunkPool shuffling_chunk_pool_;
  ChunkUnpacker chunk_unpacker_;
};

}  // namespace training
}  // namespace lczero
