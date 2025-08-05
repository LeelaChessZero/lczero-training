#pragma once

#include <cstddef>
#include <string>

#include "loader/chunk_feed/chunk_set.h"
#include "loader/chunk_feed/chunk_source_loader.h"
#include "loader/chunk_feed/file_path_provider.h"

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
  ChunkSet chunk_set_;
};

}  // namespace training
}  // namespace lczero
