#pragma once

#include <cstddef>
#include <string>

#include "chunk_feed/discovery.h"
#include "chunk_feed/chunk_set.h"

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
  FileDiscovery file_discovery_;
  ChunkSet chunk_set_;
};

}  // namespace training
}  // namespace lczero
