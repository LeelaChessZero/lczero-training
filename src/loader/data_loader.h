#pragma once

#include <cstddef>
#include <string>

#include "chunk_feed/chunk_set.h"
#include "chunk_feed/chunk_source_feed.h"
#include "chunk_feed/discovery.h"

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
  ChunkSourceFeed chunk_source_feed_;
  ChunkSet chunk_set_;
};

}  // namespace training
}  // namespace lczero
