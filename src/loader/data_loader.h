#pragma once

#include <cstddef>
#include <string>

#include "chunk_feed/discovery.h"

namespace lczero {
namespace training {

struct DataLoaderConfig {
  std::string training_data_path;
};

class DataLoader {
 public:
  DataLoader(const DataLoaderConfig& config);

 private:
  DataLoaderConfig config_;
  FileDiscovery file_discovery_;
};

}  // namespace training
}  // namespace lczero
