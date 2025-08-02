#include "data_loader.h"

#include <absl/log/log.h>

namespace lczero {
namespace training {

DataLoader::DataLoader(const DataLoaderConfig& config)
    : config_(config),
      chunk_set_(ChunkSetOptions{
          .chunks_window = config_.num_chunks_window,
      }) {
  // Initialize file discovery with the training data path
  file_discovery_.AddDirectory(config_.training_data_path,
                               [](std::span<const FileDiscovery::File> files) {
                                 // Handle newly discovered files
                                 for (const auto& file : files) {
                                   LOG(INFO) << "Discovered file: "
                                             << file.filepath.string();
                                 }
                               });
}

}  // namespace training
}  // namespace lczero