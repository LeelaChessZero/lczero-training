#include "data_loader.h"

#include <absl/log/log.h>

namespace lczero {
namespace training {

DataLoader::DataLoader(const DataLoaderConfig& config)
    : config_(config),
      file_discovery_(FileDiscoveryOptions{
          .queue_capacity = 16, .directory = config_.training_data_path}),
      chunk_set_(file_discovery_.output(),
                 ChunkSetOptions{
                     .chunks_window = config_.num_chunks_window,
                 }) {}

}  // namespace training
}  // namespace lczero