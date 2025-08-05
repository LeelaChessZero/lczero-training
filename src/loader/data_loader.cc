#include "data_loader.h"

#include <absl/log/log.h>

namespace lczero {
namespace training {

DataLoader::DataLoader(const DataLoaderConfig& config)
    : config_(config),
      file_path_provifer_(FilePathProviderOptions{
          .queue_capacity = 16, .directory = config_.training_data_path}),
      chunk_source_loader_(file_path_provifer_.output(),
                           ChunkSourceLoaderOptions{
                               .worker_threads = 1,
                               .output_queue_size = 16,
                           }),
      chunk_set_(chunk_source_loader_.output(),
                 ChunkSetOptions{
                     .chunk_pool_size = config_.num_chunks_window,
                 }) {}

}  // namespace training
}  // namespace lczero