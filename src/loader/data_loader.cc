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
      shuffling_chunk_pool_(chunk_source_loader_.output(),
                            ShufflingChunkPoolOptions{
                                .chunk_pool_size = config_.num_chunks_window,
                            }),
      chunk_unpacker_(shuffling_chunk_pool_.output(),
                      ChunkUnpackerOptions{
                          .worker_threads = 1,
                          .output_queue_size = 16,
                      }),
      shuffling_frame_sampler_(
          chunk_unpacker_.output(),
          ShufflingFrameSamplerOptions{
              .num_worker_threads = 1,
              .reservoir_size_per_thread = config_.reservoir_size_per_thread,
              .output_queue_size = 16,
          }),
      tensor_generator_(shuffling_frame_sampler_.output(),
                        TensorGeneratorOptions{
                            .worker_threads = 1,
                            .batch_size = config_.batch_size,
                            .output_queue_size = 4,
                        }) {}

Queue<TensorTuple>* DataLoader::output() { return tensor_generator_.output(); }

}  // namespace training
}  // namespace lczero