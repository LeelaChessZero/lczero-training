#include "data_loader.h"

#include <absl/log/log.h>

namespace lczero {
namespace training {

DataLoader::DataLoader(const DataLoaderConfig& config)
    : file_path_provifer_(config.file_path_provider),
      chunk_source_loader_(file_path_provifer_.output(),
                           config.chunk_source_loader),
      shuffling_chunk_pool_(chunk_source_loader_.output(),
                            config.shuffling_chunk_pool),
      chunk_unpacker_(shuffling_chunk_pool_.output(), config.chunk_unpacker),
      shuffling_frame_sampler_(chunk_unpacker_.output(),
                               config.shuffling_frame_sampler),
      tensor_generator_(shuffling_frame_sampler_.output(),
                        config.tensor_generator) {}

TensorTuple DataLoader::GetNext() { return output()->Get(); }

Queue<TensorTuple>* DataLoader::output() { return tensor_generator_.output(); }

}  // namespace training
}  // namespace lczero