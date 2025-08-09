#include "data_loader.h"

#include <absl/log/log.h>

#include <chrono>

namespace lczero {
namespace training {

DataLoader::DataLoader(const DataLoaderConfig& config)
    : file_path_provider_(config.file_path_provider),
      chunk_source_loader_(file_path_provider_.output(),
                           config.chunk_source_loader),
      shuffling_chunk_pool_(chunk_source_loader_.output(),
                            config.shuffling_chunk_pool),
      chunk_unpacker_(shuffling_chunk_pool_.output(), config.chunk_unpacker),
      shuffling_frame_sampler_(chunk_unpacker_.output(),
                               config.shuffling_frame_sampler),
      tensor_generator_(shuffling_frame_sampler_.output(),
                        config.tensor_generator),
      metrics_thread_([this](std::stop_token stop_token) {
        while (!stop_token.stop_requested()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          file_path_provider_.RecordMetricsTo(&metrics_aggregator_);
        }
      }) {}

TensorTuple DataLoader::GetNext() { return output()->Get(); }

Queue<TensorTuple>* DataLoader::output() { return tensor_generator_.output(); }

}  // namespace training
}  // namespace lczero