#pragma once

#include <cstddef>
#include <string>

#include "loader/chunk_feed/chunk_source_loader.h"
#include "loader/chunk_feed/chunk_unpacker.h"
#include "loader/chunk_feed/file_path_provider.h"
#include "loader/chunk_feed/shuffling_chunk_pool.h"
#include "loader/shuffling_frame_sampler.h"
#include "loader/tensor_generator.h"
#include "utils/tensor.h"

namespace lczero {
namespace training {

struct DataLoaderConfig {
  std::string training_data_path;
  size_t num_chunks_window;
  size_t batch_size = 1024;
  size_t reservoir_size_per_thread = 1000000;
};

class DataLoader {
 public:
  DataLoader(const DataLoaderConfig& config);

  Queue<TensorTuple>* output();

 private:
  DataLoaderConfig config_;
  FilePathProvider file_path_provifer_;
  ChunkSourceLoader chunk_source_loader_;
  ShufflingChunkPool shuffling_chunk_pool_;
  ChunkUnpacker chunk_unpacker_;
  ShufflingFrameSampler shuffling_frame_sampler_;
  TensorGenerator tensor_generator_;
};

}  // namespace training
}  // namespace lczero
