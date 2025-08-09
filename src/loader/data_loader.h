#pragma once

#include <cstddef>
#include <string>
#include <thread>

#include "loader/chunk_feed/chunk_source_loader.h"
#include "loader/chunk_feed/chunk_unpacker.h"
#include "loader/chunk_feed/file_path_provider.h"
#include "loader/chunk_feed/shuffling_chunk_pool.h"
#include "loader/shuffling_frame_sampler.h"
#include "loader/tensor_generator.h"
#include "utils/metrics/exponential_aggregator.h"
#include "utils/metrics/group.h"
#include "utils/tensor.h"

namespace lczero {
namespace training {

using DataLoaderMetric = MetricGroup<FilePathProviderMetrics>;

struct DataLoaderConfig {
  FilePathProviderOptions file_path_provider;
  ChunkSourceLoaderOptions chunk_source_loader;
  ShufflingChunkPoolOptions shuffling_chunk_pool;
  ChunkUnpackerOptions chunk_unpacker;
  ShufflingFrameSamplerOptions shuffling_frame_sampler;
  TensorGeneratorOptions tensor_generator;
};

class DataLoader {
 public:
  DataLoader(const DataLoaderConfig& config);

  TensorTuple GetNext();

 private:
  Queue<TensorTuple>* output();
  FilePathProvider file_path_provider_;
  ChunkSourceLoader chunk_source_loader_;
  ShufflingChunkPool shuffling_chunk_pool_;
  ChunkUnpacker chunk_unpacker_;
  ShufflingFrameSampler shuffling_frame_sampler_;
  TensorGenerator tensor_generator_;
  ExponentialAggregator<DataLoaderMetric, TimePeriod::k250Milliseconds>
      metrics_aggregator_;
  std::jthread metrics_thread_;
};

}  // namespace training
}  // namespace lczero
