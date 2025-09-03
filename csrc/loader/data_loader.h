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
#include "proto/data_loader_config.pb.h"
#include "utils/metrics/exponential_aggregator.h"
#include "utils/tensor.h"

namespace lczero {
namespace training {

class DataLoader {
 public:
  using MetricsAggregator = ExponentialAggregator<DataLoaderMetricsProto,
                                                  TimePeriod::k250Milliseconds>;

  DataLoader(const std::string& serialized_data_loader_config);
  ~DataLoader();

  TensorTuple GetNext();
  void Close(bool graceful_drain = false);
  std::pair<std::string, float> GetBucketMetrics(int time_period,
                                                 bool include_pending) const;
  std::pair<std::string, float> GetAggregateEndingNow(
      float duration_seconds, bool include_pending) const;

 private:
  static DataLoaderConfig ParseConfig(
      const std::string& serialized_data_loader_config);
  Queue<TensorTuple>* output();
  void MetricsThread(std::stop_token stop_token);

  DataLoaderConfig config_;
  FilePathProvider file_path_provider_;
  ChunkSourceLoader chunk_source_loader_;
  ShufflingChunkPool shuffling_chunk_pool_;
  ChunkUnpacker chunk_unpacker_;
  ShufflingFrameSampler shuffling_frame_sampler_;
  TensorGenerator tensor_generator_;
  MetricsAggregator metrics_aggregator_;
  std::jthread metrics_thread_;
};

}  // namespace training
}  // namespace lczero