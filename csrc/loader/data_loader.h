#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <thread>

#include "loader/stages/chunk_source_loader.h"
#include "loader/stages/chunk_unpacker.h"
#include "loader/stages/file_path_provider.h"
#include "loader/stages/shuffling_chunk_pool.h"
#include "loader/stages/shuffling_frame_sampler.h"
#include "loader/stages/tensor_generator.h"
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

  void Start();
  TensorTuple GetNext();
  void Stop(bool graceful_drain = false);
  std::pair<std::string, float> GetBucketMetrics(int time_period,
                                                 bool include_pending) const;
  std::pair<std::string, float> GetAggregateEndingNow(
      float duration_seconds, bool include_pending) const;

  // Chunk anchor management methods.
  std::pair<std::string, int> ResetChunkAnchor();
  int ChunksSinceAnchor();
  std::string CurrentChunkAnchor();
  void SetChunkAnchor(std::string_view anchor);

 private:
  static DataLoaderConfig ParseConfig(
      const std::string& serialized_data_loader_config);
  Queue<TensorTuple>* output();
  void MetricsThread(std::stop_token stop_token);

  DataLoaderConfig config_;
  std::string file_path_provider_stage_name_;
  std::string chunk_source_loader_stage_name_;
  std::string shuffling_chunk_pool_stage_name_;
  std::string chunk_unpacker_stage_name_;
  std::string shuffling_frame_sampler_stage_name_;
  std::string tensor_generator_stage_name_;
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
