#include "data_loader.h"

#include <absl/log/log.h>

#include <chrono>

#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {

DataLoaderConfig DataLoader::ParseConfig(const std::string& config_string) {
  DataLoaderConfig config;
  config.ParseFromString(config_string);
  return config;
}

DataLoader::DataLoader(const std::string& config_string)
    : config_(ParseConfig(config_string)),
      file_path_provider_(config_.file_path_provider()),
      chunk_source_loader_(file_path_provider_.output(),
                           config_.chunk_source_loader()),
      shuffling_chunk_pool_(chunk_source_loader_.output(),
                            config_.shuffling_chunk_pool()),
      chunk_unpacker_(shuffling_chunk_pool_.output(), config_.chunk_unpacker()),
      shuffling_frame_sampler_(chunk_unpacker_.output(),
                               config_.shuffling_frame_sampler()),
      tensor_generator_(shuffling_frame_sampler_.output(),
                        config_.tensor_generator()),
      metrics_aggregator_(
          [](DataLoaderMetricsProto& m) { m.Clear(); },
          [](DataLoaderMetricsProto& dest, const DataLoaderMetricsProto& src) {
            UpdateFrom(dest, src);
          }),
      metrics_thread_(
          [this](std::stop_token stop_token) { MetricsThread(stop_token); }) {}

TensorTuple DataLoader::GetNext() { return output()->Get(); }

Queue<TensorTuple>* DataLoader::output() { return tensor_generator_.output(); }

std::string DataLoader::GetStat() const {
  auto [metrics, duration] =
      metrics_aggregator_.GetBucketMetrics(TimePeriod::k1Second);
  return metrics.OutputAsString();
}

void DataLoader::MetricsThread(std::stop_token stop_token) {
  while (!stop_token.stop_requested()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    DataLoaderMetricsProto metrics;
    *metrics.mutable_file_path_provider() = file_path_provider_.FlushMetrics();
    metrics_aggregator_.Advance(std::chrono::steady_clock::now());
  }
}

}  // namespace training
}  // namespace lczero