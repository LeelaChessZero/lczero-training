#include "data_loader.h"

#include <absl/log/log.h>

#include <chrono>

#include "loader/data_loader_metrics.h"
#include "proto/training_config.pb.h"

namespace lczero {
namespace training {

DataLoaderConfig DataLoader::ParseConfig(const std::string& serialized_config) {
  DataLoaderConfig config;
  config.ParseFromString(serialized_config);
  return config;
}

DataLoader::DataLoader(const std::string& serialized_data_loader_config)
    : config_(ParseConfig(serialized_data_loader_config)),
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

DataLoader::~DataLoader() { file_path_provider_.Close(); }

TensorTuple DataLoader::GetNext() { return output()->Get(); }

Queue<TensorTuple>* DataLoader::output() { return tensor_generator_.output(); }

std::pair<std::string, float> DataLoader::GetBucketMetrics(
    int time_period, bool include_pending) const {
  auto [metrics, duration] = metrics_aggregator_.GetBucketMetrics(
      static_cast<TimePeriod>(time_period),
      include_pending ? std::make_optional(std::chrono::steady_clock::now())
                      : std::nullopt);
  float duration_seconds = std::chrono::duration<float>(duration).count();
  return {metrics.OutputAsString(), duration_seconds};
}

std::pair<std::string, float> DataLoader::GetAggregateEndingNow(
    float duration_seconds, bool include_pending) const {
  std::chrono::nanoseconds duration_ns =
      std::isinf(duration_seconds)
          ? std::chrono::nanoseconds::max()
          : std::chrono::nanoseconds(
                static_cast<int64_t>(duration_seconds * 1e9));
  auto [metrics, duration] = metrics_aggregator_.GetAggregateEndingNow(
      duration_ns, include_pending
                       ? std::make_optional(std::chrono::steady_clock::now())
                       : std::nullopt);
  float result_duration_seconds =
      std::chrono::duration<float>(duration).count();
  return {metrics.OutputAsString(), result_duration_seconds};
}

void DataLoader::MetricsThread(std::stop_token stop_token) {
  while (!stop_token.stop_requested()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    DataLoaderMetricsProto metrics;
    *metrics.mutable_file_path_provider() = file_path_provider_.FlushMetrics();
    *metrics.mutable_chunk_source_loader() =
        chunk_source_loader_.FlushMetrics();
    *metrics.mutable_shuffling_chunk_pool() =
        shuffling_chunk_pool_.FlushMetrics();
    *metrics.mutable_chunk_unpacker() = chunk_unpacker_.FlushMetrics();
    *metrics.mutable_shuffling_frame_sampler() =
        shuffling_frame_sampler_.FlushMetrics();
    *metrics.mutable_tensor_generator() = tensor_generator_.FlushMetrics();
    metrics_aggregator_.RecordMetrics(std::move(metrics));
    metrics_aggregator_.Advance(std::chrono::steady_clock::now());
  }
}

}  // namespace training
}  // namespace lczero