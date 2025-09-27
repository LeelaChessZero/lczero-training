#include "data_loader.h"

#include <absl/log/log.h>
#include <absl/strings/string_view.h>

#include <chrono>

#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"

namespace lczero {
namespace training {
namespace {

template <typename ConfigT>
const ConfigT& GetStageConfigOrDefault(
    const DataLoaderConfig& config, bool (StageConfig::*has_member)() const,
    const ConfigT& (StageConfig::*get_member)() const, std::string* stage_name,
    absl::string_view default_name) {
  for (const auto& stage : config.stage()) {
    if ((stage.*has_member)()) {
      if (stage_name != nullptr) {
        *stage_name =
            stage.has_name() ? stage.name() : std::string(default_name);
      }
      return (stage.*get_member)();
    }
  }
  if (stage_name != nullptr) {
    *stage_name = std::string(default_name);
  }
  static const ConfigT kDefault;
  return kDefault;
}

const FilePathProviderConfig& GetFilePathProviderConfig(
    const DataLoaderConfig& config, std::string* stage_name) {
  return GetStageConfigOrDefault<FilePathProviderConfig>(
      config, &StageConfig::has_file_path_provider,
      &StageConfig::file_path_provider, stage_name, "file_path_provider");
}

const ChunkSourceLoaderConfig& GetChunkSourceLoaderConfig(
    const DataLoaderConfig& config, std::string* stage_name) {
  return GetStageConfigOrDefault<ChunkSourceLoaderConfig>(
      config, &StageConfig::has_chunk_source_loader,
      &StageConfig::chunk_source_loader, stage_name, "chunk_source_loader");
}

const ShufflingChunkPoolConfig& GetShufflingChunkPoolConfig(
    const DataLoaderConfig& config, std::string* stage_name) {
  return GetStageConfigOrDefault<ShufflingChunkPoolConfig>(
      config, &StageConfig::has_shuffling_chunk_pool,
      &StageConfig::shuffling_chunk_pool, stage_name, "shuffling_chunk_pool");
}

const ChunkUnpackerConfig& GetChunkUnpackerConfig(
    const DataLoaderConfig& config, std::string* stage_name) {
  return GetStageConfigOrDefault<ChunkUnpackerConfig>(
      config, &StageConfig::has_chunk_unpacker, &StageConfig::chunk_unpacker,
      stage_name, "chunk_unpacker");
}

const ShufflingFrameSamplerConfig& GetShufflingFrameSamplerConfig(
    const DataLoaderConfig& config, std::string* stage_name) {
  return GetStageConfigOrDefault<ShufflingFrameSamplerConfig>(
      config, &StageConfig::has_shuffling_frame_sampler,
      &StageConfig::shuffling_frame_sampler, stage_name,
      "shuffling_frame_sampler");
}

const TensorGeneratorConfig& GetTensorGeneratorConfig(
    const DataLoaderConfig& config, std::string* stage_name) {
  return GetStageConfigOrDefault<TensorGeneratorConfig>(
      config, &StageConfig::has_tensor_generator,
      &StageConfig::tensor_generator, stage_name, "tensor_generator");
}

}  // namespace

DataLoaderConfig DataLoader::ParseConfig(const std::string& serialized_config) {
  DataLoaderConfig config;
  config.ParseFromString(serialized_config);
  return config;
}

DataLoader::DataLoader(const std::string& serialized_data_loader_config)
    : config_(ParseConfig(serialized_data_loader_config)),
      file_path_provider_stage_name_(),
      chunk_source_loader_stage_name_(),
      shuffling_chunk_pool_stage_name_(),
      chunk_unpacker_stage_name_(),
      shuffling_frame_sampler_stage_name_(),
      tensor_generator_stage_name_(),
      file_path_provider_(
          GetFilePathProviderConfig(config_, &file_path_provider_stage_name_)),
      chunk_source_loader_(
          GetChunkSourceLoaderConfig(config_, &chunk_source_loader_stage_name_),
          Stage::StageList{
              {file_path_provider_stage_name_, &file_path_provider_}}),
      shuffling_chunk_pool_(
          GetShufflingChunkPoolConfig(config_,
                                      &shuffling_chunk_pool_stage_name_),
          Stage::StageList{
              {file_path_provider_stage_name_, &file_path_provider_},
              {chunk_source_loader_stage_name_, &chunk_source_loader_}}),
      chunk_unpacker_(
          GetChunkUnpackerConfig(config_, &chunk_unpacker_stage_name_),
          Stage::StageList{
              {file_path_provider_stage_name_, &file_path_provider_},
              {chunk_source_loader_stage_name_, &chunk_source_loader_},
              {shuffling_chunk_pool_stage_name_, &shuffling_chunk_pool_}}),
      shuffling_frame_sampler_(
          GetShufflingFrameSamplerConfig(config_,
                                         &shuffling_frame_sampler_stage_name_),
          Stage::StageList{
              {file_path_provider_stage_name_, &file_path_provider_},
              {chunk_source_loader_stage_name_, &chunk_source_loader_},
              {shuffling_chunk_pool_stage_name_, &shuffling_chunk_pool_},
              {chunk_unpacker_stage_name_, &chunk_unpacker_}}),
      tensor_generator_(
          GetTensorGeneratorConfig(config_, &tensor_generator_stage_name_),
          Stage::StageList{
              {file_path_provider_stage_name_, &file_path_provider_},
              {chunk_source_loader_stage_name_, &chunk_source_loader_},
              {shuffling_chunk_pool_stage_name_, &shuffling_chunk_pool_},
              {chunk_unpacker_stage_name_, &chunk_unpacker_},
              {shuffling_frame_sampler_stage_name_,
               &shuffling_frame_sampler_}}),
      metrics_aggregator_(
          [](DataLoaderMetricsProto& m) { m.Clear(); },
          [](DataLoaderMetricsProto& dest, const DataLoaderMetricsProto& src) {
            UpdateFrom(dest, src);
          }) {
  LOG(INFO) << "DataLoader initialized (not started).";
}

void DataLoader::Start() {
  if (started_) {
    LOG(WARNING) << "DataLoader::Start called but loader already running.";
    return;
  }
  LOG(INFO) << "Starting DataLoader...";
  file_path_provider_.Start();
  chunk_source_loader_.Start();
  shuffling_chunk_pool_.Start();
  chunk_unpacker_.Start();
  shuffling_frame_sampler_.Start();
  tensor_generator_.Start();

  metrics_thread_ = std::jthread(
      [this](std::stop_token stop_token) { MetricsThread(stop_token); });
  started_ = true;
  stopped_ = false;
  LOG(INFO) << "DataLoader started.";
}

DataLoader::~DataLoader() { Stop(false); }

void DataLoader::Stop(bool graceful_drain) {
  if (stopped_) {
    return;
  }

  if (graceful_drain) {
    LOG(WARNING) << "Graceful drain is no longer supported. Proceeding with "
                    "immediate stop.";
  }

  LOG(INFO) << "Stopping DataLoader.";

  if (metrics_thread_.joinable()) {
    metrics_thread_.request_stop();
    metrics_thread_.join();
  }

  tensor_generator_.Stop();
  shuffling_frame_sampler_.Stop();
  chunk_unpacker_.Stop();
  shuffling_chunk_pool_.Stop();
  chunk_source_loader_.Stop();
  file_path_provider_.Stop();

  stopped_ = true;
  started_ = false;
  LOG(INFO) << "DataLoader stopped.";
}

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
    auto collect_metric = [&metrics](const std::string& name, Stage& stage) {
      StageMetricProto stage_metric = stage.FlushMetrics();
      stage_metric.set_name(name);
      *metrics.add_stage_metrics() = std::move(stage_metric);
    };

    collect_metric(file_path_provider_stage_name_, file_path_provider_);
    collect_metric(chunk_source_loader_stage_name_, chunk_source_loader_);
    collect_metric(shuffling_chunk_pool_stage_name_, shuffling_chunk_pool_);
    collect_metric(chunk_unpacker_stage_name_, chunk_unpacker_);
    collect_metric(shuffling_frame_sampler_stage_name_,
                   shuffling_frame_sampler_);
    collect_metric(tensor_generator_stage_name_, tensor_generator_);

    metrics_aggregator_.RecordMetrics(std::move(metrics));
    metrics_aggregator_.Advance(std::chrono::steady_clock::now());
  }
  LOG(INFO) << "Metrics thread stopping.";
}

std::pair<std::string, int> DataLoader::ResetChunkAnchor() {
  return shuffling_chunk_pool_.ResetAnchor();
}

int DataLoader::ChunksSinceAnchor() {
  return shuffling_chunk_pool_.ChunksSinceAnchor();
}

std::string DataLoader::CurrentChunkAnchor() {
  return shuffling_chunk_pool_.CurrentAnchor();
}

void DataLoader::SetChunkAnchor(std::string_view anchor) {
  shuffling_chunk_pool_.SetAnchor(anchor);
}

}  // namespace training
}  // namespace lczero
