#include "loader/data_loader.h"

#include <absl/log/log.h>
#include <absl/strings/str_cat.h>

#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>

#include "loader/data_loader_metrics.h"

namespace lczero {
namespace training {
namespace {

StageControlResponse ExtractFirstChunkPoolResponse(
    const std::vector<std::pair<std::string, StageControlResponse>>&
        responses) {
  for (const auto& [name, response] : responses) {
    (void)name;
    if (response.has_chunk_pool_response()) {
      return response;
    }
  }
  return StageControlResponse();
}

}  // namespace

DataLoaderConfig DataLoader::ParseConfig(const std::string& serialized_config) {
  DataLoaderConfig config;
  config.ParseFromString(serialized_config);
  return config;
}

DataLoader::DataLoader(const std::string& serialized_data_loader_config)
    : metrics_aggregator_(
          [](DataLoaderMetricsProto& m) { m.Clear(); },
          [](DataLoaderMetricsProto& dest, const DataLoaderMetricsProto& src) {
            UpdateFrom(dest, src);
          }) {
  AddStages(serialized_data_loader_config);
  LOG(INFO) << "DataLoader initialized with " << stages_.size() << " stage(s).";
}

DataLoader::~DataLoader() { Stop(false); }

void DataLoader::AddStages(const std::string& serialized_data_loader_config) {
  AddStages(ParseConfig(serialized_data_loader_config));
}

void DataLoader::AddStages(const DataLoaderConfig& config) {
  for (const auto& stage_config : config.stage()) {
    AddStage(stage_config);
  }
  if (stages_.empty()) {
    throw std::runtime_error(
        "DataLoader pipeline must contain at least one "
        "stage.");
  }
  if (output_queue_ == nullptr) {
    throw std::runtime_error(
        "Pipeline does not expose a TensorTuple output queue.");
  }
}

void DataLoader::AddStage(const StageConfig& stage_config) {
  if (started_) {
    throw std::runtime_error("Cannot add stages after DataLoader has started.");
  }

  const std::string stage_name = ResolveStageName(stage_config);
  for (const auto& entry : stages_) {
    if (entry.first == stage_name) {
      throw std::runtime_error(
          absl::StrCat("Duplicate stage name detected: ", stage_name));
    }
  }

  Stage::StageList existing = BuildStageList();
  auto stage = CreateStage(stage_config, existing);
  Stage* stage_raw = stage.get();
  stages_.emplace_back(stage_name, std::move(stage));
  UpdateOutputQueue(stage_name, stage_raw);
}

Stage::StageList DataLoader::BuildStageList() const {
  Stage::StageList list;
  list.reserve(stages_.size());
  for (const auto& entry : stages_) {
    list.emplace_back(entry.first, entry.second.get());
  }
  return list;
}

std::string DataLoader::ResolveStageName(
    const StageConfig& stage_config) const {
  if (stage_config.has_name() && !stage_config.name().empty()) {
    return std::string(stage_config.name());
  }
  if (stage_config.has_file_path_provider()) {
    return "file_path_provider";
  }
  if (stage_config.has_chunk_source_loader()) {
    return "chunk_source_loader";
  }
  if (stage_config.has_shuffling_chunk_pool()) {
    return "shuffling_chunk_pool";
  }
  if (stage_config.has_chunk_unpacker()) {
    return "chunk_unpacker";
  }
  if (stage_config.has_shuffling_frame_sampler()) {
    return "shuffling_frame_sampler";
  }
  if (stage_config.has_tensor_generator()) {
    return "tensor_generator";
  }
  throw std::runtime_error(
      "Cannot resolve name for stage without a specific configuration.");
}

void DataLoader::UpdateOutputQueue(const std::string& stage_name,
                                   Stage* stage) {
  QueueBase* raw_output = stage->GetOutput();
  if (raw_output == nullptr) {
    return;
  }
  auto* tensor_queue = dynamic_cast<Queue<TensorTuple>*>(raw_output);
  if (tensor_queue != nullptr) {
    output_queue_ = tensor_queue;
    output_stage_name_ = stage_name;
  }
}

Queue<TensorTuple>* DataLoader::GetOutputQueue() {
  if (output_queue_ == nullptr) {
    throw std::runtime_error("Tensor output queue is not configured.");
  }
  return output_queue_;
}

const Queue<TensorTuple>* DataLoader::GetOutputQueue() const {
  if (output_queue_ == nullptr) {
    throw std::runtime_error("Tensor output queue is not configured.");
  }
  return output_queue_;
}

void DataLoader::Start() {
  if (started_) {
    LOG(WARNING) << "DataLoader::Start called but loader already running.";
    return;
  }
  LOG(INFO) << "Starting DataLoader with output stage '" << output_stage_name_
            << "'.";
  for (auto& [name, stage] : stages_) {
    LOG(INFO) << "Starting stage '" << name << "'.";
    stage->Start();
  }

  metrics_thread_ = std::jthread(
      [this](std::stop_token stop_token) { MetricsThread(stop_token); });
  started_ = true;
  stopped_ = false;
  LOG(INFO) << "DataLoader started.";
}

TensorTuple DataLoader::GetNext() { return GetOutputQueue()->Get(); }

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

  for (auto it = stages_.rbegin(); it != stages_.rend(); ++it) {
    LOG(INFO) << "Stopping stage '" << it->first << "'.";
    it->second->Stop();
  }

  stopped_ = true;
  started_ = false;
  LOG(INFO) << "DataLoader stopped.";
}

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
    for (auto& [name, stage] : stages_) {
      StageMetricProto stage_metric = stage->FlushMetrics();
      stage_metric.set_name(name);
      *metrics.add_stage_metrics() = std::move(stage_metric);
    }

    metrics_aggregator_.RecordMetrics(std::move(metrics));
    metrics_aggregator_.Advance(std::chrono::steady_clock::now());
  }
  LOG(INFO) << "Metrics thread stopping.";
}

std::vector<std::pair<std::string, StageControlResponse>>
DataLoader::SendControlMessage(const StageControlRequest& request) {
  std::vector<std::pair<std::string, StageControlResponse>> responses;
  responses.reserve(stages_.size());
  for (auto& [name, stage] : stages_) {
    std::optional<StageControlResponse> response = stage->Control(request);
    if (response.has_value()) {
      responses.emplace_back(name, std::move(*response));
    }
  }
  return responses;
}

std::pair<std::string, int> DataLoader::ResetChunkAnchor() {
  StageControlRequest request;
  request.mutable_chunk_pool_request()->set_reset_chunk_anchor(true);
  StageControlResponse response =
      ExtractFirstChunkPoolResponse(SendControlMessage(request));
  if (!response.has_chunk_pool_response()) {
    return {"", 0};
  }
  const auto& chunk_response = response.chunk_pool_response();
  return {std::string(chunk_response.chunk_anchor()),
          chunk_response.chunks_since_anchor()};
}

int DataLoader::ChunksSinceAnchor() {
  StageControlRequest request;
  request.mutable_chunk_pool_request();
  StageControlResponse response =
      ExtractFirstChunkPoolResponse(SendControlMessage(request));
  if (!response.has_chunk_pool_response()) {
    return 0;
  }
  return response.chunk_pool_response().chunks_since_anchor();
}

std::string DataLoader::CurrentChunkAnchor() {
  StageControlRequest request;
  request.mutable_chunk_pool_request();
  StageControlResponse response =
      ExtractFirstChunkPoolResponse(SendControlMessage(request));
  if (!response.has_chunk_pool_response()) {
    return "";
  }
  return std::string(response.chunk_pool_response().chunk_anchor());
}

void DataLoader::SetChunkAnchor(std::string_view anchor) {
  StageControlRequest request;
  request.mutable_chunk_pool_request()->set_set_chunk_anchor(
      std::string(anchor));
  StageControlResponse response =
      ExtractFirstChunkPoolResponse(SendControlMessage(request));
  if (!response.has_chunk_pool_response()) {
    LOG(WARNING) << "No stage accepted chunk anchor control request.";
  }
}

}  // namespace training
}  // namespace lczero
