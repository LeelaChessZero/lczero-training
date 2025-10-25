#include "loader/data_loader.h"

#include <absl/algorithm/container.h>
#include <absl/log/log.h>
#include <absl/strings/str_cat.h>

#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>

#include "loader/data_loader_metrics.h"

namespace lczero {
namespace training {
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
  LOG(INFO) << "DataLoader initialized with " << stage_registry_.size()
            << " stage(s).";
}

DataLoader::~DataLoader() { Stop(); }

void DataLoader::AddStages(const std::string& serialized_data_loader_config) {
  AddStages(ParseConfig(serialized_data_loader_config));
}

void DataLoader::AddStages(const DataLoaderConfig& config) {
  for (const auto& stage_config : config.stage()) AddStage(stage_config);
}

void DataLoader::AddStage(const StageConfig& stage_config) {
  if (started_) {
    throw std::runtime_error("Cannot add stages after DataLoader has started.");
  }

  auto stage = CreateStage(stage_config, stage_registry_);
  if (!stage_config.has_name()) {
    throw std::runtime_error("Stage configuration is missing name.");
  }
  LOG(INFO) << "Adding stage '" << stage_config.name() << "'.";
  stage_registry_.AddStage(stage_config.name(), std::move(stage));
}

void DataLoader::Start() {
  if (started_) {
    throw std::runtime_error("DataLoader has already been started.");
  }
  for (auto& [name, stage] : stage_registry_.stages()) {
    LOG(INFO) << "Starting stage '" << name << "'.";
    stage->Start();
  }

  metrics_thread_ = std::jthread(
      [this](std::stop_token stop_token) { MetricsThread(stop_token); });
  started_ = true;
  stopped_ = false;
  LOG(INFO) << "DataLoader started.";
}

// TensorTuple DataLoader::GetNext() { return GetOutputQueue()->Get(); }

void DataLoader::Stop() {
  if (stopped_) return;

  LOG(INFO) << "Stopping DataLoader.";

  if (metrics_thread_.joinable()) {
    metrics_thread_.request_stop();
    metrics_thread_.join();
  }

  for (auto& [name, stage] : stage_registry_.stages()) {
    LOG(INFO) << "Stopping stage '" << name << "'.";
    stage->Stop();
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
    for (auto& [name, stage] : stage_registry_.stages()) {
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
  responses.reserve(stage_registry_.size());
  for (auto& [name, stage] : stage_registry_.stages()) {
    std::optional<StageControlResponse> response = stage->Control(request);
    if (response.has_value()) {
      responses.emplace_back(name, std::move(*response));
    }
  }
  return responses;
}

}  // namespace training
}  // namespace lczero
