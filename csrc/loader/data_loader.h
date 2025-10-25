#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "loader/stages/stage_factory.h"
#include "proto/data_loader_config.pb.h"
#include "proto/stage_control.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/metrics/exponential_aggregator.h"
#include "utils/queue.h"
#include "utils/tensor.h"

namespace lczero {
namespace training {

class DataLoader {
 public:
  using MetricsAggregator = ExponentialAggregator<DataLoaderMetricsProto,
                                                  TimePeriod::k250Milliseconds>;

  explicit DataLoader(const std::string& serialized_data_loader_config);
  ~DataLoader();

  void Start();
  TensorTuple GetNext(std::string_view alias);
  void Stop();
  std::pair<std::string, float> GetBucketMetrics(int time_period,
                                                 bool include_pending) const;
  std::pair<std::string, float> GetAggregateEndingNow(
      float duration_seconds, bool include_pending) const;

  void AddStage(const StageConfig& stage_config);
  void AddStages(const DataLoaderConfig& config);
  void AddStages(const std::string& serialized_data_loader_config);

  std::vector<std::pair<std::string, StageControlResponse>> SendControlMessage(
      const StageControlRequest& request);

 private:
  static DataLoaderConfig ParseConfig(
      const std::string& serialized_data_loader_config);
  void MetricsThread(std::stop_token stop_token);
  void BuildOutputMapping(const DataLoaderConfig& config);
  Queue<TensorTuple>* GetOutputQueue(std::string_view alias) const;

  StageRegistry stage_registry_;
  std::vector<std::pair<std::string, Queue<TensorTuple>*>> outputs_;
  MetricsAggregator metrics_aggregator_;
  std::jthread metrics_thread_;
  bool started_ = false;
  bool stopped_ = false;
};

}  // namespace training
}  // namespace lczero
