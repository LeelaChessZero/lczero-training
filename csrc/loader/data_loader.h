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
  TensorTuple GetNext();
  void Stop(bool graceful_drain = false);
  std::pair<std::string, float> GetBucketMetrics(int time_period,
                                                 bool include_pending) const;
  std::pair<std::string, float> GetAggregateEndingNow(
      float duration_seconds, bool include_pending) const;

  void AddStage(const StageConfig& stage_config);
  void AddStages(const DataLoaderConfig& config);
  void AddStages(const std::string& serialized_data_loader_config);

  std::vector<std::pair<std::string, StageControlResponse>> SendControlMessage(
      const StageControlRequest& request);

  std::pair<std::string, int> ResetChunkAnchor();
  int ChunksSinceAnchor();
  std::string CurrentChunkAnchor();
  void SetChunkAnchor(std::string_view anchor);

 private:
  static DataLoaderConfig ParseConfig(
      const std::string& serialized_data_loader_config);
  Stage::StageList BuildStageList() const;
  std::string ResolveStageName(const StageConfig& stage_config) const;
  void UpdateOutputQueue(const std::string& stage_name, Stage* stage);
  Queue<TensorTuple>* GetOutputQueue();
  const Queue<TensorTuple>* GetOutputQueue() const;
  void MetricsThread(std::stop_token stop_token);

  using StageEntry = std::pair<std::string, std::unique_ptr<Stage>>;

  std::vector<StageEntry> stages_;
  Queue<TensorTuple>* output_queue_ = nullptr;
  std::string output_stage_name_;
  MetricsAggregator metrics_aggregator_;
  std::jthread metrics_thread_;
  bool started_ = false;
  bool stopped_ = false;
};

}  // namespace training
}  // namespace lczero
