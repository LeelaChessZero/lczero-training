#pragma once

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "proto/stage_control.pb.h"
#include "proto/training_metrics.pb.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

// Base interface implemented by all loader stages.
class Stage {
 public:
  virtual ~Stage() = default;

  // Starts background workers owned by the stage.
  virtual void Start() = 0;

  // Requests the stage to stop and join background work.
  virtual void Stop() = 0;

  // Flushes stage-specific metrics and returns a snapshot.
  virtual StageMetricProto FlushMetrics() = 0;

  // Returns the output queue for downstream stages.
  virtual QueueBase* GetOutput(std::string_view name = "") = 0;

  // Handles control-plane messages specific to the stage.
  virtual std::optional<StageControlResponse> Control(
      const StageControlRequest& request) {
    (void)request;
    return std::nullopt;
  }
};

class StageRegistry {
 public:
  // Registers a new stage with the given name and takes ownership of it.
  void AddStage(std::string_view stage_name, std::unique_ptr<Stage> stage);

  // Returns the output queue for the specified stage.
  // If stage_name contains a dot (e.g., "stage.output"), splits it into stage
  // name and output name, passing the output name to Stage::GetOutput().
  // Returns nullptr if the stage is not found.
  QueueBase* GetStageOutput(std::string_view stage_name) const;

  template <typename StageT>
  Queue<StageT>* GetTypedStageOutput(std::string_view stage_name) const {
    QueueBase* raw_queue = GetStageOutput(stage_name);
    if (raw_queue == nullptr) return nullptr;
    auto* typed_queue = dynamic_cast<Queue<StageT>*>(raw_queue);
    if (!typed_queue) {
      throw std::runtime_error(
          absl::StrCat("Stage output type mismatch for stage: ", stage_name));
    }
    return typed_queue;
  }

  size_t size() const { return stages_.size(); }
  const std::vector<std::pair<std::string, std::unique_ptr<Stage>>>& stages()
      const {
    return stages_;
  }

 private:
  std::vector<std::pair<std::string, std::unique_ptr<Stage>>> stages_;
};

// Helper for stages that consume a single upstream queue.
template <typename ConfigT, typename InputT>
class SingleInputStage : public Stage {
 protected:
  explicit SingleInputStage(const ConfigT& config,
                            const StageRegistry& existing_stages)
      : input_queue_(nullptr) {
    input_queue_ = existing_stages.GetTypedStageOutput<InputT>(config.input());
    if (!input_queue_) {
      throw std::runtime_error(absl::StrCat("Input stage '", config.input(),
                                            "' not found or has wrong type."));
    }
  }

  Queue<InputT>* input_queue() { return input_queue_; }

 private:
  Queue<InputT>* input_queue_;
};

}  // namespace training
}  // namespace lczero
