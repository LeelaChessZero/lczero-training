#pragma once

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "proto/data_loader_config.pb.h"
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

// Helper to convert QueueConfig::OverflowBehavior to OverflowBehavior.
inline OverflowBehavior ToOverflowBehavior(
    QueueConfig::OverflowBehavior behavior) {
  switch (behavior) {
    case QueueConfig::BLOCK:
      return OverflowBehavior::BLOCK;
    case QueueConfig::DROP_NEW:
      return OverflowBehavior::DROP_NEW;
    case QueueConfig::KEEP_NEWEST:
      return OverflowBehavior::KEEP_NEWEST;
  }
  throw std::runtime_error(absl::StrCat("Unknown OverflowBehavior value: ",
                                        static_cast<int>(behavior)));
}

// Helper for stages that consume a single upstream queue.
template <typename ConfigT, typename InputT>
class SingleInputStage : virtual public Stage {
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

// Helper for stages that produce a single output queue.
template <typename OutputT>
class SingleOutputStage : virtual public Stage {
 public:
  Queue<OutputT>* output_queue() { return &output_queue_; }

  QueueBase* GetOutput(std::string_view name = "") override {
    if (name != output_name_) {
      throw std::runtime_error(absl::StrCat("Output name '", name,
                                            "' does not match configured '",
                                            output_name_, "'"));
    }
    return &output_queue_;
  }

 protected:
  explicit SingleOutputStage(const QueueConfig& config)
      : output_name_(config.name()),
        output_queue_(config.queue_capacity(),
                      ToOverflowBehavior(config.overflow_behavior())) {}

 private:
  std::string output_name_;
  Queue<OutputT> output_queue_;
};

}  // namespace training
}  // namespace lczero
