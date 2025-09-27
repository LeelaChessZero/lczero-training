#pragma once

#include <algorithm>
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
  virtual QueueBase* GetOutput() = 0;

  // Handles control-plane messages specific to the stage.
  virtual StageControlResponse Control(const StageControlRequest& request) = 0;
};

// Helper for stages that consume a single upstream queue.
template <typename InputT>
class SingleInputStage : public Stage {
 public:
  using StageList = std::vector<std::pair<std::string, Stage*>>;

 protected:
  explicit SingleInputStage(std::string_view input_queue_name,
                            const StageList& existing_stages)
      : input_queue_(nullptr) {
    if (input_queue_name.empty()) {
      throw std::runtime_error("Stage configuration is missing input binding.");
    }

    auto it = std::find_if(existing_stages.begin(), existing_stages.end(),
                           [input_queue_name](const auto& entry) {
                             return entry.first == input_queue_name;
                           });
    if (it == existing_stages.end()) {
      throw std::runtime_error(
          absl::StrCat("Stage input not found: ", input_queue_name));
    }

    QueueBase* raw_queue = it->second->GetOutput();
    auto* typed_queue = dynamic_cast<Queue<InputT>*>(raw_queue);
    if (typed_queue == nullptr) {
      throw std::runtime_error(absl::StrCat(
          "Stage input type mismatch for input: ", input_queue_name));
    }

    input_queue_ = typed_queue;
  }

  Queue<InputT>* input_queue() { return input_queue_; }

 private:
  Queue<InputT>* input_queue_;
};

}  // namespace training
}  // namespace lczero
