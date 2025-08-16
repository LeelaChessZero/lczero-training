#pragma once

#include <chrono>
#include <optional>

#include "proto/training_metrics.pb.h"

namespace lczero {

// ABOUTME: Helper class to manage timing logic for LoadMetricProto.
// ABOUTME: Tracks active periods and flushes accumulated time to the protobuf
// metric.
class LoadMetricUpdater {
 public:
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double>;

  explicit LoadMetricUpdater(training::LoadMetricProto* metric)
      : metric_(metric) {}

  // Starts tracking load from the given time point.
  void LoadStart(Clock::time_point now = Clock::now()) {
    Flush(now);
    uncounted_load_start_ = now;
  }

  // Stops tracking load at the given time point.
  void LoadStop(Clock::time_point now = Clock::now()) {
    Flush(now);
    uncounted_load_start_ = std::nullopt;
  }

  // Flushes any uncounted load time into the metric.
  void Flush(Clock::time_point now = Clock::now()) {
    if (!uncounted_load_start_.has_value()) return;

    Duration elapsed = now - *uncounted_load_start_;
    metric_->set_load_seconds(metric_->load_seconds() + elapsed.count());
    uncounted_load_start_ = now;
  }

 private:
  training::LoadMetricProto* metric_;
  std::optional<Clock::time_point> uncounted_load_start_;
};

// UpdateFrom function for LoadMetricProto - simple additive behavior
inline void UpdateFrom(training::LoadMetricProto& dest,
                       const training::LoadMetricProto& src) {
  dest.set_load_seconds(dest.load_seconds() + src.load_seconds());
}

}  // namespace lczero