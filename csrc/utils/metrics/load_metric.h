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

  explicit LoadMetricUpdater(training::LoadMetricProto* metric,
                             Clock::time_point initial_time = Clock::now())
      : metric_(metric),
        last_flush_time_(initial_time),
        is_load_active_(false) {}

  // Starts tracking load from the given time point.
  void LoadStart(Clock::time_point now = Clock::now()) {
    Flush(now);
    is_load_active_ = true;
  }

  // Stops tracking load at the given time point.
  void LoadStop(Clock::time_point now = Clock::now()) {
    Flush(now);
    is_load_active_ = false;
  }

  // Flushes any uncounted load time into the metric.
  void Flush(Clock::time_point now = Clock::now()) {
    Duration elapsed = now - last_flush_time_;
    double elapsed_seconds = elapsed.count();

    metric_->set_total_seconds(metric_->total_seconds() + elapsed_seconds);
    if (is_load_active_) {
      metric_->set_load_seconds(metric_->load_seconds() + elapsed_seconds);
    }

    last_flush_time_ = now;
  }

 private:
  training::LoadMetricProto* metric_;
  Clock::time_point last_flush_time_;
  bool is_load_active_;
};

// UpdateFrom function for LoadMetricProto - simple additive behavior
inline void UpdateFrom(training::LoadMetricProto& dest,
                       const training::LoadMetricProto& src) {
  dest.set_load_seconds(dest.load_seconds() + src.load_seconds());
  dest.set_total_seconds(dest.total_seconds() + src.total_seconds());
}

}  // namespace lczero