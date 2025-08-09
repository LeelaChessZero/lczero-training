#pragma once

#include <chrono>
#include <optional>

namespace lczero {

// ABOUTME: Simple load metric that accumulates seconds of load time.
// ABOUTME: Use LoadMetricUpdater to manage timing and flush into this metric.
class LoadMetric {
 public:
  using Clock = std::chrono::steady_clock;

  LoadMetric() : load_seconds_(0.0) {}

  // Returns the total load in seconds.
  double LoadSeconds() const { return load_seconds_; }

  // Resets the load metric to initial state.
  void Reset() { load_seconds_ = 0.0; }

  // Merges another load metric into this one.
  void MergeFrom(const LoadMetric& other) {
    load_seconds_ += other.load_seconds_;
  }

 private:
  friend class LoadMetricUpdater;
  double load_seconds_;
};

// ABOUTME: Helper class to manage timing logic for LoadMetric.
// ABOUTME: Tracks active periods and flushes accumulated time to the metric.
class LoadMetricUpdater {
 public:
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double>;

  explicit LoadMetricUpdater(LoadMetric* metric) : metric_(metric) {}

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
  void Flush(Clock::time_point now) {
    if (!uncounted_load_start_.has_value()) return;

    Duration elapsed = now - *uncounted_load_start_;
    metric_->load_seconds_ += elapsed.count();
    uncounted_load_start_ = now;
  }

 private:
  LoadMetric* metric_;
  std::optional<Clock::time_point> uncounted_load_start_;
};

}  // namespace lczero