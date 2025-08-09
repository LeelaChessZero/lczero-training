#pragma once

#include <chrono>
#include <optional>
#include <string>

#include "src/utils/metrics/additive_metric.h"

namespace lczero {

// ABOUTME: Simple load metric that accumulates seconds of load time.
// ABOUTME: Use LoadMetricUpdater to manage timing and flush into this metric.
class LoadMetric : public AdditiveMetric<double> {
 public:
  using Clock = std::chrono::steady_clock;

  LoadMetric() = default;

  // Returns the total load in seconds.
  double LoadSeconds() const { return Get(); }

  // Prints the metric value with the given name.
  template <typename MetricPrinter>
  void Print(MetricPrinter& printer,
             std::string_view name = "LoadMetric") const {
    AdditiveMetric<double>::Print(printer, name);
  }

 private:
  friend class LoadMetricUpdater;
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
    metric_->Add(elapsed.count());
    uncounted_load_start_ = now;
  }

 private:
  LoadMetric* metric_;
  std::optional<Clock::time_point> uncounted_load_start_;
};

}  // namespace lczero