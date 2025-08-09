#pragma once

#include <chrono>
#include <optional>

namespace lczero {

// ABOUTME: Load metric that tracks seconds a process was active.
// ABOUTME: Uses StartLoad/StopLoad to mark active periods and accumulates load
// time.
class LoadMetric {
 public:
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<double>;

  LoadMetric() { Reset(); }

  // Starts tracking load from the given time point.
  void StartLoad(Clock::time_point now) {
    FlushLoad(now);
    uncounted_load_start_ = now;
  }

  // Stops tracking load at the given time point.
  void StopLoad(Clock::time_point now) {
    FlushLoad(now);
    uncounted_load_start_ = std::nullopt;
  }

  // Returns the total load in seconds.
  double LoadSeconds() const { return load_seconds_; }

  // Resets the load metric to initial state.
  void Reset() {
    load_seconds_ = 0.0;
    uncounted_load_start_ = std::nullopt;
  }

  // Merges another load metric into this one.
  void MergeFrom(const LoadMetric& other) {
    load_seconds_ += other.load_seconds_;
  }

  // Merges live data and resets the source.
  void MergeLive(LoadMetric&& other, Clock::time_point now) {
    other.FlushLoad(now);
    MergeFrom(other);
    other.load_seconds_ = 0.0;
    // Keep other.uncounted_load_start_ as set by FlushLoad
  }

 private:
  // Flushes any uncounted load time into load_seconds_.
  void FlushLoad(Clock::time_point now) {
    if (!uncounted_load_start_.has_value()) return;

    Duration elapsed = now - *uncounted_load_start_;
    load_seconds_ += elapsed.count();
    uncounted_load_start_ = now;
  }

  double load_seconds_;
  std::optional<Clock::time_point> uncounted_load_start_;
};

}  // namespace lczero