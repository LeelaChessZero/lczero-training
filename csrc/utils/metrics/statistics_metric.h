#pragma once

#include <absl/synchronization/mutex.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

namespace lczero {

template <typename T, bool kOneSamplePerTick = false>
class StatisticsMetric {
 public:
  using ValueType = T;
  using Clock = std::chrono::steady_clock;

  StatisticsMetric() { Reset(); }

  // Adds a sample to the statistics.
  void AddSample(const T& value) {
    min_ = std::min(min_, value);
    max_ = std::max(max_, value);
    latest_ = value;
    if constexpr (kOneSamplePerTick) {
      if (count_ > 0) return;
    }
    sum_ += value;
    ++count_;
  }

  // Returns the mean of all samples.
  double Mean() const {
    if (count_ == 0) return 0.0;
    return static_cast<double>(sum_) / static_cast<double>(count_);
  }

  // Returns the minimum value seen.
  T Min() const { return min_; }

  // Returns the maximum value seen.
  T Max() const { return max_; }

  // Returns the number of samples.
  size_t Count() const { return count_; }

  // Returns the most recent sample.
  T Latest() const { return latest_; }

  // Returns the sum of all samples.
  T Sum() const { return sum_; }

  void Reset() {
    min_ = std::numeric_limits<T>::max();
    max_ = std::numeric_limits<T>::lowest();
    sum_ = T{};
    count_ = 0;
    latest_ = T{};
  }

  void MergeFrom(const StatisticsMetric<T, kOneSamplePerTick>& other) {
    min_ = std::min(min_, other.min_);
    max_ = std::max(max_, other.max_);
    sum_ += other.sum_;
    latest_ = other.latest_;
    count_ += other.count_;
  }

  // Prints the statistics with the given name as a group.
  template <typename MetricPrinter>
  void Print(MetricPrinter& printer,
             std::string_view name = "StatisticsMetric") const {
    printer.StartGroup(name);
    printer.Print("min", min_);
    printer.Print("max", max_);
    printer.Print("count", count_);
    printer.Print("latest", latest_);
    printer.Print("mean", Mean());
    printer.EndGroup();
  }

 private:
  T min_;
  T max_;
  T sum_;
  size_t count_;
  T latest_;
};

}  // namespace lczero