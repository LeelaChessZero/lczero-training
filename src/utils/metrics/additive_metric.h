#pragma once

#include <string>
#include <string_view>

namespace lczero {

// ABOUTME: Simple additive metric that accumulates values.
// ABOUTME: Not thread-safe, requires external synchronization.
template <typename T>
class AdditiveMetric {
 public:
  using ValueType = T;

  AdditiveMetric() : value_(T{}) {}

  // Adds a value to the metric.
  void Add(const T& value) { value_ += value; }

  // Returns the current accumulated value.
  T Get() const { return value_; }

  // Resets the metric to initial state.
  void Reset() { value_ = T{}; }

  // Merges another additive metric into this one.
  void MergeFrom(const AdditiveMetric<T>& other) { value_ += other.value_; }

  // Prints the metric value with the given name.
  template <typename MetricPrinter>
  void Print(MetricPrinter& printer,
             std::string_view name = "AdditiveMetric") const {
    printer.Print(name, value_);
  }

 private:
  T value_;
};

}  // namespace lczero