#pragma once

#include <absl/strings/str_cat.h>

#include "utils/metrics/printer.h"

namespace lczero {

// Metric is a struct that implements the following interface:
// - void Reset();  // Resets the metric to its initial state.
// - void MergeFrom(const Metric& other);  // Merges another metric into this
// one. Note that the incoming always happens later in time, so if e.g. merge
// keeps the latest value, it should update the current value with the incoming
// one. Used for bucket-to-bucket merging.
// - void MergeLive(Metric&& other);  // Ingests live data and resets source.
// Most metrics can implement as: MergeFrom(other); other.Reset();
// - (optional) std::string_view name() const;
// - (optional) std::string ToString() const; // If provided, returns a string
// representation of the metric.

// Group several metric types together. This allows us to have a single
// `MetricGroup` that contains multiple different metrics.

template <typename... StatRecords>
class MetricGroup {
 public:
  MetricGroup() = default;

  // Calls reset on all stats.
  void Reset();

  // Merges each individual stat from `other` into this group.
  void MergeFrom(const MetricGroup<StatRecords...>& other);

  // Ingests live data from `other` group into this group, resetting the source.
  void MergeLive(MetricGroup<StatRecords...>&& other);

  // Merges a single stat from `other` into this group.
  template <typename T>
  void MergeFrom(const T& other);

  // Ingests live data from `other` into this group, resetting the source.
  template <typename T>
  void MergeLive(T&& other);

  // Gets a const reference to a specific stat record.
  template <typename T>
  const T& Get() const;

  // Gets a mutable pointer to a specific stat record.
  template <typename T>
  T* GetMutable();

  // Calls MetricPrinter for each stat in the group.
  void Print(MetricPrinter& printer) const;

 private:
  std::tuple<StatRecords...> stats_;
};

template <typename... StatRecords>
void MetricGroup<StatRecords...>::Reset() {
  (std::get<StatRecords>(stats_).Reset(), ...);
}

template <typename... StatRecords>
void MetricGroup<StatRecords...>::MergeFrom(
    const MetricGroup<StatRecords...>& other) {
  (std::get<StatRecords>(stats_).MergeFrom(std::get<StatRecords>(other.stats_)),
   ...);
}

template <typename... StatRecords>
void MetricGroup<StatRecords...>::MergeLive(
    MetricGroup<StatRecords...>&& other) {
  (std::get<StatRecords>(stats_).MergeLive(
       std::move(std::get<StatRecords>(other.stats_))),
   ...);
}

template <typename... StatRecords>
template <typename T>
void MetricGroup<StatRecords...>::MergeFrom(const T& other) {
  static_assert((std::is_same_v<T, StatRecords> || ...),
                "Type T must be one of the Stats types");
  std::get<T>(stats_).MergeFrom(other);
}

template <typename... StatRecords>
template <typename T>
void MetricGroup<StatRecords...>::MergeLive(T&& other) {
  static_assert(
      (std::is_same_v<std::remove_reference_t<T>, StatRecords> || ...),
      "Type T must be one of the Stats types");
  std::get<std::remove_reference_t<T>>(stats_).MergeLive(
      std::forward<T>(other));
}

template <typename... StatRecords>
template <typename T>
const T& MetricGroup<StatRecords...>::Get() const {
  static_assert((std::is_same_v<T, StatRecords> || ...),
                "Type T must be one of the Stats types");
  return std::get<T>(stats_);
}

template <typename... StatRecords>
template <typename T>
T* MetricGroup<StatRecords...>::GetMutable() {
  static_assert((std::is_same_v<T, StatRecords> || ...),
                "Type T must be one of the Stats types");
  return &std::get<T>(stats_);
}

template <typename... StatRecords>
void MetricGroup<StatRecords...>::Print(MetricPrinter& printer) const {
  (
      [&](const auto& stat) {
        if constexpr (requires { stat.Print(printer); }) {
          stat.Print(printer);
        }
      }(std::get<StatRecords>(stats_)),
      ...);
}

}  // namespace lczero