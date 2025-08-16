#pragma once

#include <absl/strings/str_cat.h>
#include <absl/synchronization/mutex.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace lczero {

// 1 second period is exact. Other periods are powers of two.
enum class TimePeriod {
  kEmpty = -128,
  k1Millisecond = -10,
  k2Milliseconds,
  k4Milliseconds,
  k8Milliseconds,
  k16Milliseconds,
  k31Milliseconds,
  k63Milliseconds,
  k125Milliseconds,
  k250Milliseconds,
  k500Milliseconds,
  k1Second /* = 0 */,
  k2Seconds,
  k4Seconds,
  k8Seconds,
  k16Seconds,
  k32Seconds,
  k1Minute,
  k2Minutes,
  k4Minutes,
  k9Minutes,
  k17Minutes,
  k36Minutes,
  k1Hour,
  k2Hours,
  k5Hours,
  k9Hours,
  k18Hours,
  k36Hours,
  k3Days,
  k6Days,
  k12Days,
  k24Days,
  k49Days,
  k97Days,
  k194Days,
  k388Days,
  k777Days,
  k2Years,
  k4Years,
  k9Years,
  k17Years, /* = 29 */
  kAllTime = 127
};

// ExponentialAggregator metrics over exponentially increasing time periods.
//
// The template parameter `Metric` can be used in two ways:
// 1. Function-based: Pass Clear and UpdateFrom functions to constructor.
//    This is the new approach for protobuf-based metrics.
// 2. Method-based: The metric type has Reset() and MergeFrom() methods.
//    This maintains backward compatibility with existing C++ struct metrics.
//
// For method-based metrics, the type must satisfy:
// * It must have a `Reset()` method that clears its state.
// * It must have a `MergeFrom(const Metric& other)` method to merge another
//   metric into itself (used for bucket-to-bucket merging and live ingestion).
// * It must behave like a monoid (actually, unital magma is sufficient):
//   * Merging with a default-constructed (empty) metric is a no-op.
//   * The `MergeFrom` operation must be associative (actually, not really;
//   currently we always merge old to new). It does not need to be
//     commutative.
//   * Having a `std::swap(Metric&, Metric&)` method is also beneficial.
template <typename Metric, TimePeriod Resolution = TimePeriod::k16Milliseconds>
class ExponentialAggregator {
 public:
  using Duration = std::chrono::nanoseconds;
  using Clock = std::chrono::steady_clock;
  using ClearFn = std::function<void(Metric&)>;
  using UpdateFromFn = std::function<void(Metric&, const Metric&)>;
  static constexpr TimePeriod kResolution = Resolution;

  // Resets the aggregator, clearing all buckets and pending metrics.
  void Reset(Clock::time_point now = Clock::now());

  // Ingests the passed metric into the pending bucket using MergeFrom + Reset.
  void RecordMetrics(Metric&& metric);

  // Returns the latest completed metrics bucket for the given time period and
  // duration since that period finished last time. If now is nullopt, it
  // excludes the time since the last metrics flush.
  std::pair<Metric, Duration> GetBucketMetrics(
      TimePeriod period,
      std::optional<Clock::time_point> now = std::nullopt) const;

  // Returns the current metrics that have been collected for at least the
  // specified duration. Returns the metrics and the duration since the
  // beginning of the covered period. If `now` is nullopt, it excludes metrics
  // and the time since the last metrics flush.
  std::pair<Metric, Duration> GetAggregateEndingNow(
      Duration duration,
      std::optional<Clock::time_point> now = Clock::now()) const;

  // Flushes the current pending bucket into the exponential metrics and
  // advances time by the elapsed duration, potentially processing multiple
  // ticks. Returns the largest time period that was updated by this advance
  // (all smaller periods are also updated).
  //
  // Note: Advance() should typically be called more frequently than the tick
  // frequency. When called less frequently (advancing multiple ticks at once),
  // live statistics go into the first bucket and subsequent buckets are padded
  // with empty metrics.
  TimePeriod Advance(Clock::time_point now = Clock::now());

  constexpr Duration GetResolution() const { return kPeriodDuration; }

  // Primary constructor for new protobuf-based metrics.
  // Takes two free functions that define the metric's behavior.
  ExponentialAggregator(ClearFn clear_fn, UpdateFromFn update_from_fn);

  // Default constructor for backward compatibility with old C++ metrics.
  // This will only compile if `Metric` has Reset() and MergeFrom() methods.
  ExponentialAggregator();

 private:
  // Constexpr power of 2 using bit shifts
  static constexpr double constexpr_pow2(int exp) {
    if (exp >= 0) {
      return static_cast<double>(1LL << exp);
    } else {
      return 1.0 / static_cast<double>(1LL << (-exp));
    }
  }

  static constexpr Duration kPeriodDuration =
      std::chrono::duration_cast<Duration>(std::chrono::duration<double>(
          constexpr_pow2(static_cast<int>(Resolution))));

  static size_t GetBucketIndex(TimePeriod period) {
    return static_cast<size_t>(period) - static_cast<size_t>(Resolution);
  }

  // The aggregation strategy is analogous to a binary counter. `tick_count_`
  // represents the counter's value, and the `buckets_` array corresponds to its
  // bits, each covering an exponentially larger time period.
  //
  // Advancing time increments `tick_count_`. When a bit flips from 1 to 0,
  // its bucket's metric is merged (the "carry") into the next higher bucket.
  //
  // Note that buckets are never empty. A bucket whose corresponding bit in
  // `tick_count_` is '0' simply holds the last complete metric for its time
  // period. This ensures that a valid, historical metric is always available
  // for the bucket query.

  ClearFn clear_fn_;
  UpdateFromFn update_from_fn_;

  mutable absl::Mutex mutex_;
  size_t tick_count_ ABSL_GUARDED_BY(mutex_);
  // Buckets for each time period, starting from Resolution.
  std::vector<Metric> buckets_ ABSL_GUARDED_BY(mutex_);
  Clock::time_point last_tick_time_ ABSL_GUARDED_BY(mutex_);

  mutable absl::Mutex pending_bucket_mutex_ ABSL_ACQUIRED_AFTER(mutex_);
  Metric pending_bucket_ ABSL_GUARDED_BY(pending_bucket_mutex_);
};

template <typename Metric, TimePeriod Resolution>
ExponentialAggregator<Metric, Resolution>::ExponentialAggregator(
    ClearFn clear_fn, UpdateFromFn update_from_fn)
    : clear_fn_(std::move(clear_fn)),
      update_from_fn_(std::move(update_from_fn)),
      tick_count_(0),
      last_tick_time_(Clock::now()) {}

template <typename Metric, TimePeriod Resolution>
ExponentialAggregator<Metric, Resolution>::ExponentialAggregator()
    : tick_count_(0), last_tick_time_(Clock::now()) {
  // For backward compatibility, use metric methods if available
  if constexpr (requires(Metric& m, const Metric& other) {
                  m.Reset();
                  m.MergeFrom(other);
                }) {
    clear_fn_ = [](Metric& m) { m.Reset(); };
    update_from_fn_ = [](Metric& dest, const Metric& src) {
      dest.MergeFrom(src);
    };
  } else {
    static_assert(sizeof(Metric) == 0,
                  "Metric type must have Reset() and MergeFrom() methods or "
                  "use function-based constructor");
  }
}

template <typename Metric, TimePeriod Resolution>
void ExponentialAggregator<Metric, Resolution>::Reset(
    std::chrono::steady_clock::time_point now) {
  absl::MutexLock lock(&mutex_);
  tick_count_ = 0;
  buckets_.clear();
  last_tick_time_ = now;

  absl::MutexLock pending_bucket_lock(&pending_bucket_mutex_);
  clear_fn_(pending_bucket_);
}

template <typename Metric, TimePeriod Resolution>
void ExponentialAggregator<Metric, Resolution>::RecordMetrics(Metric&& metric) {
  absl::MutexLock lock(&pending_bucket_mutex_);
  update_from_fn_(pending_bucket_, metric);
  clear_fn_(metric);
}

template <typename Metric, TimePeriod Resolution>
auto ExponentialAggregator<Metric, Resolution>::GetBucketMetrics(
    TimePeriod period, std::optional<Clock::time_point> now) const
    -> std::pair<Metric, Duration> {
  absl::MutexLock lock(&mutex_);
  const size_t index = GetBucketIndex(period);
  const Duration duration_since_update =
      kPeriodDuration * (tick_count_ % (1ULL << index)) +
      (now.has_value() ? Duration(*now - last_tick_time_) : Duration::zero());
  if (index >= buckets_.size()) return {Metric(), duration_since_update};
  return {buckets_[index], duration_since_update};
}

template <typename Metric, TimePeriod Resolution>
auto ExponentialAggregator<Metric, Resolution>::GetAggregateEndingNow(
    Duration duration, std::optional<Clock::time_point> now) const
    -> std::pair<Metric, Duration> {
  Duration result_duration = Duration::zero();
  Metric result;

  {
    absl::MutexLock lock(&mutex_);
    if (now.has_value()) {
      // If we'll use pending bucket, remove its duration from the request.
      // The actual bucket we'll merge in the end as we have to merge newer
      // into older buckets.
      Duration duration_since_update = *now - last_tick_time_;
      duration -= duration_since_update;
      result_duration += duration_since_update;
    }

    if (duration > Duration::zero()) {
      // Convert the input `duration` into `num_ticks` (the number of base
      //    time periods), rounding up.
      const auto div = std::div(duration.count(), kPeriodDuration.count());
      const size_t num_ticks = div.quot + bool(div.rem);

      // To cover the remaining `duration`, we select the minimal set of active
      // historical buckets (where the corresponding bit in `tick_count_` is 1)
      // that, when combined, meet or exceed the target duration.
      //
      // 1. First we determine the bit width of `num_ticks` (e.g., for 13
      //    (1101b), the width is 4). Create a candidate set of ticks by
      //    masking `tick_count_` to this width. If this masked value is >=
      //    `num_ticks`, the corresponding set of buckets is sufficient.
      // 2. If the masked value from step 2 is insufficient, we include one
      //    additional bucket. It doesn't matter if it's active or not.
      size_t masked_ticks = [&]() {
        const auto bit_width = std::bit_width(num_ticks);
        const size_t mask = ~((~size_t{0}) << bit_width);
        const size_t candidate_ticks = tick_count_ & mask;
        if (candidate_ticks >= num_ticks) return candidate_ticks;
        // One additional tick is needed.
        return candidate_ticks | (size_t{1} << bit_width);
      }();

      while (masked_ticks) {
        // Start merging from the highest bit (older bucket) to the lowest.
        size_t idx = std::bit_width(masked_ticks) - 1;
        masked_ticks &= ~(1ULL << idx);
        if (idx < buckets_.size()) update_from_fn_(result, buckets_[idx]);
        result_duration += kPeriodDuration * (1ULL << idx);
      }
    }
  }

  if (now.has_value()) {
    // The pending bucket is merged last, as we have to merge newer into older
    // buckets.
    absl::MutexLock lock(&pending_bucket_mutex_);
    update_from_fn_(result, pending_bucket_);
  }

  return {result, result_duration};
}

template <typename Metric, TimePeriod Resolution>
auto ExponentialAggregator<Metric, Resolution>::Advance(Clock::time_point now)
    -> TimePeriod {
  absl::MutexLock lock(&mutex_);
  const int num_ticks_to_advance = (now - last_tick_time_) / kPeriodDuration;
  if (num_ticks_to_advance <= 0) return TimePeriod::kEmpty;

  last_tick_time_ += num_ticks_to_advance * kPeriodDuration;
  Metric live_carry;
  {
    // What was pending, now becomes carry. Pending bucket is cleared.
    absl::MutexLock pending_bucket_lock(&pending_bucket_mutex_);
    live_carry = std::move(pending_bucket_);
    clear_fn_(pending_bucket_);
  }

  const size_t initial_tick_count = tick_count_;

  auto one_tick = [&](Metric& carry) {
    ++tick_count_;

    for (size_t i = 0;; ++i) {
      const uint64_t interval_size = 1ULL << i;
      if ((tick_count_ % interval_size) != 0) break;
      while (i >= buckets_.size()) buckets_.emplace_back();
      // We always merge new into the old, so we swap the carry first, and then
      // merge into it.
      std::swap(carry, buckets_[i]);
      update_from_fn_(carry, buckets_[i]);
    }
  };

  // Carry the pending bucket into the first tick.
  one_tick(live_carry);
  // Then, if more than one tick is requested, we carry the empty bucket
  // through the remaining ticks.
  for (int i = 1; i < num_ticks_to_advance; ++i) {
    Metric empty_carry;
    one_tick(empty_carry);
  }

  // Largest time period is the highest bit that was flipped in the process.
  // To find it, we XOR the initial tick count with the current one, and
  // find the highest bit.
  return static_cast<TimePeriod>(
      std::bit_width(initial_tick_count ^ tick_count_) - 1 +
      static_cast<int>(Resolution));
}

}  // namespace lczero