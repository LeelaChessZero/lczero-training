#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
#include <thread>

#include "utils/metrics/exponential_aggregator.h"
#include "utils/metrics/group.h"
#include "utils/metrics/printer.h"

namespace lczero {

class CounterMetric {
 public:
  CounterMetric() : count_(0) {}
  CounterMetric(int count) : count_(count) {}

  void Reset() { count_ = 0; }

  void MergeFrom(const CounterMetric& other) { count_ += other.count_; }

  void Print(MetricPrinter& printer) const {
    printer.StartGroup("CounterMetric");
    printer.Print("count", static_cast<size_t>(count_));
    printer.EndGroup();
  }

  int count() const { return count_; }
  void set_count(int count) { count_ = count; }

 private:
  int count_;
};

class AverageMetric {
 public:
  AverageMetric() : sum_(0), count_(0) {}
  AverageMetric(double sum, int count) : sum_(sum), count_(count) {}

  void Reset() {
    sum_ = 0;
    count_ = 0;
  }

  void MergeFrom(const AverageMetric& other) {
    sum_ += other.sum_;
    count_ += other.count_;
  }

  void Print(MetricPrinter& printer) const {
    printer.StartGroup("AverageMetric");
    printer.Print("sum", std::to_string(sum_));
    printer.Print("count", static_cast<size_t>(count_));
    if (count_ > 0) {
      printer.Print("average", std::to_string(sum_ / count_));
    }
    printer.EndGroup();
  }

  double average() const { return count_ > 0 ? sum_ / count_ : 0.0; }
  void add_sample(double value) {
    sum_ += value;
    count_++;
  }

  double sum() const { return sum_; }
  int count() const { return count_; }

 private:
  double sum_;
  int count_;
};

class MaxMetric {
 public:
  MaxMetric() : max_value_(0), has_value_(false) {}
  MaxMetric(double max_value) : max_value_(max_value), has_value_(true) {}

  void Reset() {
    max_value_ = 0;
    has_value_ = false;
  }

  void MergeFrom(const MaxMetric& other) {
    if (other.has_value_) {
      if (!has_value_ || other.max_value_ > max_value_) {
        max_value_ = other.max_value_;
        has_value_ = true;
      }
    }
  }

  void Print(MetricPrinter& printer) const {
    printer.StartGroup("MaxMetric");
    if (has_value_) {
      printer.Print("max_value", std::to_string(max_value_));
      printer.Print("has_value", static_cast<size_t>(1));
    } else {
      printer.Print("has_value", static_cast<size_t>(0));
    }
    printer.EndGroup();
  }

  double max_value() const { return max_value_; }
  bool has_value() const { return has_value_; }
  void set_value(double value) {
    if (!has_value_ || value > max_value_) {
      max_value_ = value;
      has_value_ = true;
    }
  }

 private:
  double max_value_;
  bool has_value_;
};

// Optional value metric that demonstrates overshadowing behavior
class OptionalValueMetric {
 public:
  OptionalValueMetric() : value_(std::nullopt) {}
  OptionalValueMetric(int value) : value_(value) {}

  void Reset() { value_ = std::nullopt; }

  void MergeFrom(const OptionalValueMetric& other) {
    // Only copy the value if the other metric has one (overshadowing behavior)
    if (other.value_.has_value()) {
      value_ = other.value_;
    }
  }

  void Print(MetricPrinter& printer) const {
    printer.StartGroup("OptionalValueMetric");
    if (value_.has_value()) {
      printer.Print("value", std::to_string(value_.value()));
      printer.Print("has_value", static_cast<size_t>(1));
    } else {
      printer.Print("has_value", static_cast<size_t>(0));
    }
    printer.EndGroup();
  }

  std::optional<int> value() const { return value_; }
  bool has_value() const { return value_.has_value(); }
  void set_value(int value) { value_ = value; }

 private:
  std::optional<int> value_;
};

// Test MetricGroup functionality
class MetricGroupTest : public ::testing::Test {
 protected:
  using TestGroup = MetricGroup<CounterMetric, AverageMetric, MaxMetric>;
  TestGroup group_;
};

TEST_F(MetricGroupTest, InitialState) {
  // Test that metrics are initialized in their default state
  EXPECT_EQ(group_.Get<CounterMetric>().count(), 0);
  EXPECT_EQ(group_.Get<AverageMetric>().count(), 0);
  EXPECT_FALSE(group_.Get<MaxMetric>().has_value());
}

TEST_F(MetricGroupTest, GetMutable) {
  // Test getting mutable references and modifying them
  auto* counter = group_.GetMutable<CounterMetric>();
  counter->set_count(42);
  EXPECT_EQ(group_.Get<CounterMetric>().count(), 42);

  auto* average = group_.GetMutable<AverageMetric>();
  average->add_sample(10.0);
  average->add_sample(20.0);
  EXPECT_EQ(group_.Get<AverageMetric>().average(), 15.0);

  auto* max_metric = group_.GetMutable<MaxMetric>();
  max_metric->set_value(100.0);
  EXPECT_EQ(group_.Get<MaxMetric>().max_value(), 100.0);
}

TEST_F(MetricGroupTest, Reset) {
  // Set up some data
  group_.GetMutable<CounterMetric>()->set_count(42);
  group_.GetMutable<AverageMetric>()->add_sample(10.0);
  group_.GetMutable<MaxMetric>()->set_value(100.0);

  // Reset and verify everything is back to initial state
  group_.Reset();

  EXPECT_EQ(group_.Get<CounterMetric>().count(), 0);
  EXPECT_EQ(group_.Get<AverageMetric>().count(), 0);
  EXPECT_FALSE(group_.Get<MaxMetric>().has_value());
}

TEST_F(MetricGroupTest, MergeFromGroup) {
  // Set up source group
  TestGroup other;
  other.GetMutable<CounterMetric>()->set_count(10);
  other.GetMutable<AverageMetric>()->add_sample(5.0);
  other.GetMutable<MaxMetric>()->set_value(50.0);

  // Set up destination group
  group_.GetMutable<CounterMetric>()->set_count(20);
  group_.GetMutable<AverageMetric>()->add_sample(15.0);
  group_.GetMutable<MaxMetric>()->set_value(30.0);

  // Merge
  group_.MergeFrom(other);

  // Verify results
  EXPECT_EQ(group_.Get<CounterMetric>().count(), 30);      // 20 + 10
  EXPECT_EQ(group_.Get<AverageMetric>().average(), 10.0);  // (15 + 5) / 2
  EXPECT_EQ(group_.Get<MaxMetric>().max_value(), 50.0);    // max(30, 50)
}

TEST_F(MetricGroupTest, MergeFromSingleMetric) {
  // Set up initial state
  group_.GetMutable<CounterMetric>()->set_count(20);

  // Create a single metric to merge
  CounterMetric counter(15);

  // Merge single metric
  group_.MergeFrom(counter);

  // Verify result
  EXPECT_EQ(group_.Get<CounterMetric>().count(), 35);  // 20 + 15
}

TEST_F(MetricGroupTest, Print) {
  // Set up data
  group_.GetMutable<CounterMetric>()->set_count(42);
  group_.GetMutable<AverageMetric>()->add_sample(10.0);
  group_.GetMutable<AverageMetric>()->add_sample(20.0);
  group_.GetMutable<MaxMetric>()->set_value(100.0);

  std::string result = MetricToString(group_);

  // Should contain all metric names and values
  EXPECT_NE(result.find("CounterMetric"), std::string::npos);
  EXPECT_NE(result.find("count=42"), std::string::npos);
  EXPECT_NE(result.find("AverageMetric"), std::string::npos);
  EXPECT_NE(result.find("average=15"), std::string::npos);  // (10+20)/2
  EXPECT_NE(result.find("MaxMetric"), std::string::npos);
  EXPECT_NE(result.find("max_value=100"), std::string::npos);
}

// Test MetricToString functionality
class MetricPrinterTest : public ::testing::Test {};

TEST_F(MetricPrinterTest, StringMetricPrinter) {
  std::string output;
  StringMetricPrinter printer(&output);

  printer.StartGroup("test_group");
  printer.Print("metric1", std::string("value1"));
  printer.Print("metric2", std::string("42"));
  printer.EndGroup();

  EXPECT_EQ(output, "test_group={metric1=value1, metric2=42}");
}

TEST_F(MetricPrinterTest, MultipleGroups) {
  std::string output;
  StringMetricPrinter printer(&output);

  printer.StartGroup("group1");
  printer.Print("metric1", std::string("value1"));
  printer.EndGroup();

  printer.StartGroup("group2");
  printer.Print("metric2", std::string("value2"));
  printer.EndGroup();

  EXPECT_EQ(output, "group1={metric1=value1}, group2={metric2=value2}");
}

TEST_F(MetricPrinterTest, EmptyGroup) {
  std::string output;
  StringMetricPrinter printer(&output);

  printer.StartGroup("empty_group");
  printer.EndGroup();

  EXPECT_EQ(output, "empty_group={}");
}

TEST_F(MetricPrinterTest, SizeTOverload) {
  std::string output;
  StringMetricPrinter string_printer(&output);
  MetricPrinter& printer = string_printer;  // Use base class interface

  printer.StartGroup("test_group");
  printer.Print("count", static_cast<size_t>(123));
  printer.EndGroup();

  EXPECT_EQ(output, "test_group={count=123}");
}

TEST_F(MetricPrinterTest, MetricToStringFunction) {
  CounterMetric counter(123);
  std::string result = MetricToString(counter);

  EXPECT_NE(result.find("CounterMetric"), std::string::npos);
  EXPECT_NE(result.find("123"), std::string::npos);
}

class ExponentialAggregatorTest : public ::testing::Test {
 protected:
  using TestMetric =
      MetricGroup<CounterMetric, AverageMetric, OptionalValueMetric>;
  using TestAggregator =
      ExponentialAggregator<TestMetric, TimePeriod::k16Milliseconds>;

  void SetUp() override {
    // Create a fresh aggregator for each test to avoid state contamination
    aggregator_ = std::make_unique<TestAggregator>();
    start_time_ = TestAggregator::Clock::now();
    aggregator_->Reset(start_time_);
  }

  std::unique_ptr<TestAggregator> aggregator_;
  TestAggregator::Clock::time_point start_time_;
};

TEST_F(ExponentialAggregatorTest, RecordMetrics) {
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  metric.GetMutable<AverageMetric>()->add_sample(5.0);

  // Update live metrics
  aggregator_->RecordMetrics(std::move(metric));

  // The original metric should be reset after move
  EXPECT_EQ(metric.Get<CounterMetric>().count(), 0);
  EXPECT_EQ(metric.Get<AverageMetric>().count(), 0);

  // Get live metrics to verify they were updated
  auto [live_metrics, age] =
      aggregator_->GetAggregateEndingNow(TestAggregator::Duration::zero());
  EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), 10);
  EXPECT_EQ(live_metrics.Get<AverageMetric>().average(), 5.0);
}

TEST_F(ExponentialAggregatorTest, MultipleUpdatesLiveMetrics) {
  // Update multiple times
  for (int i = 1; i <= 5; ++i) {
    TestMetric metric;
    metric.GetMutable<CounterMetric>()->set_count(i);
    metric.GetMutable<AverageMetric>()->add_sample(i * 2.0);
    aggregator_->RecordMetrics(std::move(metric));
  }

  // Get live metrics
  auto [live_metrics, age] =
      aggregator_->GetAggregateEndingNow(TestAggregator::Duration::zero());
  EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), 15);  // 1+2+3+4+5
  EXPECT_EQ(live_metrics.Get<AverageMetric>().average(),
            6.0);  // (2+4+6+8+10)/5
}

TEST_F(ExponentialAggregatorTest, Advance) {
  // Add some live metrics
  TestMetric metric;
  metric.GetMutable<CounterMetric>()->set_count(10);
  aggregator_->RecordMetrics(std::move(metric));

  // Advance to move live metrics to buckets
  auto tick_time = start_time_ + aggregator_->GetResolution();
  auto period = aggregator_->Advance(tick_time);

  // Should return the base time period
  EXPECT_EQ(period, TimePeriod::k16Milliseconds);

  // Live metrics should be empty after tick
  auto [live_metrics, age] = aggregator_->GetAggregateEndingNow(
      TestAggregator::Duration::zero(), tick_time);
  EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), 0);
}

TEST_F(ExponentialAggregatorTest, MultipleAdvances) {
  // Add metrics and tick multiple times to test bucket management
  auto current_time = start_time_;
  for (const auto expected_period : {
           TimePeriod::k16Milliseconds,
           TimePeriod::k31Milliseconds,
           TimePeriod::k16Milliseconds,
           TimePeriod::k63Milliseconds,
           TimePeriod::k16Milliseconds,
           TimePeriod::k31Milliseconds,
           TimePeriod::k16Milliseconds,
           TimePeriod::k125Milliseconds,
       }) {
    TestMetric metric;
    metric.GetMutable<CounterMetric>()->set_count(1);
    aggregator_->RecordMetrics(std::move(metric));

    current_time += aggregator_->GetResolution();
    auto period = aggregator_->Advance(current_time);

    EXPECT_EQ(period, expected_period);
  }
}

TEST_F(ExponentialAggregatorTest, MultipleAdvancesThreeTicks) {
  // Add metrics and tick multiple times to test bucket management
  auto current_time = start_time_;
  for (const auto expected_period : {
           TimePeriod::k31Milliseconds,
           TimePeriod::k63Milliseconds,
           TimePeriod::k125Milliseconds,
           TimePeriod::k63Milliseconds,
       }) {
    TestMetric metric;
    metric.GetMutable<CounterMetric>()->set_count(1);
    aggregator_->RecordMetrics(std::move(metric));

    current_time += aggregator_->GetResolution() * 3;
    auto period = aggregator_->Advance(current_time);

    EXPECT_EQ(period, expected_period);
  }
}

TEST_F(ExponentialAggregatorTest, AggregationTest) {
  TestMetric metric;
  // We do 37 (0b100101) updates.
  for (int i = 0; i < 37; ++i) {
    metric.GetMutable<CounterMetric>()->set_count(i + 200);
    metric.GetMutable<OptionalValueMetric>()->set_value(i + 100);
    aggregator_->RecordMetrics(std::move(metric));
    start_time_ += aggregator_->GetResolution();
    aggregator_->Advance(start_time_);
  }

  // One more tick, but we don't advance this time.
  metric.GetMutable<CounterMetric>()->set_count(1001);
  metric.GetMutable<OptionalValueMetric>()->set_value(1002);
  aggregator_->RecordMetrics(std::move(metric));

  const auto kRes = aggregator_->GetResolution();
  start_time_ += kRes / 3;  // Not a tick yet.

  auto check_completed_bucket = [&](TimePeriod period, int expected_count,
                                    std::optional<int> expected_value,
                                    TestAggregator::Duration expected_duration,
                                    bool include_pending = false) {
    auto [live_metrics, age] = aggregator_->GetBucketMetrics(
        period,
        include_pending ? std::make_optional(start_time_) : std::nullopt);
    EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), expected_count);
    EXPECT_EQ(live_metrics.Get<OptionalValueMetric>().has_value(),
              expected_value.has_value());
    if (expected_value.has_value()) {
      EXPECT_EQ(live_metrics.Get<OptionalValueMetric>().value(),
                expected_value.value());
    }
    EXPECT_EQ(age, expected_duration);
  };

  check_completed_bucket(TimePeriod::k16Milliseconds, 236, 136, kRes * 0);
  check_completed_bucket(TimePeriod::k31Milliseconds, 234 + 235, 135, kRes);
  check_completed_bucket(TimePeriod::k63Milliseconds, 232 + 233 + 234 + 235,
                         135, kRes);
  check_completed_bucket(TimePeriod::k125Milliseconds,
                         224 + 225 + 226 + 227 + 228 + 229 + 230 + 231, 131,
                         kRes * 5);
  check_completed_bucket(TimePeriod::k125Milliseconds,
                         224 + 225 + 226 + 227 + 228 + 229 + 230 + 231, 131,
                         kRes * 5 + kRes / 3, true);
  check_completed_bucket(TimePeriod::k250Milliseconds,
                         216 + 217 + 218 + 219 + 220 + 221 + 222 + 223 + 224 +
                             225 + 226 + 227 + 228 + 229 + 230 + 231,
                         131, kRes * 5);
  check_completed_bucket(TimePeriod::k500Milliseconds,
                         200 + 201 + 202 + 203 + 204 + 205 + 206 + 207 + 208 +
                             209 + 210 + 211 + 212 + 213 + 214 + 215 + 216 +
                             217 + 218 + 219 + 220 + 221 + 222 + 223 + 224 +
                             225 + 226 + 227 + 228 + 229 + 230 + 231,
                         131, kRes * 5);
  check_completed_bucket(TimePeriod::k1Second, 0, std::nullopt, kRes * 37);

  auto check_aggregate = [&](TestAggregator::Duration duration,
                             int expected_count,
                             std::optional<int> expected_value,
                             TestAggregator::Duration expected_duration,
                             bool include_pending = false) {
    auto [live_metrics, age] = aggregator_->GetAggregateEndingNow(
        duration,
        include_pending ? std::make_optional(start_time_) : std::nullopt);
    EXPECT_EQ(live_metrics.Get<CounterMetric>().count(), expected_count);
    EXPECT_EQ(live_metrics.Get<OptionalValueMetric>().has_value(),
              expected_value.has_value());
    if (expected_value.has_value()) {
      EXPECT_EQ(live_metrics.Get<OptionalValueMetric>().value(),
                expected_value.value());
    }
    EXPECT_EQ(age, expected_duration);
  };

  check_aggregate(TestAggregator::Duration::zero(), 0, std::nullopt, kRes * 0);
  check_aggregate(TestAggregator::Duration::zero(), 1001, 1002, kRes / 3, true);
  check_aggregate(kRes / 4, 1001, 1002, kRes / 3, true);
  check_aggregate(kRes / 10, 236, 136, kRes);
  check_aggregate(kRes * 45 / 10, 232 + 233 + 234 + 235 + 236, 136, kRes * 5);
  check_aggregate(kRes * 55 / 10,
                  224 + 225 + 226 + 227 + 228 + 229 + 230 + 231 + 232 + 233 +
                      234 + 235 + 236,
                  136, kRes * (5 + 8));
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}