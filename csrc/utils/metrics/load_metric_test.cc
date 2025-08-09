#include "utils/metrics/load_metric.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>

#include "utils/metrics/exponential_aggregator.h"

namespace lczero {

class LoadMetricTest : public ::testing::Test {
 protected:
  using Clock = LoadMetric::Clock;

  void SetUp() override { start_time_ = Clock::now(); }

  Clock::time_point start_time_;
};

TEST_F(LoadMetricTest, BasicLoadMetric) {
  LoadMetric metric;
  EXPECT_EQ(metric.LoadSeconds(), 0.0);

  // Test MergeFrom
  LoadMetric other;
  LoadMetricUpdater other_updater(&other);
  other_updater.LoadStart(start_time_);
  other_updater.LoadStop(start_time_ + std::chrono::milliseconds(500));
  metric.MergeFrom(other);
  EXPECT_NEAR(metric.LoadSeconds(), 0.5, 1e-6);

  // Test Reset
  metric.Reset();
  EXPECT_EQ(metric.LoadSeconds(), 0.0);
}

TEST_F(LoadMetricTest, LoadMetricUpdaterBasic) {
  LoadMetric metric;
  LoadMetricUpdater updater(&metric);
  auto now = start_time_;

  // Start load and verify initial state
  updater.LoadStart(now);
  EXPECT_EQ(metric.LoadSeconds(), 0.0);

  // Advance time and stop load
  now += std::chrono::milliseconds(100);
  updater.LoadStop(now);
  EXPECT_NEAR(metric.LoadSeconds(), 0.1, 1e-6);

  // Start again
  now += std::chrono::milliseconds(50);
  updater.LoadStart(now);
  now += std::chrono::milliseconds(200);
  updater.LoadStop(now);
  EXPECT_NEAR(metric.LoadSeconds(), 0.3, 1e-6);  // 0.1 + 0.2
}

TEST_F(LoadMetricTest, LoadMetricUpdaterFlush) {
  LoadMetric metric;
  LoadMetricUpdater updater(&metric);
  auto now = start_time_;

  // Start load
  updater.LoadStart(now);
  now += std::chrono::milliseconds(100);

  // Flush should update the metric
  updater.Flush(now);
  EXPECT_NEAR(metric.LoadSeconds(), 0.1, 1e-6);

  // Continue loading
  now += std::chrono::milliseconds(50);
  updater.LoadStop(now);
  EXPECT_NEAR(metric.LoadSeconds(), 0.15, 1e-6);  // 0.1 + 0.05
}

TEST_F(LoadMetricTest, LoadMetricMerging) {
  LoadMetric metric1, metric2;
  LoadMetricUpdater updater1(&metric1), updater2(&metric2);
  auto now = start_time_;

  // Create load in metric1
  updater1.LoadStart(now);
  updater1.LoadStop(now + std::chrono::milliseconds(100));
  EXPECT_NEAR(metric1.LoadSeconds(), 0.1, 1e-6);

  // Create load in metric2
  updater2.LoadStart(now + std::chrono::milliseconds(50));
  updater2.LoadStop(now + std::chrono::milliseconds(150));
  EXPECT_NEAR(metric2.LoadSeconds(), 0.1, 1e-6);

  // Merge
  metric1.MergeFrom(metric2);
  EXPECT_NEAR(metric1.LoadSeconds(), 0.2, 1e-6);
  EXPECT_NEAR(metric2.LoadSeconds(), 0.1, 1e-6);  // Source unchanged
}

TEST_F(LoadMetricTest, LoadMetricMoveSemantics) {
  // Test that LoadMetric move semantics work correctly
  LoadMetric source;
  LoadMetricUpdater source_updater(&source);
  source_updater.LoadStart(start_time_);
  source_updater.LoadStop(start_time_ + std::chrono::milliseconds(100));
  EXPECT_NEAR(source.LoadSeconds(), 0.1, 1e-6);

  // Test move construction
  LoadMetric moved_constructed(std::move(source));
  EXPECT_NEAR(moved_constructed.LoadSeconds(), 0.1, 1e-6);

  // Test move assignment
  LoadMetric move_assigned;
  LoadMetric another_source;
  LoadMetricUpdater another_updater(&another_source);
  another_updater.LoadStart(start_time_);
  another_updater.LoadStop(start_time_ + std::chrono::milliseconds(50));
  EXPECT_NEAR(another_source.LoadSeconds(), 0.05, 1e-6);

  move_assigned = std::move(another_source);
  EXPECT_NEAR(move_assigned.LoadSeconds(), 0.05, 1e-6);

  // Test MergeFrom
  LoadMetric dest;
  dest.MergeFrom(moved_constructed);
  EXPECT_NEAR(dest.LoadSeconds(), 0.1, 1e-6);

  dest.MergeFrom(move_assigned);
  EXPECT_NEAR(dest.LoadSeconds(), 0.15, 1e-6);
}

class LoadMetricIntegrationTest : public ::testing::Test {
 protected:
  using TestAggregator =
      ExponentialAggregator<LoadMetric, TimePeriod::k16Milliseconds>;
  using Clock = TestAggregator::Clock;

  void SetUp() override {
    aggregator_ = std::make_unique<TestAggregator>();
    start_time_ = Clock::now();
    aggregator_->Reset(start_time_);
  }

  std::unique_ptr<TestAggregator> aggregator_;
  Clock::time_point start_time_;
};

TEST_F(LoadMetricIntegrationTest, RecordMetricsWithUpdater) {
  auto current_time = start_time_;

  // Create metric with updater, simulate some load
  LoadMetric metric;
  LoadMetricUpdater updater(&metric);
  updater.LoadStart(current_time);
  current_time += std::chrono::milliseconds(150);

  // Flush before recording
  updater.Flush(current_time);

  // Record the metric (this should use MergeFrom + Reset)
  aggregator_->RecordMetrics(std::move(metric));

  // Get live metrics
  auto [live_metrics, age] = aggregator_->GetAggregateEndingNow(
      TestAggregator::Duration::zero(), current_time);
  EXPECT_NEAR(live_metrics.LoadSeconds(), 0.15, 1e-6);
}

TEST_F(LoadMetricIntegrationTest, MultipleRecordMetrics) {
  auto current_time = start_time_;

  // First metric
  LoadMetric metric1;
  LoadMetricUpdater updater1(&metric1);
  updater1.LoadStart(current_time);
  current_time += std::chrono::milliseconds(100);
  updater1.Flush(current_time);
  aggregator_->RecordMetrics(std::move(metric1));

  // Second metric
  LoadMetric metric2;
  LoadMetricUpdater updater2(&metric2);
  updater2.LoadStart(current_time);
  current_time += std::chrono::milliseconds(75);
  updater2.Flush(current_time);
  aggregator_->RecordMetrics(std::move(metric2));

  // Get live metrics
  auto [live_metrics, age] = aggregator_->GetAggregateEndingNow(
      TestAggregator::Duration::zero(), current_time);
  EXPECT_NEAR(live_metrics.LoadSeconds(), 0.175, 1e-6);  // 0.1 + 0.075
}

TEST_F(LoadMetricIntegrationTest, AdvanceTest) {
  auto current_time = start_time_;

  // Add some metrics
  LoadMetric metric;
  LoadMetricUpdater updater(&metric);
  updater.LoadStart(current_time);
  updater.LoadStop(current_time + std::chrono::milliseconds(100));
  aggregator_->RecordMetrics(std::move(metric));

  // Advance to move live metrics to buckets
  auto tick_time = start_time_ + aggregator_->GetResolution();
  auto period = aggregator_->Advance(tick_time);

  // Should return the base time period
  EXPECT_EQ(period, TimePeriod::k16Milliseconds);

  // Live metrics should be empty after advance
  auto [live_metrics, age] = aggregator_->GetAggregateEndingNow(
      TestAggregator::Duration::zero(), tick_time);
  EXPECT_EQ(live_metrics.LoadSeconds(), 0.0);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}