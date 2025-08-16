#include "utils/metrics/load_metric.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>

#include "proto/training_metrics.pb.h"
#include "utils/metrics/exponential_aggregator.h"

namespace lczero {
namespace training {

class LoadMetricTest : public ::testing::Test {
 protected:
  using Clock = LoadMetricUpdater::Clock;

  void SetUp() override { start_time_ = Clock::now(); }

  Clock::time_point start_time_;
};

TEST_F(LoadMetricTest, BasicLoadMetricProto) {
  LoadMetricProto metric;
  EXPECT_EQ(metric.load_seconds(), 0.0);
  EXPECT_EQ(metric.total_seconds(), 0.0);

  // Test UpdateFrom with LoadMetricUpdater
  LoadMetricUpdater other_updater(start_time_);
  other_updater.LoadStart(start_time_);
  other_updater.LoadStop(start_time_ + std::chrono::milliseconds(500));
  LoadMetricProto other =
      other_updater.FlushMetrics(start_time_ + std::chrono::milliseconds(500));
  UpdateFrom(metric, other);
  EXPECT_NEAR(metric.load_seconds(), 0.5, 1e-6);
  EXPECT_NEAR(metric.total_seconds(), 0.5, 1e-6);

  // Test Clear (used to be Reset)
  metric.Clear();
  EXPECT_EQ(metric.load_seconds(), 0.0);
  EXPECT_EQ(metric.total_seconds(), 0.0);
}

TEST_F(LoadMetricTest, LoadMetricUpdaterBasic) {
  LoadMetricUpdater updater(start_time_);
  auto now = start_time_;

  // Start load and verify initial state
  updater.LoadStart(now);
  LoadMetricProto metric = updater.FlushMetrics(now);
  EXPECT_EQ(metric.load_seconds(), 0.0);
  EXPECT_EQ(metric.total_seconds(), 0.0);

  // Advance time and stop load
  now += std::chrono::milliseconds(100);
  updater.LoadStop(now);
  metric = updater.FlushMetrics(now);
  EXPECT_NEAR(metric.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(metric.total_seconds(), 0.1, 1e-6);

  // Wait idle time, then start again
  now += std::chrono::milliseconds(50);
  updater.LoadStart(now);
  metric = updater.FlushMetrics(now);
  EXPECT_NEAR(metric.load_seconds(), 0.0, 1e-6);    // Reset after flush
  EXPECT_NEAR(metric.total_seconds(), 0.05, 1e-6);  // Only idle time

  now += std::chrono::milliseconds(200);
  updater.LoadStop(now);
  metric = updater.FlushMetrics(now);
  EXPECT_NEAR(metric.load_seconds(), 0.2, 1e-6);   // 0.2 load
  EXPECT_NEAR(metric.total_seconds(), 0.2, 1e-6);  // 0.2 total
}

TEST_F(LoadMetricTest, LoadMetricUpdaterFlush) {
  LoadMetricUpdater updater(start_time_);
  auto now = start_time_;

  // Start load
  updater.LoadStart(now);
  now += std::chrono::milliseconds(100);

  // Flush should update the internal metric
  updater.Flush(now);
  LoadMetricProto metric = updater.FlushMetrics(now);
  EXPECT_NEAR(metric.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(metric.total_seconds(), 0.1, 1e-6);

  // Continue loading
  now += std::chrono::milliseconds(50);
  updater.LoadStop(now);
  metric = updater.FlushMetrics(now);
  EXPECT_NEAR(metric.load_seconds(), 0.05, 1e-6);   // Only new load time
  EXPECT_NEAR(metric.total_seconds(), 0.05, 1e-6);  // Only new total time
}

TEST_F(LoadMetricTest, LoadMetricProtoMerging) {
  LoadMetricUpdater updater1(start_time_);
  LoadMetricUpdater updater2(start_time_);
  auto now = start_time_;

  // Create load in updater1
  updater1.LoadStart(now);
  updater1.LoadStop(now + std::chrono::milliseconds(100));
  LoadMetricProto metric1 =
      updater1.FlushMetrics(now + std::chrono::milliseconds(100));
  EXPECT_NEAR(metric1.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(metric1.total_seconds(), 0.1, 1e-6);

  // Create load in updater2
  updater2.LoadStart(now);
  updater2.LoadStop(now + std::chrono::milliseconds(100));
  LoadMetricProto metric2 =
      updater2.FlushMetrics(now + std::chrono::milliseconds(100));
  EXPECT_NEAR(metric2.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(metric2.total_seconds(), 0.1, 1e-6);

  // Merge
  UpdateFrom(metric1, metric2);
  EXPECT_NEAR(metric1.load_seconds(), 0.2, 1e-6);
  EXPECT_NEAR(metric1.total_seconds(), 0.2, 1e-6);
  EXPECT_NEAR(metric2.load_seconds(), 0.1, 1e-6);   // Source unchanged
  EXPECT_NEAR(metric2.total_seconds(), 0.1, 1e-6);  // Source unchanged
}

TEST_F(LoadMetricTest, LoadMetricProtoMoveSemantics) {
  // Test that LoadMetricProto move semantics work correctly
  LoadMetricUpdater source_updater(start_time_);
  source_updater.LoadStart(start_time_);
  source_updater.LoadStop(start_time_ + std::chrono::milliseconds(100));
  LoadMetricProto source =
      source_updater.FlushMetrics(start_time_ + std::chrono::milliseconds(100));
  EXPECT_NEAR(source.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(source.total_seconds(), 0.1, 1e-6);

  // Test move construction
  LoadMetricProto moved_constructed(std::move(source));
  EXPECT_NEAR(moved_constructed.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(moved_constructed.total_seconds(), 0.1, 1e-6);

  // Test move assignment
  LoadMetricProto move_assigned;
  LoadMetricUpdater another_updater(start_time_);
  another_updater.LoadStart(start_time_);
  another_updater.LoadStop(start_time_ + std::chrono::milliseconds(50));
  LoadMetricProto another_source =
      another_updater.FlushMetrics(start_time_ + std::chrono::milliseconds(50));
  EXPECT_NEAR(another_source.load_seconds(), 0.05, 1e-6);
  EXPECT_NEAR(another_source.total_seconds(), 0.05, 1e-6);

  move_assigned = std::move(another_source);
  EXPECT_NEAR(move_assigned.load_seconds(), 0.05, 1e-6);
  EXPECT_NEAR(move_assigned.total_seconds(), 0.05, 1e-6);

  // Test UpdateFrom
  LoadMetricProto dest;
  UpdateFrom(dest, moved_constructed);
  EXPECT_NEAR(dest.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(dest.total_seconds(), 0.1, 1e-6);

  UpdateFrom(dest, move_assigned);
  EXPECT_NEAR(dest.load_seconds(), 0.15, 1e-6);
  EXPECT_NEAR(dest.total_seconds(), 0.15, 1e-6);
}

TEST_F(LoadMetricTest, LoadUtilizationTracking) {
  LoadMetricUpdater updater(start_time_);
  auto now = start_time_;

  // Start with some idle time before any load
  now += std::chrono::milliseconds(100);
  updater.LoadStart(now);
  LoadMetricProto metric = updater.FlushMetrics(now);
  EXPECT_NEAR(metric.load_seconds(), 0.0, 1e-6);
  EXPECT_NEAR(metric.total_seconds(), 0.1, 1e-6);  // 100ms idle

  // Add some load time
  now += std::chrono::milliseconds(200);
  updater.LoadStop(now);
  metric = updater.FlushMetrics(now);
  EXPECT_NEAR(metric.load_seconds(), 0.2, 1e-6);  // 200ms load
  EXPECT_NEAR(metric.total_seconds(), 0.2,
              1e-6);  // 200ms total (after flush reset)

  // Add more idle time
  now += std::chrono::milliseconds(100);
  updater.Flush(now);
  metric = updater.FlushMetrics(now);
  EXPECT_NEAR(metric.load_seconds(), 0.0, 1e-6);   // No load time
  EXPECT_NEAR(metric.total_seconds(), 0.1, 1e-6);  // 100ms idle

  // Test complete utilization tracking with one updater
  LoadMetricUpdater total_updater(start_time_);
  auto total_now = start_time_;

  // 100ms idle
  total_now += std::chrono::milliseconds(100);
  total_updater.LoadStart(total_now);

  // 200ms load
  total_now += std::chrono::milliseconds(200);
  total_updater.LoadStop(total_now);

  // 100ms idle
  total_now += std::chrono::milliseconds(100);
  LoadMetricProto total_metric = total_updater.FlushMetrics(total_now);

  // Calculate utilization
  double utilization =
      total_metric.load_seconds() / total_metric.total_seconds();
  EXPECT_NEAR(utilization, 0.5,
              1e-6);  // 50% utilization (200ms load / 400ms total)
}

class LoadMetricProtoIntegrationTest : public ::testing::Test {
 protected:
  using TestAggregator =
      ExponentialAggregator<LoadMetricProto, TimePeriod::k16Milliseconds>;
  using Clock = TestAggregator::Clock;

  void SetUp() override {
    aggregator_ = std::make_unique<TestAggregator>(
        [](LoadMetricProto& m) { m.Clear(); },
        [](LoadMetricProto& dest, const LoadMetricProto& src) {
          UpdateFrom(dest, src);
        });
    start_time_ = Clock::now();
  }

  std::unique_ptr<TestAggregator> aggregator_;
  Clock::time_point start_time_;
};

TEST_F(LoadMetricProtoIntegrationTest, RecordMetricsWithUpdater) {
  auto current_time = start_time_;

  // Create metric with updater, simulate some load
  LoadMetricUpdater updater(current_time);
  updater.LoadStart(current_time);
  current_time += std::chrono::milliseconds(150);

  // Flush and get metric
  LoadMetricProto metric = updater.FlushMetrics(current_time);

  // Record the metric (this should use UpdateFrom + Reset)
  aggregator_->RecordMetrics(std::move(metric));

  // Get live metrics
  auto [live_metrics, age] = aggregator_->GetAggregateEndingNow(
      TestAggregator::Duration::zero(), current_time);
  EXPECT_NEAR(live_metrics.load_seconds(), 0.15, 1e-6);
}

TEST_F(LoadMetricProtoIntegrationTest, MultipleRecordMetrics) {
  auto current_time = start_time_;

  // First metric
  LoadMetricUpdater updater1(current_time);
  updater1.LoadStart(current_time);
  current_time += std::chrono::milliseconds(100);
  LoadMetricProto metric1 = updater1.FlushMetrics(current_time);
  aggregator_->RecordMetrics(std::move(metric1));

  // Second metric
  LoadMetricUpdater updater2(current_time);
  updater2.LoadStart(current_time);
  current_time += std::chrono::milliseconds(75);
  LoadMetricProto metric2 = updater2.FlushMetrics(current_time);
  aggregator_->RecordMetrics(std::move(metric2));

  // Get live metrics
  auto [live_metrics, age] = aggregator_->GetAggregateEndingNow(
      TestAggregator::Duration::zero(), current_time);
  EXPECT_NEAR(live_metrics.load_seconds(), 0.175, 1e-6);  // 0.1 + 0.075
}

TEST_F(LoadMetricProtoIntegrationTest, AdvanceTest) {
  auto current_time = start_time_;

  // Add some metrics
  LoadMetricUpdater updater(current_time);
  updater.LoadStart(current_time);
  updater.LoadStop(current_time + std::chrono::milliseconds(100));
  LoadMetricProto metric =
      updater.FlushMetrics(current_time + std::chrono::milliseconds(100));
  aggregator_->RecordMetrics(std::move(metric));

  // Advance to move live metrics to buckets
  auto tick_time = start_time_ + aggregator_->GetResolution();
  auto period = aggregator_->Advance(tick_time);

  // Should return the base time period
  EXPECT_EQ(period, TimePeriod::k16Milliseconds);

  // Live metrics should be empty after advance
  auto [live_metrics, age] = aggregator_->GetAggregateEndingNow(
      TestAggregator::Duration::zero(), tick_time);
  EXPECT_EQ(live_metrics.load_seconds(), 0.0);
}

}  // namespace training
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}