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

  // Test UpdateFrom (used to be UpdateFrom)
  LoadMetricProto other;
  LoadMetricUpdater other_updater(&other, start_time_);
  other_updater.LoadStart(start_time_);
  other_updater.LoadStop(start_time_ + std::chrono::milliseconds(500));
  UpdateFrom(metric, other);
  EXPECT_NEAR(metric.load_seconds(), 0.5, 1e-6);
  EXPECT_NEAR(metric.total_seconds(), 0.5, 1e-6);

  // Test Clear (used to be Reset)
  metric.Clear();
  EXPECT_EQ(metric.load_seconds(), 0.0);
  EXPECT_EQ(metric.total_seconds(), 0.0);
}

TEST_F(LoadMetricTest, LoadMetricUpdaterBasic) {
  LoadMetricProto metric;
  LoadMetricUpdater updater(&metric, start_time_);
  auto now = start_time_;

  // Start load and verify initial state
  updater.LoadStart(now);
  EXPECT_EQ(metric.load_seconds(), 0.0);
  EXPECT_EQ(metric.total_seconds(), 0.0);

  // Advance time and stop load
  now += std::chrono::milliseconds(100);
  updater.LoadStop(now);
  EXPECT_NEAR(metric.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(metric.total_seconds(), 0.1, 1e-6);

  // Wait idle time, then start again
  now += std::chrono::milliseconds(50);
  updater.LoadStart(now);
  EXPECT_NEAR(metric.load_seconds(), 0.1, 1e-6);    // No change in load
  EXPECT_NEAR(metric.total_seconds(), 0.15, 1e-6);  // 0.1 + 0.05 idle

  now += std::chrono::milliseconds(200);
  updater.LoadStop(now);
  EXPECT_NEAR(metric.load_seconds(), 0.3, 1e-6);    // 0.1 + 0.2
  EXPECT_NEAR(metric.total_seconds(), 0.35, 1e-6);  // 0.15 + 0.2
}

TEST_F(LoadMetricTest, LoadMetricUpdaterFlush) {
  LoadMetricProto metric;
  LoadMetricUpdater updater(&metric, start_time_);
  auto now = start_time_;

  // Start load
  updater.LoadStart(now);
  now += std::chrono::milliseconds(100);

  // Flush should update the metric
  updater.Flush(now);
  EXPECT_NEAR(metric.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(metric.total_seconds(), 0.1, 1e-6);

  // Continue loading
  now += std::chrono::milliseconds(50);
  updater.LoadStop(now);
  EXPECT_NEAR(metric.load_seconds(), 0.15, 1e-6);   // 0.1 + 0.05
  EXPECT_NEAR(metric.total_seconds(), 0.15, 1e-6);  // 0.1 + 0.05
}

TEST_F(LoadMetricTest, LoadMetricProtoMerging) {
  LoadMetricProto metric1, metric2;
  LoadMetricUpdater updater1(&metric1, start_time_),
      updater2(&metric2, start_time_);
  auto now = start_time_;

  // Create load in metric1
  updater1.LoadStart(now);
  updater1.LoadStop(now + std::chrono::milliseconds(100));
  EXPECT_NEAR(metric1.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(metric1.total_seconds(), 0.1, 1e-6);

  // Create load in metric2 (initialize updater at the start time to avoid idle)
  LoadMetricUpdater updater2_correct(&metric2,
                                     now + std::chrono::milliseconds(50));
  updater2_correct.LoadStart(now + std::chrono::milliseconds(50));
  updater2_correct.LoadStop(now + std::chrono::milliseconds(150));
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
  LoadMetricProto source;
  LoadMetricUpdater source_updater(&source, start_time_);
  source_updater.LoadStart(start_time_);
  source_updater.LoadStop(start_time_ + std::chrono::milliseconds(100));
  EXPECT_NEAR(source.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(source.total_seconds(), 0.1, 1e-6);

  // Test move construction
  LoadMetricProto moved_constructed(std::move(source));
  EXPECT_NEAR(moved_constructed.load_seconds(), 0.1, 1e-6);
  EXPECT_NEAR(moved_constructed.total_seconds(), 0.1, 1e-6);

  // Test move assignment
  LoadMetricProto move_assigned;
  LoadMetricProto another_source;
  LoadMetricUpdater another_updater(&another_source, start_time_);
  another_updater.LoadStart(start_time_);
  another_updater.LoadStop(start_time_ + std::chrono::milliseconds(50));
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
  LoadMetricProto metric;
  LoadMetricUpdater updater(&metric, start_time_);
  auto now = start_time_;

  // Start with some idle time before any load
  now += std::chrono::milliseconds(100);
  updater.LoadStart(now);
  EXPECT_NEAR(metric.load_seconds(), 0.0, 1e-6);
  EXPECT_NEAR(metric.total_seconds(), 0.1, 1e-6);  // 100ms idle

  // Add some load time
  now += std::chrono::milliseconds(200);
  updater.LoadStop(now);
  EXPECT_NEAR(metric.load_seconds(), 0.2, 1e-6);   // 200ms load
  EXPECT_NEAR(metric.total_seconds(), 0.3, 1e-6);  // 100ms idle + 200ms load

  // Add more idle time
  now += std::chrono::milliseconds(100);
  updater.Flush(now);
  EXPECT_NEAR(metric.load_seconds(), 0.2, 1e-6);   // No change in load
  EXPECT_NEAR(metric.total_seconds(), 0.4, 1e-6);  // 100ms additional idle

  // Calculate utilization
  double utilization = metric.load_seconds() / metric.total_seconds();
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
  LoadMetricProto metric;
  LoadMetricUpdater updater(&metric, current_time);
  updater.LoadStart(current_time);
  current_time += std::chrono::milliseconds(150);

  // Flush before recording
  updater.Flush(current_time);

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
  LoadMetricProto metric1;
  LoadMetricUpdater updater1(&metric1, current_time);
  updater1.LoadStart(current_time);
  current_time += std::chrono::milliseconds(100);
  updater1.Flush(current_time);
  aggregator_->RecordMetrics(std::move(metric1));

  // Second metric
  LoadMetricProto metric2;
  LoadMetricUpdater updater2(&metric2, current_time);
  updater2.LoadStart(current_time);
  current_time += std::chrono::milliseconds(75);
  updater2.Flush(current_time);
  aggregator_->RecordMetrics(std::move(metric2));

  // Get live metrics
  auto [live_metrics, age] = aggregator_->GetAggregateEndingNow(
      TestAggregator::Duration::zero(), current_time);
  EXPECT_NEAR(live_metrics.load_seconds(), 0.175, 1e-6);  // 0.1 + 0.075
}

TEST_F(LoadMetricProtoIntegrationTest, AdvanceTest) {
  auto current_time = start_time_;

  // Add some metrics
  LoadMetricProto metric;
  LoadMetricUpdater updater(&metric, current_time);
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
  EXPECT_EQ(live_metrics.load_seconds(), 0.0);
}

}  // namespace training
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}