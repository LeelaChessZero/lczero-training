#pragma once

#include <algorithm>

#include "proto/training_metrics.pb.h"

namespace lczero {

// Helper function to add a sample to StatisticsProtoInt64
inline void AddSample(StatisticsProtoInt64& stats, int64_t value) {
  stats.set_min(std::min(stats.min(), value));
  stats.set_max(std::max(stats.max(), value));
  stats.set_sum(stats.sum() + value);
  stats.set_count(stats.count() + 1);
  stats.set_latest(value);
}

// Helper function to add a sample to StatisticsProtoDouble
inline void AddSample(StatisticsProtoDouble& stats, double value) {
  stats.set_min(std::min(stats.min(), value));
  stats.set_max(std::max(stats.max(), value));
  stats.set_sum(stats.sum() + value);
  stats.set_count(stats.count() + 1);
  stats.set_latest(value);
}

// UpdateFrom function for StatisticsProtoInt64 - merges statistics
inline void UpdateFrom(StatisticsProtoInt64& dest,
                       const StatisticsProtoInt64& src) {
  if (src.count() == 0) return;  // Nothing to merge from empty source

  dest.set_min(std::min(dest.min(), src.min()));
  dest.set_max(std::max(dest.max(), src.max()));
  dest.set_sum(dest.sum() + src.sum());
  dest.set_count(dest.count() + src.count());
  dest.set_latest(src.latest());  // Source is newer, use its latest value
}

// UpdateFrom function for StatisticsProtoDouble - merges statistics
inline void UpdateFrom(StatisticsProtoDouble& dest,
                       const StatisticsProtoDouble& src) {
  if (src.count() == 0) return;  // Nothing to merge from empty source

  if (src.has_name()) dest.set_name(src.name());
  dest.set_min(std::min(dest.min(), src.min()));
  dest.set_max(std::max(dest.max(), src.max()));
  dest.set_sum(dest.sum() + src.sum());
  dest.set_count(dest.count() + src.count());
  dest.set_latest(src.latest());  // Source is newer, use its latest value
}

}  // namespace lczero