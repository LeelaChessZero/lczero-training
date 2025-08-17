// ABOUTME: Implementation of UpdateFrom functions for data loader metric
// protobuf messages. ABOUTME: Handles aggregation of FilePathProvider metrics
// and top-level DataLoader metrics.

#include "loader/data_loader_metrics.h"

#include <algorithm>

#include "utils/metrics/statistics_metric.h"

namespace lczero {
namespace training {

void UpdateFrom(LoadMetricProto& dest, const LoadMetricProto& src) {
  dest.set_load_seconds(dest.load_seconds() + src.load_seconds());
  dest.set_total_seconds(dest.total_seconds() + src.total_seconds());
}

void UpdateFrom(QueueMetricProto& dest, const QueueMetricProto& src) {
  dest.set_message_count(dest.message_count() + src.message_count());
  UpdateFrom(*dest.mutable_queue_fullness(), src.queue_fullness());
  if (src.has_queue_capacity()) dest.set_queue_capacity(src.queue_capacity());
}

void UpdateFrom(FilePathProviderMetricsProto& dest,
                const FilePathProviderMetricsProto& src) {
  UpdateFrom(*dest.mutable_load(), src.load());
  UpdateFrom(*dest.mutable_queue(), src.queue());
}

void UpdateFrom(DataLoaderMetricsProto& dest,
                const DataLoaderMetricsProto& src) {
  UpdateFrom(*dest.mutable_file_path_provider(), src.file_path_provider());
}

}  // namespace training
}  // namespace lczero