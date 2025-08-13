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
}

void UpdateFrom(FilePathProviderMetricsProto& dest,
                const FilePathProviderMetricsProto& src) {
  dest.set_total_files_discovered(dest.total_files_discovered() +
                                  src.total_files_discovered());
  UpdateFrom(*dest.mutable_load(), src.load());
  UpdateFrom(*dest.mutable_queue_size(), src.queue_size());
}

void UpdateFrom(DataLoaderMetricsProto& dest,
                const DataLoaderMetricsProto& src) {
  UpdateFrom(*dest.mutable_file_path_provider(), src.file_path_provider());
}

}  // namespace training
}  // namespace lczero