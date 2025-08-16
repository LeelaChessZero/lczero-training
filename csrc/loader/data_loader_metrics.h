// ABOUTME: Header for UpdateFrom functions for data loader metric protobuf
// messages. ABOUTME: Declares functions for aggregating FilePathProvider and
// DataLoader metrics.

#pragma once

#include "proto/training_metrics.pb.h"

namespace lczero {
namespace training {

void UpdateFrom(LoadMetricProto& dest, const LoadMetricProto& src);
void UpdateFrom(FilePathProviderMetricsProto& dest,
                const FilePathProviderMetricsProto& src);
void UpdateFrom(DataLoaderMetricsProto& dest,
                const DataLoaderMetricsProto& src);

}  // namespace training
}  // namespace lczero