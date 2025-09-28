// ABOUTME: Header for UpdateFrom functions for data loader metric protobuf
// messages. ABOUTME: Declares functions for aggregating FilePathProvider and
// DataLoader metrics.

#pragma once

#include "absl/strings/string_view.h"
#include "proto/training_metrics.pb.h"
#include "utils/metrics/load_metric.h"
#include "utils/metrics/statistics_metric.h"
#include "utils/queue.h"

namespace lczero {
namespace training {

void UpdateFrom(QueueMetricProto& dest, const QueueMetricProto& src);
void UpdateFrom(FilePathProviderMetricsProto& dest,
                const FilePathProviderMetricsProto& src);
void UpdateFrom(ChunkSourceLoaderMetricsProto& dest,
                const ChunkSourceLoaderMetricsProto& src);
void UpdateFrom(ShufflingChunkPoolMetricsProto& dest,
                const ShufflingChunkPoolMetricsProto& src);
void UpdateFrom(ChunkRescorerMetricsProto& dest,
                const ChunkRescorerMetricsProto& src);
void UpdateFrom(ChunkUnpackerMetricsProto& dest,
                const ChunkUnpackerMetricsProto& src);
void UpdateFrom(ShufflingFrameSamplerMetricsProto& dest,
                const ShufflingFrameSamplerMetricsProto& src);
void UpdateFrom(TensorGeneratorMetricsProto& dest,
                const TensorGeneratorMetricsProto& src);
void UpdateFrom(StageMetricProto& dest, const StageMetricProto& src);
void UpdateFrom(DataLoaderMetricsProto& dest,
                const DataLoaderMetricsProto& src);

template <typename T>
QueueMetricProto MetricsFromQueue(absl::string_view name, Queue<T>& queue) {
  QueueMetricProto result;
  result.set_name(std::string(name));
  result.set_put_count(queue.GetTotalPutCount(true));
  result.set_get_count(queue.GetTotalGetCount(true));
  result.set_drop_count(queue.GetTotalDropCount(true));
  AddSample(*result.mutable_queue_fullness(), queue.Size());
  result.set_queue_capacity(queue.Capacity());
  return result;
}

}  // namespace training
}  // namespace lczero
