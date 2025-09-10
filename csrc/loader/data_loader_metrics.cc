// ABOUTME: Implementation of UpdateFrom functions for data loader metric
// protobuf messages. ABOUTME: Handles aggregation of FilePathProvider metrics
// and top-level DataLoader metrics.

#include "loader/data_loader_metrics.h"

#include <algorithm>

#include "utils/metrics/statistics_metric.h"

namespace lczero {
namespace training {

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

void UpdateFrom(ChunkSourceLoaderMetricsProto& dest,
                const ChunkSourceLoaderMetricsProto& src) {
  UpdateFrom(*dest.mutable_load(), src.load());
  UpdateFrom(*dest.mutable_queue(), src.queue());
  if (src.has_last_chunk_key()) dest.set_last_chunk_key(src.last_chunk_key());
}

void UpdateFrom(ShufflingChunkPoolMetricsProto& dest,
                const ShufflingChunkPoolMetricsProto& src) {
  UpdateFrom(*dest.mutable_indexing_load(), src.indexing_load());
  UpdateFrom(*dest.mutable_chunk_loading_load(), src.chunk_loading_load());
  UpdateFrom(*dest.mutable_queue(), src.queue());
  UpdateFrom(*dest.mutable_chunk_sources_count(), src.chunk_sources_count());
  if (src.has_current_chunks()) dest.set_current_chunks(src.current_chunks());
  if (src.has_pool_capacity()) dest.set_pool_capacity(src.pool_capacity());
}

void UpdateFrom(ChunkUnpackerMetricsProto& dest,
                const ChunkUnpackerMetricsProto& src) {
  UpdateFrom(*dest.mutable_load(), src.load());
  UpdateFrom(*dest.mutable_queue(), src.queue());
  UpdateFrom(*dest.mutable_bad_chunks_count(), src.bad_chunks_count());
}

void UpdateFrom(ShufflingFrameSamplerMetricsProto& dest,
                const ShufflingFrameSamplerMetricsProto& src) {
  UpdateFrom(*dest.mutable_load(), src.load());
  UpdateFrom(*dest.mutable_queue(), src.queue());
  if (src.has_reservoir_capacity()) {
    dest.set_reservoir_capacity(src.reservoir_capacity());
  }
  if (src.has_current_reservoir_size()) {
    dest.set_current_reservoir_size(src.current_reservoir_size());
  }
}

void UpdateFrom(TensorGeneratorMetricsProto& dest,
                const TensorGeneratorMetricsProto& src) {
  UpdateFrom(*dest.mutable_load(), src.load());
  UpdateFrom(*dest.mutable_queue(), src.queue());
}

void UpdateFrom(DataLoaderMetricsProto& dest,
                const DataLoaderMetricsProto& src) {
  UpdateFrom(*dest.mutable_file_path_provider(), src.file_path_provider());
  UpdateFrom(*dest.mutable_chunk_source_loader(), src.chunk_source_loader());
  UpdateFrom(*dest.mutable_shuffling_chunk_pool(), src.shuffling_chunk_pool());
  UpdateFrom(*dest.mutable_chunk_unpacker(), src.chunk_unpacker());
  UpdateFrom(*dest.mutable_shuffling_frame_sampler(),
             src.shuffling_frame_sampler());
  UpdateFrom(*dest.mutable_tensor_generator(), src.tensor_generator());
}

}  // namespace training
}  // namespace lczero