// ABOUTME: Implementation of UpdateFrom functions for data loader metric
// protobuf messages. ABOUTME: Handles aggregation of FilePathProvider metrics
// and top-level DataLoader metrics.

#include "loader/data_loader_metrics.h"

#include <algorithm>
#include <vector>

#include "absl/strings/string_view.h"
#include "utils/metrics/statistics_metric.h"

namespace lczero {
namespace training {
namespace {

template <typename ProtoT>
ProtoT* FindByName(std::vector<ProtoT>* entries, absl::string_view name) {
  if (entries == nullptr || name.empty()) {
    return nullptr;
  }
  for (auto& entry : *entries) {
    if (entry.has_name() && entry.name() == name) {
      return &entry;
    }
  }
  return nullptr;
}

}  // namespace

void UpdateFrom(QueueMetricProto& dest, const QueueMetricProto& src) {
  if (src.has_name()) dest.set_name(src.name());
  dest.set_put_count(dest.put_count() + src.put_count());
  dest.set_get_count(dest.get_count() + src.get_count());
  dest.set_drop_count(dest.drop_count() + src.drop_count());
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
  UpdateFrom(*dest.mutable_dropped_chunks(), src.dropped_chunks());
  if (src.has_current_chunks()) dest.set_current_chunks(src.current_chunks());
  if (src.has_pool_capacity()) dest.set_pool_capacity(src.pool_capacity());
  if (src.has_chunks_since_anchor())
    dest.set_chunks_since_anchor(src.chunks_since_anchor());
  if (src.has_anchor()) dest.set_anchor(src.anchor());
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

void UpdateFrom(StageMetricProto& dest, const StageMetricProto& src) {
  if (src.has_name()) dest.set_name(src.name());
  if (src.has_file_path_provider()) {
    UpdateFrom(*dest.mutable_file_path_provider(), src.file_path_provider());
  }
  if (src.has_chunk_source_loader()) {
    UpdateFrom(*dest.mutable_chunk_source_loader(), src.chunk_source_loader());
  }
  if (src.has_shuffling_chunk_pool()) {
    UpdateFrom(*dest.mutable_shuffling_chunk_pool(),
               src.shuffling_chunk_pool());
  }
  if (src.has_chunk_unpacker()) {
    UpdateFrom(*dest.mutable_chunk_unpacker(), src.chunk_unpacker());
  }
  if (src.has_shuffling_frame_sampler()) {
    UpdateFrom(*dest.mutable_shuffling_frame_sampler(),
               src.shuffling_frame_sampler());
  }
  if (src.has_tensor_generator()) {
    UpdateFrom(*dest.mutable_tensor_generator(), src.tensor_generator());
  }

  for (const auto& queue_metrics : src.output_queue_metrics()) {
    QueueMetricProto* dest_queue =
        queue_metrics.has_name()
            ? FindByName(dest.mutable_output_queue_metrics(),
                         queue_metrics.name())
            : nullptr;
    if (dest_queue == nullptr) {
      dest_queue = dest.add_output_queue_metrics();
    }
    UpdateFrom(*dest_queue, queue_metrics);
  }
}

void UpdateFrom(DataLoaderMetricsProto& dest,
                const DataLoaderMetricsProto& src) {
  for (const auto& stage_metrics : src.stage_metrics()) {
    StageMetricProto* dest_stage =
        stage_metrics.has_name()
            ? FindByName(dest.mutable_stage_metrics(), stage_metrics.name())
            : nullptr;
    if (dest_stage == nullptr) {
      dest_stage = dest.add_stage_metrics();
    }

    UpdateFrom(*dest_stage, stage_metrics);
  }
}

}  // namespace training
}  // namespace lczero
