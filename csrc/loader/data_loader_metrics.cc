// ABOUTME: Implementation of UpdateFrom functions for data loader metric
// protobuf messages. ABOUTME: Handles aggregation of generic stage metrics.

#include "loader/data_loader_metrics.h"

#include <algorithm>

#include "absl/strings/string_view.h"
#include "utils/metrics/statistics_metric.h"

namespace lczero {
namespace training {
namespace {

template <typename ProtoT>
ProtoT* FindByName(std::vector<ProtoT>* entries, absl::string_view name) {
  for (auto& entry : *entries) {
    if (entry.name() == name) return &entry;
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

void UpdateFrom(CountMetricProto& dest, const CountMetricProto& src) {
  if (src.has_name()) dest.set_name(src.name());
  if (src.has_count()) dest.set_count(dest.count() + src.count());
  if (src.has_capacity()) dest.set_capacity(src.capacity());
}

void UpdateFrom(StageMetricProto& dest, const StageMetricProto& src) {
  if (src.has_name()) dest.set_name(src.name());

  for (const auto& load_metrics : src.load_metrics()) {
    LoadMetricProto* dest_load =
        load_metrics.has_name()
            ? FindByName(dest.mutable_load_metrics(), load_metrics.name())
            : nullptr;
    if (dest_load == nullptr) {
      dest_load = dest.add_load_metrics();
    }
    UpdateFrom(*dest_load, load_metrics);
  }

  for (const auto& queue_metrics : src.queue_metrics()) {
    QueueMetricProto* dest_queue =
        queue_metrics.has_name()
            ? FindByName(dest.mutable_queue_metrics(), queue_metrics.name())
            : nullptr;
    if (dest_queue == nullptr) {
      dest_queue = dest.add_queue_metrics();
    }
    UpdateFrom(*dest_queue, queue_metrics);
  }

  for (const auto& count_metrics : src.count_metrics()) {
    CountMetricProto* dest_count =
        count_metrics.has_name()
            ? FindByName(dest.mutable_count_metrics(), count_metrics.name())
            : nullptr;
    if (dest_count == nullptr) {
      dest_count = dest.add_count_metrics();
    }
    UpdateFrom(*dest_count, count_metrics);
  }

  if (src.has_dropped()) {
    dest.set_dropped(dest.dropped() + src.dropped());
  }
  if (src.has_skipped_files_count()) {
    dest.set_skipped_files_count(dest.skipped_files_count() +
                                 src.skipped_files_count());
  }
  if (src.has_last_chunk_key()) dest.set_last_chunk_key(src.last_chunk_key());
  if (src.has_anchor()) dest.set_anchor(src.anchor());
  if (src.has_chunks_since_anchor()) {
    dest.set_chunks_since_anchor(src.chunks_since_anchor());
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
