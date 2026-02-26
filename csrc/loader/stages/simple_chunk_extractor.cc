#include "loader/stages/simple_chunk_extractor.h"

#include <absl/algorithm/container.h>
#include <absl/log/log.h>

#include <numeric>

#include "loader/data_loader_metrics.h"

namespace lczero {
namespace training {

SimpleChunkExtractor::SimpleChunkExtractor(
    const SimpleChunkExtractorConfig& config)
    : SingleInputStage<SimpleChunkExtractorConfig, ChunkSourceWithPhase>(
          config),
      SingleOutputStage<TrainingChunk>(config.output()),
      bitgen_(absl::MakeSeedSeq()) {}

SimpleChunkExtractor::~SimpleChunkExtractor() { Stop(); }

void SimpleChunkExtractor::Start() {
  thread_pool_.Enqueue(
      [this](std::stop_token stop_token) { Worker(stop_token); });
}

void SimpleChunkExtractor::Stop() {
  if (thread_pool_.stop_token().stop_requested()) return;
  LOG(INFO) << "Stopping SimpleChunkExtractor.";
  thread_pool_.Shutdown();
  output_queue()->Close();
}

void SimpleChunkExtractor::Worker(std::stop_token stop_token) {
  auto producer = output_queue()->CreateProducer();

  try {
    while (true) {
      auto item = input_queue()->Get(stop_token);
      if (item.message_type != FilePathProvider::MessageType::kFile ||
          !item.source) {
        continue;
      }

      ProcessSource(producer, std::move(item.source), stop_token);
    }
  } catch (const QueueClosedException&) {
  }
}

void SimpleChunkExtractor::ProcessSource(
    Queue<TrainingChunk>::Producer& producer,
    std::unique_ptr<ChunkSource> source, std::stop_token stop_token) {
  const size_t chunk_count = source->GetChunkCount();
  if (chunk_count == 0) return;

  std::vector<size_t> indices(chunk_count);
  std::iota(indices.begin(), indices.end(), 0);
  absl::c_shuffle(indices, bitgen_);

  const std::string sort_key = source->GetChunkSortKey();
  for (size_t idx : indices) {
    if (auto chunk = LoadChunk(*source, sort_key, idx)) {
      producer.Put(std::move(*chunk), stop_token);
      ++chunks_processed_;
    }
  }
  ++sources_processed_;
}

std::optional<TrainingChunk> SimpleChunkExtractor::LoadChunk(
    ChunkSource& source, const std::string& sort_key, size_t index) {
  auto data = source.GetChunkData(index);
  if (!data || data->empty()) {
    ++chunks_dropped_;
    return std::nullopt;
  }

  TrainingChunk chunk;
  chunk.sort_key = sort_key;
  chunk.index_within_sort_key = index;
  chunk.global_index = chunks_processed_;
  chunk.use_count = 0;
  chunk.frames = std::move(*data);

  return chunk;
}

StageMetricProto SimpleChunkExtractor::FlushMetrics() {
  StageMetricProto metric;
  auto add_count = [&](const char* name, std::atomic<uint64_t>& counter) {
    auto* m = metric.add_count_metrics();
    m->set_name(name);
    m->set_count(counter.exchange(0));
  };

  add_count("chunks_processed", chunks_processed_);
  add_count("chunks_dropped", chunks_dropped_);
  add_count("sources_processed", sources_processed_);

  *metric.add_queue_metrics() = MetricsFromQueue("output", *output_queue());
  return metric;
}

}  // namespace training
}  // namespace lczero
