#include "loader/stages/chunk_rescorer.h"

#include <stdexcept>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "chess/board.h"
#include "loader/data_loader_metrics.h"

namespace lczero {
namespace training {

ChunkRescorer::ChunkRescorer(const ChunkRescorerConfig& config,
                             const StageRegistry& existing_stages,
                             RescoreFn rescore_fn)
    : SingleInputStage<ChunkRescorerConfig, InputType>(config, existing_stages),
      syzygy_paths_(config.syzygy_paths()),
      dist_temp_(config.dist_temp()),
      dist_offset_(config.dist_offset()),
      dtz_boost_(config.dtz_boost()),
      new_input_format_(config.new_input_format()),
      output_queue_(config.queue_capacity()),
      thread_pool_(config.threads(), ThreadPoolOptions{}),
      rescore_fn_(std::move(rescore_fn)) {
  if (!rescore_fn_) {
    throw std::invalid_argument("ChunkRescorer requires rescore function");
  }
  static absl::once_flag bitboards_initialized_flag;
  absl::call_once(bitboards_initialized_flag, InitializeMagicBitboards);

  if (config.has_deblunder_threshold() && config.has_deblunder_width()) {
    RescorerDeblunderSetup(config.deblunder_threshold(),
                           config.deblunder_width());
  }

  LOG(INFO) << "Initializing ChunkRescorer with " << config.threads()
            << " worker thread(s) and queue capacity "
            << config.queue_capacity();

  thread_contexts_.reserve(config.threads());
  for (size_t i = 0; i < config.threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
  }
}

ChunkRescorer::~ChunkRescorer() { Stop(); }

Queue<ChunkRescorer::OutputType>* ChunkRescorer::output() {
  return &output_queue_;
}

void ChunkRescorer::InitializeTablebase() {
  if (tablebase_initialized_) {
    return;
  }

  LOG(INFO) << "ChunkRescorer initializing Syzygy tablebase with paths '"
            << syzygy_paths_ << "'.";
  tablebase_initialized_ = tablebase_.init(syzygy_paths_);
  if (tablebase_initialized_) {
    LOG(INFO) << "ChunkRescorer Syzygy max cardinality: "
              << tablebase_.max_cardinality();
  } else {
    LOG(WARNING) << "ChunkRescorer failed to initialize Syzygy tablebase; "
                    "rescoring will continue without tablebase lookups.";
  }
}

void ChunkRescorer::Start() {
  LOG(INFO) << "Starting ChunkRescorer worker threads.";
  InitializeTablebase();
  for (size_t i = 0; i < thread_contexts_.size(); ++i) {
    thread_pool_.Enqueue([this, i]() { Worker(thread_contexts_[i].get()); });
  }
}

void ChunkRescorer::Stop() {
  bool expected = false;
  if (!stop_requested_.compare_exchange_strong(expected, true)) {
    return;
  }

  LOG(INFO) << "Stopping ChunkRescorer.";
  input_queue()->Close();
  output_queue_.Close();
  thread_pool_.WaitAll();
  thread_pool_.Shutdown();
  LOG(INFO) << "ChunkRescorer stopped.";
}

QueueBase* ChunkRescorer::GetOutput(std::string_view name) {
  (void)name;
  return &output_queue_;
}

void ChunkRescorer::Worker(ThreadContext* context) {
  auto producer = output_queue_.CreateProducer();

  try {
    while (true) {
      TrainingChunk chunk = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue()->Get();
      }();

      try {
        chunk.frames = rescore_fn_(chunk.frames, &tablebase_, dist_temp_,
                                   dist_offset_, dtz_boost_, new_input_format_);
        LoadMetricPauser pauser(context->load_metric_updater);
        producer.Put(std::move(chunk));
      } catch (const std::exception& exception) {
        LOG(ERROR) << "ChunkRescorer failed to rescore chunk: "
                   << exception.what() << "; sort_key=" << chunk.sort_key
                   << "; index_within_sort_key=" << chunk.index_within_sort_key
                   << "; global_index=" << chunk.global_index
                   << "; use_count=" << chunk.use_count
                   << "; frame_count=" << chunk.frames.size();
        continue;
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "ChunkRescorer worker stopping, queue closed.";
  }
}

StageMetricProto ChunkRescorer::FlushMetrics() {
  StageMetricProto stage_metric;
  stage_metric.set_stage_type("chunk_rescorer");
  LoadMetricProto aggregated_load;
  aggregated_load.set_name("load");
  for (const auto& context : thread_contexts_) {
    UpdateFrom(aggregated_load, context->load_metric_updater.FlushMetrics());
  }
  *stage_metric.add_load_metrics() = std::move(aggregated_load);
  *stage_metric.add_queue_metrics() = MetricsFromQueue("output", output_queue_);
  return stage_metric;
}

}  // namespace training
}  // namespace lczero
