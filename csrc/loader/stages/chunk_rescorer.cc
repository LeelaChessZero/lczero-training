#include "loader/stages/chunk_rescorer.h"

#include <stdexcept>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "chess/board.h"
#include "loader/data_loader_metrics.h"

namespace lczero {
namespace training {
namespace {

void V6ToV7(std::span<FrameType> data, float theta = 5.0f / 6.0f) {
  if (data.empty()) return;

  const float beta = 1.0f - theta;

  float st_q = 0.0f;
  float st_d = 0.0f;

  // Iterate backwards to calculate EMA and lookaheads in one go.
  for (size_t i = data.size(); i-- > 0;) {
    FrameType& item = data[i];

    // For Q, we operate in an alternating sign domain.
    const float sign = (i % 2 != 0) ? -1.0f : 1.0f;
    const float cur_q = item.root_q * sign;

    if (i == data.size() - 1) {
      st_q = cur_q;
      st_d = item.root_d;
    } else {
      st_q = theta * st_q + beta * cur_q;
      st_d = theta * st_d + beta * item.root_d;
    }

    item.version = 7;
    item.q_st = st_q * sign;
    item.d_st = std::max(st_d, 0.0f);

    // Handle lookahead indices safely.
    auto get_idx = [&](size_t offset) {
      return (i + offset < data.size()) ? data[i + offset].played_idx : 65535;
    };
    item.opp_played_idx = get_idx(1);
    item.next_played_idx = get_idx(2);
  }
}

}  // namespace

ChunkRescorer::ChunkRescorer(const ChunkRescorerConfig& config)
    : SingleInputStage<ChunkRescorerConfig, InputType>(config),
      SingleOutputStage<OutputType>(config.output()),
      syzygy_paths_(config.syzygy_paths()),
      dist_temp_(config.dist_temp()),
      dist_offset_(config.dist_offset()),
      dtz_boost_(config.dtz_boost()),
      new_input_format_(config.new_input_format()),
      thread_pool_(config.threads(), ThreadPoolOptions{}),
      st_q_theta_(config.st_q_theta()) {
  static absl::once_flag bitboards_initialized_flag;
  absl::call_once(bitboards_initialized_flag, InitializeMagicBitboards);

  if (config.has_deblunder_threshold() && config.has_deblunder_width()) {
    RescorerDeblunderSetup(config.deblunder_threshold(),
                           config.deblunder_width());
  }

  if (config.has_gaviota_paths()) {
    RescorerGaviotaSetup(std::string(config.gaviota_paths()));
  }

  LOG(INFO) << "Initializing ChunkRescorer with " << config.threads()
            << " worker thread(s)";

  thread_contexts_.reserve(config.threads());
  for (size_t i = 0; i < config.threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
  }
}

ChunkRescorer::~ChunkRescorer() { Stop(); }

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
    thread_pool_.Enqueue([this, i](std::stop_token stop_token) {
      Worker(stop_token, thread_contexts_[i].get());
    });
  }
}

void ChunkRescorer::Stop() {
  if (thread_pool_.stop_token().stop_requested()) return;

  LOG(INFO) << "Stopping ChunkRescorer.";
  thread_pool_.Shutdown();
  output_queue()->Close();
  LOG(INFO) << "ChunkRescorer stopped.";
}

void ChunkRescorer::Worker(std::stop_token stop_token, ThreadContext* context) {
  auto producer = output_queue()->CreateProducer();

  try {
    while (true) {
      TrainingChunk chunk = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue()->Get(stop_token);
      }();

      try {
        chunk.frames = RescoreTrainingData<FrameType>(
            chunk.frames, &tablebase_, dist_temp_, dist_offset_, dtz_boost_,
            new_input_format_);
        if (chunk.frames.at(0).version == 6) V6ToV7(chunk.frames, st_q_theta_);
        LoadMetricPauser pauser(context->load_metric_updater);
        producer.Put(std::move(chunk), stop_token);
      } catch (const std::exception& exception) {
        failed_rescores_.fetch_add(1, std::memory_order_acq_rel);
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
  } catch (const QueueRequestCancelled&) {
    LOG(INFO) << "ChunkRescorer worker stopping, request cancelled.";
  }
}

StageMetricProto ChunkRescorer::FlushMetrics() {
  StageMetricProto stage_metric;
  LoadMetricProto aggregated_load;
  aggregated_load.set_name("load");
  for (const auto& context : thread_contexts_) {
    UpdateFrom(aggregated_load, context->load_metric_updater.FlushMetrics());
  }
  *stage_metric.add_load_metrics() = std::move(aggregated_load);

  auto* failed = stage_metric.add_count_metrics();
  failed->set_name("failed_rescores");
  failed->set_count(failed_rescores_.exchange(0, std::memory_order_acq_rel));

  *stage_metric.add_queue_metrics() =
      MetricsFromQueue("output", *output_queue());
  return stage_metric;
}

}  // namespace training
}  // namespace lczero
