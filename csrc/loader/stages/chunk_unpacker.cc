#include "loader/stages/chunk_unpacker.h"

#include <absl/algorithm/container.h>
#include <absl/container/flat_hash_set.h>
#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/random/random.h>
#include <absl/random/seed_sequences.h>

#include <algorithm>
#include <random>
#include <utility>
#include <vector>

#include "loader/data_loader_metrics.h"
#include "proto/data_loader_config.pb.h"
#include "proto/training_metrics.pb.h"

namespace lczero {
namespace training {

// Deterministically partitions `n` positions into disjoint subsets of size
// `~p*n`, returning the subset for a given `iteration`. While each selection
// individually behaves like a Bernoulli sample with probability `p`, the
// samples are correlated to ensure all positions are selected exactly once over
// `1/p` iterations. To sample disjoint subsets from the same set of positions,
// `gen` must be seeded identically for each call with a different `iteration`.
std::vector<uint32_t> PickSampledPositions(int32_t n, double p,
                                           int32_t iteration,
                                           absl::BitGen& gen) {
  assert(p > 0.0 && p <= 1.0);
  double carried_prob = p;

  std::vector<uint32_t> result;
  absl::flat_hash_set<int32_t> skip_next_round;

  while (true) {
    int32_t num_this_round = (1.0 - carried_prob) / p + 1;
    double last_partial_prob = 1 - (carried_prob + (num_this_round - 1) * p);
    const bool return_this_round = iteration < num_this_round;
    absl::flat_hash_set<int32_t> skip_this_round(skip_next_round);
    skip_next_round.clear();
    for (int32_t i = 0; i < n; ++i) {
      if (skip_this_round.contains(i)) continue;
      const double toss = absl::Uniform<double>(gen, 0.0, 1.0);
      const int32_t value = (toss - carried_prob) / p + 1;
      if (value == iteration) result.push_back(static_cast<uint32_t>(i));
      if (value >= num_this_round) {
        skip_next_round.insert(static_cast<int32_t>(i));
      }
    }
    if (return_this_round) return result;
    iteration -= num_this_round;
    carried_prob = p - last_partial_prob;
  }
}

// Deterministically partitions `n` positions into disjoint subsets of size
// `k` and returns the subset for a given `iteration`. To obtain disjoint
// subsets from the same set of positions, `gen` must be seeded identically
// for each call with a different `iteration`.
std::vector<uint32_t> PickFixedPositionCount(uint32_t n, uint32_t k,
                                             absl::BitGen& gen,
                                             uint32_t iteration) {
  if (!n || !k) return {};
  const uint32_t actual_k = std::min(n, k);
  std::vector<uint32_t> v(n);
  absl::c_iota(v, 0u);

  std::shuffle(v.begin(), v.end(), gen);

  const uint32_t per = n / actual_k;
  const uint32_t cut = per * actual_k;
  const uint32_t rem = n - cut;

  for (uint32_t r = iteration / per; r--;) {
    std::rotate(v.begin(), v.begin() + cut, v.end());
    std::shuffle(v.begin() + rem, v.end(), gen);
  }

  const uint32_t off = (iteration % per) * actual_k;
  return {v.begin() + off, v.begin() + off + actual_k};
}

namespace {

uint32_t GenerateRunSeed() {
  absl::BitGen gen(absl::MakeSeedSeq());
  return absl::Uniform<uint32_t>(gen);
}

}  // namespace

ChunkUnpacker::ChunkUnpacker(const ChunkUnpackerConfig& config)
    : SingleInputStage<ChunkUnpackerConfig, InputType>(config),
      config_(config),
      run_seed_(GenerateRunSeed()),
      primary_output_queue_(
          config.output().queue_capacity(),
          ToOverflowBehavior(config.output().overflow_behavior())),
      thread_pool_(config.threads(), ThreadPoolOptions{}) {
  const bool has_rate = config.has_position_sampling_rate();
  const bool has_count = config.has_position_count();
  const bool has_prefetch_count = config.has_prefetch_count();
  const bool has_prefetch_output = config.has_prefetch_output();

  CHECK(has_prefetch_count == has_prefetch_output)
      << "prefetch_count and prefetch_output must both be set or both unset.";

  if (has_prefetch_count) {
    CHECK(has_count) << "position_count must be set when using prefetch mode.";
    CHECK(!has_rate)
        << "position_sampling_rate cannot be used in prefetch mode.";
    CHECK(config.position_count() == 1)
        << "position_count must equal 1 in prefetch mode, got "
        << config.position_count();
    prefetch_output_queue_.emplace(
        config.prefetch_output().queue_capacity(),
        ToOverflowBehavior(config.prefetch_output().overflow_behavior()));
    if (config.output().name() == config.prefetch_output().name()) {
      throw std::runtime_error(
          absl::StrCat("ChunkUnpacker output names must be different, got: '",
                       config.output().name(), "'"));
    }
  } else {
    CHECK(has_rate != has_count)
        << "Exactly one of position_sampling_rate or position_count must be "
           "set.";
  }

  LOG(INFO) << "Initializing ChunkUnpacker with " << config.threads()
            << " worker threads";

  // Initialize thread contexts but don't start worker threads yet.
  thread_contexts_.reserve(config.threads());
  for (size_t i = 0; i < config.threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
  }
}

ChunkUnpacker::~ChunkUnpacker() { Stop(); }

void ChunkUnpacker::Start() {
  LOG(INFO) << "Starting ChunkUnpacker worker threads.";
  for (size_t i = 0; i < thread_contexts_.size(); ++i) {
    thread_pool_.Enqueue([this, i](std::stop_token stop_token) {
      Worker(stop_token, thread_contexts_[i].get());
    });
  }
}

void ChunkUnpacker::Stop() {
  if (thread_pool_.stop_token().stop_requested()) return;

  LOG(INFO) << "Stopping ChunkUnpacker.";
  thread_pool_.Shutdown();
  primary_output_queue_.Close();
  if (prefetch_output_queue_) prefetch_output_queue_->Close();
  LOG(INFO) << "ChunkUnpacker stopped.";
}

QueueBase* ChunkUnpacker::GetOutput(std::string_view name) {
  if (name == config_.output().name()) return &primary_output_queue_;
  if (config_.has_prefetch_output() &&
      name == config_.prefetch_output().name()) {
    return &*prefetch_output_queue_;
  }
  std::string available = absl::StrCat("'", config_.output().name(), "'");
  if (config_.has_prefetch_output()) {
    absl::StrAppend(&available, ", '", config_.prefetch_output().name(), "'");
  }
  throw std::runtime_error(absl::StrCat("ChunkUnpacker unknown output '", name,
                                        "'. Available outputs: ", available));
}

void ChunkUnpacker::Worker(std::stop_token stop_token, ThreadContext* context) {
  // Create a local producer for this worker thread.
  auto primary_producer = primary_output_queue_.CreateProducer();
  std::optional<decltype(prefetch_output_queue_->CreateProducer())>
      prefetch_producer;
  if (prefetch_output_queue_.has_value()) {
    prefetch_producer.emplace(prefetch_output_queue_->CreateProducer());
  }

  try {
    while (true) {
      auto chunk = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue()->Get(stop_token);
      }();

      absl::BitGen gen(
          std::seed_seq{run_seed_, static_cast<uint32_t>(chunk.global_index)});
      std::vector<uint32_t> positions;
      if (config_.has_position_sampling_rate()) {
        positions = PickSampledPositions(
            static_cast<int32_t>(chunk.frames.size()),
            config_.position_sampling_rate(), chunk.use_count, gen);
      } else if (config_.has_prefetch_count()) {
        const uint32_t frame_count = static_cast<uint32_t>(chunk.frames.size());
        const uint32_t total_count =
            config_.position_count() + config_.prefetch_count();
        const uint32_t selection_count = std::min(frame_count, total_count);
        positions = PickFixedPositionCount(frame_count, selection_count, gen,
                                           chunk.use_count);
      } else {
        const uint32_t frame_count = static_cast<uint32_t>(chunk.frames.size());
        const uint32_t selection_count =
            std::min(frame_count, config_.position_count());
        positions = PickFixedPositionCount(frame_count, selection_count, gen,
                                           chunk.use_count);
      }

      if (config_.has_prefetch_count()) {
        // Prefetch mode: output first position to primary, rest to prefetch.
        if (!positions.empty()) {
          LoadMetricPauser pauser(context->load_metric_updater);
          primary_producer.Put(std::move(chunk.frames[positions[0]]),
                               stop_token);
        }
        if (positions.size() > 1) {
          CacheRequest cache_request;
          cache_request.global_index = chunk.global_index;
          cache_request.next_use = chunk.use_count + 1;
          cache_request.items.reserve(positions.size() - 1);
          for (size_t i = 1; i < positions.size(); ++i) {
            cache_request.items.push_back(chunk.frames[positions[i]]);
          }
          LoadMetricPauser pauser(context->load_metric_updater);
          prefetch_producer->Put(std::move(cache_request), stop_token);
        }
      } else {
        // Normal mode: output all positions to primary.
        for (uint32_t pos : positions) {
          LoadMetricPauser pauser(context->load_metric_updater);
          primary_producer.Put(std::move(chunk.frames[pos]), stop_token);
        }
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "ChunkUnpacker worker stopping, queue closed.";
  } catch (const QueueRequestCancelled&) {
    LOG(INFO) << "ChunkUnpacker worker stopping, request cancelled.";
  }
}

StageMetricProto ChunkUnpacker::FlushMetrics() {
  StageMetricProto stage_metric;
  LoadMetricProto aggregated_load;
  aggregated_load.set_name("load");
  for (const auto& context : thread_contexts_) {
    UpdateFrom(aggregated_load, context->load_metric_updater.FlushMetrics());
  }
  *stage_metric.add_load_metrics() = std::move(aggregated_load);
  *stage_metric.add_queue_metrics() =
      MetricsFromQueue(config_.output().name(), primary_output_queue_);
  if (prefetch_output_queue_.has_value()) {
    *stage_metric.add_queue_metrics() = MetricsFromQueue(
        config_.prefetch_output().name(), *prefetch_output_queue_);
  }
  return stage_metric;
}

}  // namespace training
}  // namespace lczero
