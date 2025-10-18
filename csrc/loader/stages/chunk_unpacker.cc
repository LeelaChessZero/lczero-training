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

ChunkUnpacker::ChunkUnpacker(const ChunkUnpackerConfig& config,
                             const StageList& existing_stages)
    : SingleInputStage<ChunkUnpackerConfig, InputType>(config, existing_stages),
      position_sampling_rate_(
          config.has_position_sampling_rate()
              ? absl::make_optional(config.position_sampling_rate())
              : absl::nullopt),
      position_count_(config.has_position_count()
                          ? absl::make_optional(config.position_count())
                          : absl::nullopt),
      run_seed_(GenerateRunSeed()),
      output_queue_(config.queue_capacity()),
      thread_pool_(config.threads(), ThreadPoolOptions{}) {
  const bool has_rate = config.has_position_sampling_rate();
  const bool has_count = config.has_position_count();
  CHECK(has_rate != has_count)
      << "Exactly one of position_sampling_rate or position_count must be "
         "set.";
  LOG(INFO) << "Initializing ChunkUnpacker with " << config.threads()
            << " worker threads";

  // Initialize thread contexts but don't start worker threads yet.
  thread_contexts_.reserve(config.threads());
  for (size_t i = 0; i < config.threads(); ++i) {
    thread_contexts_.push_back(std::make_unique<ThreadContext>());
  }
}

ChunkUnpacker::~ChunkUnpacker() { Stop(); }

Queue<ChunkUnpacker::OutputType>* ChunkUnpacker::output() {
  return &output_queue_;
}

QueueBase* ChunkUnpacker::GetOutput(std::string_view name) {
  (void)name;
  return &output_queue_;
}

void ChunkUnpacker::Start() {
  LOG(INFO) << "Starting ChunkUnpacker worker threads.";
  for (size_t i = 0; i < thread_contexts_.size(); ++i) {
    thread_pool_.Enqueue([this, i]() { Worker(thread_contexts_[i].get()); });
  }
}

void ChunkUnpacker::Stop() {
  bool expected = false;
  if (!stop_requested_.compare_exchange_strong(expected, true)) {
    return;
  }

  LOG(INFO) << "Stopping ChunkUnpacker.";
  input_queue()->Close();
  output_queue_.Close();
  thread_pool_.WaitAll();
  thread_pool_.Shutdown();
  LOG(INFO) << "ChunkUnpacker stopped.";
}

void ChunkUnpacker::Worker(ThreadContext* context) {
  // Create a local producer for this worker thread.
  auto producer = output_queue_.CreateProducer();

  try {
    while (true) {
      auto chunk = [&]() {
        LoadMetricPauser pauser(context->load_metric_updater);
        return input_queue()->Get();
      }();

      absl::BitGen gen(
          std::seed_seq{run_seed_, static_cast<uint32_t>(chunk.global_index)});
      std::vector<uint32_t> positions;
      if (position_sampling_rate_) {
        positions = PickSampledPositions(
            static_cast<int32_t>(chunk.frames.size()), *position_sampling_rate_,
            chunk.use_count, gen);
      } else {
        const uint32_t frame_count = static_cast<uint32_t>(chunk.frames.size());
        const uint32_t selection_count =
            std::min(frame_count, *position_count_);
        positions = PickFixedPositionCount(frame_count, selection_count, gen,
                                           chunk.use_count);
      }

      for (uint32_t pos : positions) {
        LoadMetricPauser pauser(context->load_metric_updater);
        producer.Put(std::move(chunk.frames[pos]));
      }
    }
  } catch (const QueueClosedException&) {
    LOG(INFO) << "ChunkUnpacker worker stopping, input queue closed.";
    // Input queue is closed, the local producer will be destroyed when this
    // function exits which may close the output queue if this is the last
    // producer.
  }
}

StageMetricProto ChunkUnpacker::FlushMetrics() {
  StageMetricProto stage_metric;
  stage_metric.set_stage_type("chunk_unpacker");
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
