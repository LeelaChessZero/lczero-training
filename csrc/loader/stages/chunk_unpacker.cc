#include "loader/stages/chunk_unpacker.h"

#include <absl/algorithm/container.h>
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
namespace {
// Deal the i-th block of size k from a shuffled 0..n−1, rotating leftovers
// forward and reshuffling the rest between rounds.
std::vector<uint32_t> ShuffledBlock(uint32_t n, uint32_t k, uint64_t run_seed,
                                    uint64_t chunk_seed, uint32_t i) {
  if (!n || !k) return {};
  std::vector<uint32_t> v(n);
  absl::c_iota(v, 0u);

  absl::BitGen gen(std::seed_seq{static_cast<uint32_t>(run_seed),
                                 static_cast<uint32_t>(run_seed >> 32),
                                 static_cast<uint32_t>(chunk_seed),
                                 static_cast<uint32_t>(chunk_seed >> 32)});
  std::shuffle(v.begin(), v.end(), gen);

  const uint32_t per = n / k;    // full K-blocks per round
  const uint32_t cut = per * k;  // position after dealing one round
  const uint32_t rem = n - cut;  // leftovers kept at front

  // Jump over whole rounds: rotate leftovers to front, reshuffle the tail each
  // time.
  for (uint32_t r = i / per; r--;) {
    std::rotate(v.begin(), v.begin() + cut, v.end());
    std::shuffle(v.begin() + rem, v.end(), gen);  // reshuffle suffix
  }

  const uint32_t off = (i % per) * k;  // block offset within the current round
  return {v.begin() + off, v.begin() + off + k};
}

uint64_t GenerateRunSeed() {
  absl::BitGen gen(absl::MakeSeedSeq());
  return absl::Uniform<uint64_t>(gen);
}

}  // namespace

ChunkUnpacker::ChunkUnpacker(const ChunkUnpackerConfig& config,
                             const StageList& existing_stages)
    : SingleInputStage<ChunkUnpackerConfig, InputType>(config, existing_stages),
      position_sampling_rate_(config.position_sampling_rate()),
      run_seed_(GenerateRunSeed()),
      output_queue_(config.queue_capacity()),
      thread_pool_(config.threads(), ThreadPoolOptions{}) {
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

      size_t positions_to_sample =
          round(chunk.frames.size() * position_sampling_rate_);
      if (positions_to_sample == 0) positions_to_sample = 1;
      std::vector<uint32_t> positions = ShuffledBlock(
          chunk.frames.size(), positions_to_sample, run_seed_,
          static_cast<uint64_t>(chunk.global_index), chunk.reshuffle_count);

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
